#!/usr/bin/env python3
"""
Script: 1_initial_eval.py

Performs a initial multiple-choice (MCQ) evaluation using an OpenAI-style Chat Completion.
Each sample in the dataset is queried once without any persuasion context.
Outputs a new JSON file containing correctness and confidence information.
"""

import os
import json
import argparse
import re
import string
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Your own modules
import prompts  # Contains ask_mcq_question(question_text, options)
import utils    # Contains compute_letter_confidence(letter_answer, options, logprobs_raw)

# Example: from openai import OpenAI
# For demonstration, we'll show a mock approach. Replace with your actual client or import.
# from openai import OpenAI

def run_initial_evaluation(
    input_json,
    output_json,
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=1,
    top_logprobs=20,
    debug=False,
    debug_samples=20,
    use_local=False,
    system_prompt_type="",
):
    """
    1) Loads the dataset from 'input_json'.
    2) For each sample, build an MCQ prompt using prompts.ask_mcq_question.
    3) Calls the model via an OpenAI-style chat completion.
    4) Parses single-letter answer => check correctness => compute confidence => store results.
       Confidence fields:
         - initial_model_confidence_selected
         - initial_model_confidence_answer
         - initial_model_confidence_target
    5) Writes updated dataset to 'output_json'.

    If debug=True, only processes the first `debug_samples` samples to confirm pipeline correctness.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if debug:
        dataset = dataset[:debug_samples]
        print(f"[DEBUG] Only processing first {debug_samples} samples...")

    for i, sample in enumerate(tqdm(dataset, desc="Initial Evaluation")):
        question_text = sample["question"]
        options = sample["options"]
        correct_idx = sample["answer_idx"]  # ground-truth index

        # Build the prompt
        prompt_text = prompts.ask_mcq_question(question_text, options)
        sample["initial_prompt"] = prompt_text
        
        if "system_prompt_type" == "cautious":
            # Prepare messages
            messages = [
                {"role": "system", "content": prompts.cautious_asst_system_prompt()},
                {"role": "user", "content": prompt_text}
            ]
        else:
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]

        # Attempt model call
        try:
            completion = utils.chat_completions_create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=top_logprobs,
                use_local=use_local
            )
        except Exception as e:
            print(f"[Error on sample {i}]: {e}")
            sample["initial_error"] = str(e)
            sample["initial_correct"] = False
            sample["initial_model_answer"] = ""
            sample["initial_model_name"] = model_name
            # Zero out confidence fields
            sample["initial_model_confidence_selected"] = 0.0
            sample["initial_model_confidence_answer"] = 0.0
            sample["initial_model_confidence_target"] = 0.0
            assert(False)  # Stop here
            continue

        # Extract text
        if completion and completion.choices:
            content = completion.choices[0].message.content
            letter_answer = utils.parse_single_letter_answer(content)
            sample["initial_model_answer"] = content
            sample["initial_model_letter"] = letter_answer
            sample["initial_model_name"] = model_name

            # We only fetch the logprobs to compute confidence
            logprobs_raw = completion.choices[0].logprobs
        else:
            letter_answer = None
            sample["initial_model_answer"] = ""
            sample["initial_model_letter"] = None
            sample["initial_model_name"] = model_name
            logprobs_raw = None
            
        sample["initial_valid_char"] = True  # we'll override if invalid
        sample["initial_confidence_distribution"] = {}

        # Check if letter is valid
        # (1) parse letter_answer if not None => e.g. chosen_idx = ord(letter_answer) - ord("A")
        # (2) if chosen_idx out of range => fallback
        chosen_idx = None
        if letter_answer is not None:
            chosen_idx = ord(letter_answer) - ord("A")

        options_len = len(options)
        if (letter_answer is None) or (chosen_idx < 0) or (chosen_idx >= options_len):
            # invalid => fallback
            sample["initial_valid_char"] = False
            
            # Build distribution
            if logprobs_raw is not None:
                dist = utils.compute_options_confidence(options, logprobs_raw)
                sample["initial_confidence_distribution"] = dist
                # fallback letter
                fallback_letter = utils.fallback_to_highest_conf(dist)
                letter_answer = fallback_letter
            else:
                # no logprobs => can't fallback meaningfully
                letter_answer = None
        else:
            # letter is valid => we can still store distribution
            if logprobs_raw is not None:
                dist = utils.compute_options_confidence(options, logprobs_raw)
                sample["initial_confidence_distribution"] = dist
            else:
                sample["initial_confidence_distribution"] = {}

        # store the final letter after fallback if needed
        sample["initial_model_letter"] = letter_answer

        # Now compute correctness
        if letter_answer is not None:
            chosen_idx = ord(letter_answer) - ord("A")
            sample["initial_correct"] = (chosen_idx == correct_idx)
        else:
            sample["initial_correct"] = False

        # Initialize confidence fields
        sample["initial_model_confidence_selected"] = 0.0
        sample["initial_model_confidence_answer"] = 0.0
        sample["initial_model_confidence_target"] = 0.0

        # If we have a selected letter & logprobs, compute confidence
        if letter_answer and logprobs_raw:
            # 1) Confidence of the letter the model ended up selecting
            conf_sel = utils.compute_letter_confidence(
                letter_answer=letter_answer,
                options=options,
                logprobs_raw=logprobs_raw
            )
            sample["initial_model_confidence_selected"] = conf_sel

            # 2) Confidence for the correct letter (answer_idx)
            correct_letter = string.ascii_uppercase[correct_idx]
            conf_ans = utils.compute_letter_confidence(
                letter_answer=correct_letter,
                options=options,
                logprobs_raw=logprobs_raw
            )
            sample["initial_model_confidence_answer"] = conf_ans

            # 3) Confidence for target_idx (if present)
            if "target_idx" in sample and sample["target_idx"] is not None:
                t_idx = sample["target_idx"]
                if 0 <= t_idx < len(options):
                    t_letter = string.ascii_uppercase[t_idx]
                    conf_t = utils.compute_letter_confidence(
                        letter_answer=t_letter,
                        options=options,
                        logprobs_raw=logprobs_raw
                    )
                    sample["initial_model_confidence_target"] = conf_t

        if debug:
            print(f"[Sample {i}] letter_answer={letter_answer}, correct={sample['initial_correct']}")
            print("Conf(selected, answer, target) =>",
                  sample["initial_model_confidence_selected"],
                  sample["initial_model_confidence_answer"],
                  sample["initial_model_confidence_target"])

    # Save
    # for the purpose of saving, model name should only be the stem; eg meta-llama/Llama-3.1-8B-Instruct --> Llama-3.1-8B-Instruct; split by "/"
    model_name = model_name.split("/")[-1]
    # Optionally append model name to output filename
    output_json = output_json.replace(".json", f"_{model_name}.json")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Summarize
    total = len(dataset)
    correct_count = sum(1 for s in dataset if s.get("initial_correct") == True)
    print(f"Initial accuracy: {correct_count}/{total} = {correct_count/total:.1%}")
    print(f"Results written to {output_json}.")

###########################################
# CLI invocation
###########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to input dataset JSON.")
    parser.add_argument("--output_json", required=True, help="Path to save output initial results.")
    parser.add_argument("--model_name", default="gpt-4o-mini", help="Name of the model.")
    parser.add_argument("--use_local", action="store_true", help="If set, uses local model instead of OpenAI.")
    parser.add_argument("--temperature", default=0.0, type=float, help="Sampling temperature.")
    parser.add_argument("--max_tokens", default=1, type=int, help="Max tokens in the response.")
    parser.add_argument("--top_logprobs", default=20, type=int, help="Number of top tokens for logprobs.")
    parser.add_argument("--debug", action="store_true", help="If set, only processes first N samples (default=20).")
    parser.add_argument("--debug_samples", default=5, type=int, help="Number of samples to process in debug mode.")
    parser.add_argument("--system_prompt_type", default="", help="System prompt to use; options: ['cautious']")
    args = parser.parse_args()
    
    if args.system_prompt_type == "cautious":
        print(f"Using cautious system prompt")
    else:
        print(f"Using default system prompt")

    run_initial_evaluation(
        input_json=args.input_json,
        output_json=args.output_json,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
        debug=args.debug,
        debug_samples=args.debug_samples,
        use_local=args.use_local,
        system_prompt_type=args.system_prompt_type
    )