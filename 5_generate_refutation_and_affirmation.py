#!/usr/bin/env python3
"""
5_generate_refutation_and_affirmation.py

Reads JSON files from a stance-change evaluation folder. Each file has the form:
{
  "samples": [...],
  "metrics": {...}
}

For each sample in 'samples':
  - If sample["initial_correct"] == True => NEG => generate 'refutations' 
    for the negative appeals (the persuasion attempts).
  - If sample["initial_correct"] == False => POS => generate 'affirmations'
    for the positive appeals (the persuasion attempts).
We store these new texts in each attempt's "intended_response".

Example usage:
  python 5_generate_refutation_and_affirmation.py \
    --input_dir results/stance_change_Llama-3.1-8B-Instruct \
    --output_dir results/stance_change_Llama-3.1-8B-Instruct_refuted_and_affirmed \
    --debug
"""

import os
import json
import argparse
import time
import string
from tqdm.auto import tqdm
from dotenv import load_dotenv

import prompts
import utils  # your asynchronous batch approach
from openai import OpenAI


###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing stance-change result JSONs, each with {samples, metrics}.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where we will store new JSONs with refutations/affirmations.")
    parser.add_argument("--debug", action="store_true",
                        help="If set, only process a small subset of samples from each file.")
    parser.add_argument("--debug_samples", type=int, default=10,
                        help="How many samples per file to process if debug is set.")
    args = parser.parse_args()

    load_dotenv()
    API_KEY = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=API_KEY)

    os.makedirs(args.output_dir, exist_ok=True)

    all_tasks = []
    custom_id_map = []
    # We'll keep a structure: file_data_map = { filename: main_data }, 
    # where main_data = { "samples": [...], "metrics": {...} }
    # so that after we get the results, we can insert them and then write out the final file.
    file_data_map = {}

    def add_task(file_id, sample_idx, attempt_idx, custom_id, prompt_text):
        """
        Build a single request object for openai batch usage.
        We store (file_id, sample_idx, attempt_idx) => to place the result later.
        """
        body = {
            "model": "gpt-4o-mini",
            "temperature": 1.0,
            "messages": [
                {"role": "user", "content": prompt_text}
            ]
        }
        all_tasks.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        })
        # We'll store a mapping => so we can place results later
        custom_id_map.append((file_id, sample_idx, attempt_idx, custom_id))

    # 1) gather tasks from each JSON
    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    for jf in json_files:
        filepath = os.path.join(args.input_dir, jf)
        with open(filepath, "r", encoding="utf-8") as f:
            main_data = json.load(f)
        # main_data has { "samples": [...], "metrics": {...} }
        samples = main_data["samples"]
        if args.debug:
            samples = samples[:args.debug_samples]

        # We'll store back the partial or entire list in main_data["samples"]
        main_data["samples"] = samples
        file_data_map[jf] = main_data

        # Now for each sample => check initial_correct => build tasks
        for i, sample in enumerate(samples):
            # We reconstruct the question text from initial_prompt minus trailing newline "Answer:"
            original_prompt = sample.get("initial_prompt", "")
            question_text = original_prompt.rsplit("\nAnswer:", 1)[0].strip()
            # We'll check the persuasion attempts
            attempts = sample.get("persuasion_attempts", [])

            if sample.get("initial_correct", False):
                # This is the NEG scenario => we want to create "refutations" for the negative appeals
                # The "correct" answer is sample["answer_idx"] => letter, sample["answer"] => text
                idx = sample["answer_idx"]
                letter = string.ascii_uppercase[idx]
                correct_text = sample["answer"]
                answer_str = f"{letter}: {correct_text}"

                # For each attempt => we have an "appeal_text" in attempt["persuasion_user_text"]
                for att_i, attempt in enumerate(attempts):
                    appeal_text = attempt.get("persuasion_user_text", "")
                    # Build the prompt
                    prompt_text = prompts.generate_neg_refutation(
                        question_text=question_text,
                        answer_text=answer_str,
                        appeal_text=appeal_text
                    )
                    cid = f"refute_{jf}_{i}_{att_i}"
                    add_task(file_id=jf, sample_idx=i, attempt_idx=att_i,
                             custom_id=cid, prompt_text=prompt_text)

            else:
                # This is the POS scenario => we want to create "affirmations" for the positive appeals
                # The "incorrect" answer => sample["target_idx"], sample["target"]
                idx = sample["target_idx"]
                letter = string.ascii_uppercase[idx]
                incorrect_text = sample["target"]
                inc_ans = f"{letter}: {incorrect_text}"

                for att_i, attempt in enumerate(attempts):
                    appeal_text = attempt.get("persuasion_user_text", "")
                    prompt_text = prompts.generate_pos_affirmation(
                        question_text=question_text,
                        incorrect_answer=inc_ans,
                        appeal_text=appeal_text
                    )
                    cid = f"affirm_{jf}_{i}_{att_i}"
                    add_task(file_id=jf, sample_idx=i, attempt_idx=att_i,
                             custom_id=cid, prompt_text=prompt_text)

    # 2) split into 2 halves, run async
    all_count = len(all_tasks)
    half = all_count // 2
    tasks1 = all_tasks[:half]
    tasks2 = all_tasks[half:]

    print(f"Collected {all_count} tasks from {len(json_files)} files. Splitting => {len(tasks1)} vs {len(tasks2)}")

    if all_count == 0:
        print("No tasks found, nothing to do.")
        return

    print("Submitting batch 1 ...")
    batch1 = utils.submit_batch(tasks1, "refute_affirm_batch1", client)
    batch_id1 = batch1.id

    print("Submitting batch 2 ...")
    batch2 = utils.submit_batch(tasks2, "refute_affirm_batch2", client)
    batch_id2 = batch2.id

    # poll
    done1 = False
    done2 = False
    while True:
        if not done1:
            job1 = client.batches.retrieve(batch_id1)
            if job1.status == "completed":
                print(f"Batch job {batch_id1} completed.")
                done1 = True
            elif job1.status == "failed":
                print(f"Batch job {batch_id1} failed.")
                done1 = True
            else:
                if hasattr(job1, 'request_counts'):
                    c1 = job1.request_counts.completed
                    t1 = job1.request_counts.total
                    print(f"[{batch_id1}] {c1}/{t1} done.")
        if not done2:
            job2 = client.batches.retrieve(batch_id2)
            if job2.status == "completed":
                print(f"Batch job {batch_id2} completed.")
                done2 = True
            elif job2.status == "failed":
                print(f"Batch job {batch_id2} failed.")
                done2 = True
            else:
                if hasattr(job2, 'request_counts'):
                    c2 = job2.request_counts.completed
                    t2 = job2.request_counts.total
                    print(f"[{batch_id2}] {c2}/{t2} done.")
        if done1 and done2:
            break
        time.sleep(30)

    # retrieve
    results1 = []
    results2 = []
    job1 = client.batches.retrieve(batch_id1)
    if job1.status == "completed":
        results1 = utils.retrieve_batch_results(batch_id1, client, batch_name="refute_affirm_batch1")
    job2 = client.batches.retrieve(batch_id2)
    if job2.status == "completed":
        results2 = utils.retrieve_batch_results(batch_id2, client, batch_name="refute_affirm_batch2")

    results = results1 + results2
    print(f"Merged total {len(results)} completions from 2 batches.")

    # 3) build a map from custom_id => text
    response_map = {}
    for r in results:
        cid = r["custom_id"]
        body = r.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if choices:
            final_text = choices[0]["message"]["content"].strip()
        else:
            final_text = "[No response]"
        response_map[cid] = final_text

    # 4) place them into the attempts => "intended_response"
    for (cid_info, attempt_cid) in zip(custom_id_map, [x["custom_id"] for x in all_tasks]):
        # cid_info => (file_id, sample_idx, attempt_idx, custom_id)
        # attempt_cid => the actual custom_id
        # we can just rely on cid_info[-1] or attempt_cid
        file_id, samp_idx, att_idx, cid_val = cid_info
        if cid_val in response_map:
            final_text = response_map[cid_val]
            # retrieve that sample from file_data_map
            main_data = file_data_map[file_id]
            sample = main_data["samples"][samp_idx]
            # find the relevant attempt
            # we place it in attempt["intended_response"]
            attempts = sample.get("persuasion_attempts", [])
            # We assume att_idx is valid
            if att_idx < len(attempts):
                attempts[att_idx]["intended_response"] = final_text

    # 5) write out each file
    for jf, main_data in file_data_map.items():
        # main_data has { "samples": [...], "metrics": {...} }
        out_fname = jf.replace(".json", "_refuted_affirmed.json")
        out_path = os.path.join(args.output_dir, out_fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(main_data, f, ensure_ascii=False, indent=2)
        print(f"Wrote updated file => {out_path}")

if __name__ == "__main__":
    main()
