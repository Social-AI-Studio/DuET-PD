import os
import json
import time
import string
import math
import re
from tqdm.auto import tqdm
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests

def chat_completions_create(
    model="gpt-4o-mini",
    messages=None,
    temperature=0.0,
    max_tokens=1,
    logprobs=True,
    top_logprobs=20,
    use_local=False,
    api_key=None
):
    """
    A minimal function calling an OpenAI-like chat completion endpoint.
    If base_url is provided, it might be a local endpoint. Otherwise, use the normal API key approach.
    """
    load_dotenv()
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY', '')

    client = None
    if use_local:
        client = OpenAI(
        # api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
        )
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
    else:
        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
    return completion



def submit_and_retrieve_batch(tasks, batch_name, client, output_dir="batch_results"):
    """
    Submits the tasks to a batch endpoint, polls until completion,
    then retrieves the results from the server. Saves intermediate
    and final results into the specified output directory.

    Parameters:
        tasks (list): A list of tasks prepared by prepare_batch_prompts.
        batch_name (str): A name to uniquely identify the batch job.
        client: A hypothetical client object to submit the batch with. Replace
                with your own logic or OpenAI code as needed.
        output_dir (str): The directory where batch files and results are saved.
                          Default is 'batches'.

    Returns:
        results (list of dict): The raw JSON results from the batch job.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save tasks to a JSONL file within the output directory
    jsonl_file_path = os.path.join(output_dir, f"{batch_name}_batch_tasks.jsonl")
    with open(jsonl_file_path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

    # Create or upload the batch file
    batch_file = client.files.create(
        file=open(jsonl_file_path, "rb"),
        purpose="batch"
    )

    # Submit the batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        completion_window="24h",
        endpoint="/v1/chat/completions",
    )

    print(f"Submitted batch job: {batch_job.id}")

    # Poll for job completion
    while True:
        batch_job = client.batches.retrieve(batch_job.id)
        if batch_job.status == "completed":
            print("Batch job completed.")
            break
        elif batch_job.status == "failed":
            print("Batch job failed.")
            return None
        else:
            # Print progress if available
            if hasattr(batch_job, 'request_counts'):
                completed_count = batch_job.request_counts.completed
                total_count = batch_job.request_counts.total
                print(f"({completed_count}/{total_count}) completed. Waiting...")
            else:
                print("Batch job not completed yet. Waiting...")
            time.sleep(30)  # Wait 30 seconds before checking again

    # Retrieve results
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content

    # Save results to a JSONL file within the output directory
    result_path = os.path.join(output_dir, f"{batch_name}_batch_results.jsonl")
    with open(result_path, 'wb') as file:
        file.write(result)

    # Parse each line of the results file
    results = []
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    return results


###############################################################################
# The asynchronous batch functions
###############################################################################

def submit_batch(tasks, batch_name, client):
    """
    Submits tasks to the batch endpoint, returns the batch_job object (which includes an id).
    Does NOT wait for completion.
    """
    import uuid
    import os

    output_dir = "batch_results"
    os.makedirs(output_dir, exist_ok=True)

    jsonl_file_path = os.path.join(output_dir, f"{batch_name}_batch_tasks.jsonl")
    with open(jsonl_file_path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

    # Create or upload the batch file
    batch_file = client.files.create(
        file=open(jsonl_file_path, "rb"),
        purpose="batch"
    )

    # Submit the batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        completion_window="24h",
        endpoint="/v1/chat/completions",
    )
    print(f"Submitted batch job: {batch_job.id}")
    return batch_job  # We'll store the job object or at least job id

def poll_batch_until_done(job_id, client, poll_interval=30):
    """
    Poll for job completion, returns final status ("completed" or "failed").
    Does NOT retrieve the results. If completed, we can call retrieve_batch_results.
    """
    while True:
        batch_job = client.batches.retrieve(job_id)
        if batch_job.status == "completed":
            print(f"Batch job {job_id} completed.")
            return "completed"
        elif batch_job.status == "failed":
            print(f"Batch job {job_id} failed.")
            return "failed"
        else:
            # Print progress if available
            if hasattr(batch_job, 'request_counts'):
                completed_count = batch_job.request_counts.completed
                total_count = batch_job.request_counts.total
                print(f"[{job_id}] {completed_count}/{total_count} completed. Waiting...")
            else:
                print(f"[{job_id}] not completed yet. Waiting...")
            time.sleep(poll_interval)

def retrieve_batch_results(job_id, client, batch_name="batch_part", output_dir="batch_results"):
    """
    After the job is completed, download the results from the batch job's output_file_id.
    Returns a list of dict results.
    """
    os.makedirs(output_dir, exist_ok=True)

    batch_job = client.batches.retrieve(job_id)
    result_file_id = batch_job.output_file_id
    if not result_file_id:
        print(f"No result file for job {job_id}. Possibly failed.")
        return []

    result = client.files.content(result_file_id).content
    result_path = os.path.join(output_dir, f"{batch_name}_batch_results.jsonl")
    with open(result_path, 'wb') as file:
        file.write(result)

    # Parse each line
    results = []
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results
###############################################################################

def parse_single_letter_answer(content: str):
    """
    Extract a single uppercase letter from the content.
    e.g. content might say: "A"
    or "Best answer: B"
    We'll return the letter if found, else None.
    """
    match = re.search(r"\b([A-Z])\b", content)
    if match:
        return match.group(1)
    return None


def compute_letter_confidence(letter_answer, options, logprobs_raw):
    """
    Compute the model's normalized confidence for a selected letter among a set of MCQ options.

    Key points:
      - We look at the *last* token's top_logprobs from logprobs_raw["content"].
      - We only consider the relevant letters corresponding to 'options' (A, B, C, etc.).
      - If the model provided a letter (e.g. 'J') not in the set (A, B, C, D),
        its confidence is effectively 0 if we strictly only
        normalize over the valid letters. Or, if letter_answer is e.g. 'J', 
        the function will return 0.0 because 'J' is not in letter_probs.
      - This function returns 0.0 if data is missing or the letter is not found.

    Parameters:
        letter_answer (str): The single-letter answer chosen by the model (e.g., "A").
        options (list[str]): The list of multiple-choice options, e.g. ["Apple", "Banana", "Carrot"]
        logprobs_raw (dict): The logprobs structure from the model's response, e.g.:
          {
            "content": [
              {
                "token": " ... ",
                "logprob": -X.XX,
                "top_logprobs": [
                  {"token": "A", "logprob": -0.1},
                  ...
                ]
              },
              ... # Possibly multiple token logs
            ]
          }

    Returns:
        float: Normalized confidence for the selected letter (range 0.0 ~ 1.0).
    """

    content_info = logprobs_raw.content


    # 2) Usually the single-letter answer is in the LAST token (the model's final output)
    last_token_info = content_info[-1]
    top_logprobs = last_token_info.top_logprobs
    if not top_logprobs or not isinstance(top_logprobs, list):
        print(f"Invalid top_logprobs data: {top_logprobs}")
        return 0.0
    # 3) Build a map: token -> logprob
    token_to_logprob = {}
    for item in top_logprobs:
        tk = item.token
        lp = item.logprob
        if tk is not None and lp is not None:
            token_to_logprob[tk] = lp

    # 4) Figure out the valid letters for the MCQ (A, B, C, D, etc.)
    #    We'll only consider these letters in the distribution.
    letters = list(string.ascii_uppercase[:len(options)])

    # 5) For each letter, sum up the exponentiated logprob of letter and " letter"
    #    e.g. letter='A' => check tokens: "A" and " A"
    letter_scores = {}
    sum_scores = 0.0

    for letter in letters:
        variants = [letter, f" {letter}"]
        letter_score = 0.0
        for var in variants:
            if var in token_to_logprob:
                letter_score += math.exp(token_to_logprob[var])
        letter_scores[letter] = letter_score
        sum_scores += letter_score

    # 6) Convert to probabilities
    if sum_scores > 0:
        letter_probs = {
            letter: letter_scores[letter] / sum_scores for letter in letters
        }
    else:
        # If sum_scores=0, everything is 0
        letter_probs = {letter: 0.0 for letter in letters}

    # 7) Return the selected letter's confidence
    return letter_probs.get(letter_answer, 0.0)


def compute_options_confidence(options, logprobs_raw):
    """
    Given the full logprobs (top_logprobs) from the model's first token
    and the MCQ 'options' list,
    returns a dict: { "A": confidence_for_A, "B": ..., ... }.
    """
    import string
    letter_labels = string.ascii_uppercase[:len(options)]
    
    letter_conf = {}
    for letter in letter_labels:
        conf = compute_letter_confidence(
            letter_answer=letter,
            options=options,
            logprobs_raw=logprobs_raw
        )
        letter_conf[letter] = conf
        
    return letter_conf


def fallback_to_highest_conf(letter_conf_dict):
    """
    letter_conf_dict is e.g. {"A": 0.4, "B": 0.1, "C": 0.5, ...}
    Return the letter (e.g. "C") with the maximum confidence.
    """
    if not letter_conf_dict:
        return None
    best_letter = max(letter_conf_dict, key=letter_conf_dict.get)
    return best_letter



