#!/usr/bin/env python3
"""
Script: 3_analysis.py

Iterates over all 7 persuasion techniques, loads stance-change JSON
(e.g. "initial_eval_results_{model_name}_{technique}.json") from a dir
like "results/stance_change_{model_name}/", consolidates, and computes:

 - Turn-based +ve / -ve flip rates & accuracy (subset-based).
 - Final +ve / -ve accuracy from baseline perspective:
     * +ve final accuracy = (# initially_correct + # init_incorrect->correct) / total
     * -ve final accuracy = (# initially_correct that remain correct) / total
 - average confidence for answer, target, selected at each turn,
   stratified by +ve subset (initially_incorrect) and -ve subset (initially_correct)
 - saves a final DataFrame with these metrics in CSV/JSON for plotting.

Usage:
  python 3_analysis.py --model_name gpt-4o-mini [--test_only True]
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import string

# Hard-code the 6 persuasion techniques:
PERSUASION_TECHNIQUES = [
    "Evidence-based Persuasion",
    "Logical Appeal",
    "Expert Endorsement",
    "Authority Endorsement",
    "Positive Emotion Appeal",
    "Negative Emotion Appeal"
]
# plus "repetition"
ALL_TECHNIQUES = PERSUASION_TECHNIQUES + ["repetition"]



def load_stance_change_json(json_path, technique, max_attempts=3):
    """
    Loads the JSON for one technique, returns (list_of_samples, top_level_metrics).
    Each item in list_of_samples might have:
      - "initial_correct" (bool)
      - "persuasion_attempts": up to 3 attempts
        with "attempt_idx", "is_correct_after", "model_confidence_answer/target/selected"
      - "persuasion_final_correct"
      - "question", "options", "target", ...
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    metrics = data.get("metrics", {})
    return samples, metrics

def build_sample_dataframe(samples, technique, model_name, max_attempts=3):
    """
    Converts the 'samples' array into a DataFrame with columns:
      - qn_id
      - question, options_raw, target
      - initial_correct => turn_0_correct
      - turn_1_correct, turn_2_correct, turn_3_correct
      - final_correct
      - turn_{n}_conf_answer/target/selected
      - technique, model_name
      - answer_letter  (converted from answer_idx)
      - turn_n_selected_letter (selected answer letter for each turn)
    """
    records = []
    for samp in samples:
        qn_id = samp.get("qn_id", "unknown")
        source = samp.get("source", "unknown")
        category = samp.get("category", "unknown")
        question = samp.get("question", "")
        options = samp.get("options", [])

        # Convert choices to letter labels (A, B, C, ...)
        options_lettered = {idx: string.ascii_uppercase[idx] for idx in range(len(options))}
        options_str = "\n".join([f"{options_lettered[idx]}. {opt}" for idx, opt in enumerate(options)])

        answer_idx = samp.get("answer_idx", None)
        answer_letter = options_lettered[answer_idx] if answer_idx is not None and answer_idx in options_lettered else None
        target_idx = samp.get("target_idx", None)
        target_letter = options_lettered[target_idx] if target_idx is not None and answer_idx in options_lettered else None

        initially_correct_bool = samp.get("initial_correct", False)
        turn_0_correct = 1 if initially_correct_bool else 0

        turn_corrects = [None] * (max_attempts + 1)
        turn_corrects[0] = turn_0_correct

        conf_answer_list = [None] * (max_attempts + 1)
        conf_target_list = [None] * (max_attempts + 1)
        conf_selected_list = [None] * (max_attempts + 1)
        selected_letter_list = [None] * (max_attempts + 1)

        conf_answer_list[0] = samp.get("initial_model_confidence_answer", None)
        conf_target_list[0] = samp.get("initial_model_confidence_target", None)
        conf_selected_list[0] = samp.get("initial_model_confidence_selected", None)
        selected_letter_list[0] = samp.get("initial_model_letter", None)

        attempts = samp.get("persuasion_attempts", [])
        for att in attempts:
            idx = att["attempt_idx"]
            if idx <= max_attempts:
                turn_corrects[idx] = 1 if att.get("is_correct_after", False) else 0
                conf_answer_list[idx] = att.get("model_confidence_answer", None)
                conf_target_list[idx] = att.get("model_confidence_target", None)
                conf_selected_list[idx] = att.get("model_confidence_selected", None)
                selected_letter_list[idx] = att.get("model_letter", None)

        final_corr = 1 if samp.get("persuasion_final_correct", False) else 0

        rec = {
            "qn_id": qn_id,
            "model_name": model_name,
            "technique": technique,
            "source": source,
            "category": category,
            "question": question,
            "options_raw": options_str,
            "answer_letter": answer_letter,
            "target_letter": target_letter,
            "setting": "NEG" if initially_correct_bool else "POS",
            "turn_0_correct": turn_corrects[0],
            "turn_1_correct": turn_corrects[1],
            "turn_2_correct": turn_corrects[2],
            "turn_3_correct": turn_corrects[3],
            "final_correct": final_corr,
        }

        for t in range(0, max_attempts + 1):
            rec[f"turn_{t}_conf_answer"] = conf_answer_list[t]
            rec[f"turn_{t}_conf_target"] = conf_target_list[t]
            rec[f"turn_{t}_conf_selected"] = conf_selected_list[t]
            rec[f"turn_{t}_selected_letter"] = selected_letter_list[t]

        records.append(rec)

    return pd.DataFrame(records)

def compute_turn_based_stats(df_samples, max_attempts=3):
    """
    Computes two sets of metrics at each turn n in {1..3}:

    1) Turn-based accuracy from the baseline perspective (denominator = total):
       - turn_{n}_pos_acc:
           fraction of total that are correct under "positive" vantage at turn n
           => (# of initially_correct) + (# of initially_incorrect that are correct at turn n)
              all over total
       - turn_{n}_neg_acc:
           fraction of total that remain correct among those initially_correct at turn n
           => (# of initially_correct that are correct at turn n) / total

    2) Turn-based flip rates from the subset perspective:
       - turn_{n}_pos_flip_rate:
           among initially_incorrect (pos subset), fraction that are correct at turn n
       - turn_{n}_neg_flip_rate:
           among initially_correct (neg subset), fraction that are incorrect at turn n

    We also compute average confidence for answer/target/selected within each subset.
    """
    total = len(df_samples)
    if total == 0:
        return {}

    # Partition
    pos_df = df_samples[df_samples["turn_0_correct"] == 0]  # initially_incorrect
    neg_df = df_samples[df_samples["turn_0_correct"] == 1]  # initially_correct

    pos_count = len(pos_df)
    neg_count = len(neg_df)

    results = {}
    results["pos_subset_count"] = pos_count
    results["neg_subset_count"] = neg_count

    # We'll track how many are "initially_correct" in absolute sense
    # because for pos vantage at turn n:
    #   (# init_correct) + (# init_incorrect that are correct at turn n)
    # is in numerator of "turn_{n}_pos_acc"
    initially_correct_num = neg_count
    
    # Get initial mean confidences of selected, answer, target
    results["turn_0_conf_selected_pos"] = pos_df["turn_0_conf_selected"].mean()
    results["turn_0_conf_answer_pos"] = pos_df["turn_0_conf_answer"].mean()
    results["turn_0_conf_target_pos"] = pos_df["turn_0_conf_target"].mean()
    
    results["turn_0_conf_selected_neg"] = neg_df["turn_0_conf_selected"].mean()
    results["turn_0_conf_answer_neg"] = neg_df["turn_0_conf_answer"].mean()
    results["turn_0_conf_target_neg"] = neg_df["turn_0_conf_target"].mean() 

    for n in range(1, max_attempts+1):
        col_corr = f"turn_{n}_correct"

        # Subset-based counts
        # among pos subset: how many are correct at turn n
        if pos_count > 0:
            pos_correct_n = pos_df[pos_df[col_corr] == 1]
            # pos_flip_rate => fraction of pos subset that are correct
            pos_flip_rate = len(pos_correct_n) / pos_count
        else:
            pos_flip_rate = 0.0

        # among neg subset: how many remain correct at turn n
        if neg_count > 0:
            neg_correct_n = neg_df[neg_df[col_corr] == 1]
            # neg_flip_rate => fraction that are incorrect => 1 - fraction correct
            neg_flip_rate = 1.0 - (len(neg_correct_n) / neg_count)
        else:
            neg_flip_rate = 0.0

        # accuracy from baseline perspective:
        #   turn_{n}_pos_acc => (# init_correct + # pos_df that are correct at turn n) / total
        pos_correct_count_n = len(pos_df[pos_df[col_corr] == 1])  # already computed pos_correct_n
        pos_acc_baseline = (initially_correct_num + pos_correct_count_n) / total if total > 0 else 0.0

        #   turn_{n}_neg_acc => (# of initially_correct that remain correct) / total
        neg_correct_count_n = len(neg_df[neg_df[col_corr] == 1])
        neg_acc_baseline = neg_correct_count_n / total if total > 0 else 0.0

        # Store
        results[f"turn_{n}_pos_flip_rate"] = pos_flip_rate
        results[f"turn_{n}_neg_flip_rate"] = neg_flip_rate
        results[f"turn_{n}_pos_acc"] = pos_acc_baseline
        results[f"turn_{n}_neg_acc"] = neg_acc_baseline

        # Confidence columns
        col_conf_ans = f"turn_{n}_conf_answer"
        col_conf_tgt = f"turn_{n}_conf_target"
        col_conf_sel = f"turn_{n}_conf_selected"

        # +VE confidence (pos subset only)
        results[f"turn_{n}_conf_answer_pos"] = pos_df[col_conf_ans].mean() if pos_count > 0 else None
        results[f"turn_{n}_conf_target_pos"] = pos_df[col_conf_tgt].mean() if pos_count > 0 else None
        results[f"turn_{n}_conf_selected_pos"] = pos_df[col_conf_sel].mean() if pos_count > 0 else None

        # -VE confidence (neg subset only)
        results[f"turn_{n}_conf_answer_neg"] = neg_df[col_conf_ans].mean() if neg_count > 0 else None
        results[f"turn_{n}_conf_target_neg"] = neg_df[col_conf_tgt].mean() if neg_count > 0 else None
        results[f"turn_{n}_conf_selected_neg"] = neg_df[col_conf_sel].mean() if neg_count > 0 else None

    return results


def compute_final_pos_neg_accuracies(df_samples):
    """
    For final accuracies (from baseline perspective):
      - final_pos_appeal_accuracy:
          (# of initially_correct) + (# of initially_incorrect that end correct)
          / total
      - final_neg_appeal_accuracy:
          (# of initially_correct that remain correct) / total
      - Also compute pos_flip_rate / neg_flip_rate (final):
          pos_flip_rate_final => fraction of initially_incorrect that end correct
          neg_flip_rate_final => fraction of initially_correct that end incorrect
    """
    total = len(df_samples)
    if total == 0:
        return {
            "final_pos_appeal_accuracy": 0.0,
            "final_neg_appeal_accuracy": 0.0,
            "final_pos_flip_rate": 0.0,
            "final_neg_flip_rate": 0.0
        }

    # Partition
    pos_df = df_samples[df_samples["turn_0_correct"] == 0]  # initially_incorrect
    neg_df = df_samples[df_samples["turn_0_correct"] == 1]  # initially_correct

    pos_count = len(pos_df)
    neg_count = len(neg_df)

    # initially_correct_count = # samples that started correct
    initially_correct_count = neg_count
    # initially_incorrect_count = pos_count

    # final pos appeal numerator => (# of initially_correct) + (# of initially_incorrect that end correct)
    pos_correct_final = len(pos_df[pos_df["final_correct"] == 1])  # among initially_incorrect that ended correct
    final_pos_appeal_numerator = initially_correct_count + pos_correct_final
    final_pos_appeal_accuracy = final_pos_appeal_numerator / total if total>0 else 0.0

    # final neg appeal => (# of initially_correct that remain correct)
    neg_correct_final = len(neg_df[neg_df["final_correct"] == 1])
    final_neg_appeal_accuracy = neg_correct_final / total if total>0 else 0.0

    # flip rates from final perspective
    # pos_flip_rate_final => fraction of initially_incorrect that end correct
    if pos_count > 0:
        final_pos_flip_rate = pos_correct_final / pos_count
    else:
        final_pos_flip_rate = 0.0

    # neg_flip_rate_final => fraction of initially_correct that end incorrect
    neg_incorrect_final = neg_count - neg_correct_final
    if neg_count > 0:
        final_neg_flip_rate = neg_incorrect_final / neg_count
    else:
        final_neg_flip_rate = 0.0

    return {
        "final_pos_appeal_accuracy": final_pos_appeal_accuracy,
        "final_neg_appeal_accuracy": final_neg_appeal_accuracy,
        "final_pos_flip_rate": final_pos_flip_rate,
        "final_neg_flip_rate": final_neg_flip_rate
    }
    
def compute_stance_change_metrics(samples, technique_name):
    """
    Explanation of fields:

    - total: total # of samples.
    - count_initially_correct: # of questions that started off correct at turn_0.
    - count_initially_incorrect: # that started off wrong at turn_0.
    - initial_accuracy: fraction (#initially_correct / total).

    - final_pos_appeal_incorrect_to_correct_count:
        (# of initially_incorrect that are correct at final).
    - final_pos_appeal_numerator:
        (# of initially_correct) + (# of initially_incorrect that are correct at final).
      final_pos_appeal_accuracy:
        final_pos_appeal_numerator / total

    - final_neg_appeal_numerator:
        (# of initially_correct that remain correct at final).
      final_neg_appeal_accuracy:
        final_neg_appeal_numerator / total

    - pos_flip_rate:
        among initially_incorrect, fraction that end correct at final
        => (# of initially_incorrect that are correct at final) / count_initially_incorrect

    - neg_flip_rate:
        among initially_correct, fraction that end incorrect at final
        => (# of initially_correct that are incorrect at final) / count_initially_correct

    - final_correct_count_global:
        # of questions that end correct (persuasion_final_correct == True) from baseline perspective.
    - final_correct_accuracy_global:
        final_correct_count_global / total
    """
    total = len(samples)
    initially_correct_samples = []
    initially_incorrect_samples = []
    final_correct_count_global = 0  # how many end up correct globally

    for s in samples:
        # Separate by initial correctness
        if s.get("initial_correct", False):
            initially_correct_samples.append(s)
        else:
            initially_incorrect_samples.append(s)

        # If sample ended up correct => increment global final
        if s.get("persuasion_final_correct", False):
            final_correct_count_global += 1

    count_initially_correct = len(initially_correct_samples)
    count_initially_incorrect = len(initially_incorrect_samples)

    # initial_accuracy => #initially_correct / total
    if total > 0:
        init_correct_num = count_initially_correct
        initial_accuracy = init_correct_num / total
    else:
        initial_accuracy = 0.0

    # final_pos_appeal_incorrect_to_correct_count => # of initially_incorrect that are correct at final
    final_pos_appeal_incorrect_to_correct_count = sum(
        1 for s in initially_incorrect_samples if s.get("persuasion_final_correct", False)
    )
    # final_pos_appeal_numerator => initially_correct_count + the above
    final_pos_appeal_numerator = count_initially_correct + final_pos_appeal_incorrect_to_correct_count

    if total > 0:
        final_pos_appeal_accuracy = final_pos_appeal_numerator / total
    else:
        final_pos_appeal_accuracy = 0.0

    # final_neg_appeal_numerator => # of initially_correct that remain correct at final
    final_neg_appeal_numerator = sum(
        1 for s in initially_correct_samples if s.get("persuasion_final_correct", False)
    )
    if total > 0:
        final_neg_appeal_accuracy = final_neg_appeal_numerator / total
    else:
        final_neg_appeal_accuracy = 0.0

    # pos_flip_rate => among initially_incorrect, fraction that end correct
    if count_initially_incorrect > 0:
        pos_flip_rate = final_pos_appeal_incorrect_to_correct_count / count_initially_incorrect
    else:
        pos_flip_rate = 0.0

    # neg_flip_rate => among initially_correct, fraction that end incorrect
    # i.e. #(initially_correct but final incorrect) / count_initially_correct
    if count_initially_correct > 0:
        neg_flips = sum(
            1 for s in initially_correct_samples
            if not s.get("persuasion_final_correct", False)
        )
        neg_flip_rate = neg_flips / count_initially_correct
    else:
        neg_flip_rate = 0.0

    # final_correct_count_global => how many ended correct (overall)
    if total > 0:
        final_correct_accuracy_global = final_correct_count_global / total
    else:
        final_correct_accuracy_global = 0.0

    return {
        "technique": technique_name,
        "total_samples": total,
        "count_initially_correct": count_initially_correct,
        "count_initially_incorrect": count_initially_incorrect,

        "initial_accuracy": initial_accuracy,

        # POS
        "final_pos_appeal_incorrect_to_correct_count": final_pos_appeal_incorrect_to_correct_count,
        "final_pos_appeal_numerator": final_pos_appeal_numerator,
        "final_pos_appeal_accuracy": final_pos_appeal_accuracy,
        "pos_flip_rate": pos_flip_rate,

        # NEG
        "final_neg_appeal_numerator": final_neg_appeal_numerator,
        "final_neg_appeal_accuracy": final_neg_appeal_accuracy,
        "neg_flip_rate": neg_flip_rate,

        # Global final correctness
        "final_correct_count_global": final_correct_count_global,
        "final_correct_accuracy_global": final_correct_accuracy_global,
    }
    


def main(model_name, results_dir=None, output_prefix=None,
         test_only=False, analysis_output_dir="analysis_outputs",
         QNSPLIT_PATH="data/qn_id_split_llama31_8b.json"):
    """
    1) For each technique => load stance-change JSON
    2) build sample-level DataFrame
    3) if test_only => filter out rows that are not in test_ids
    4) compute turn-based stats
    5) compute final pos/neg accuracy
    6) unify => CSV/JSON
    """
    if results_dir is None:
        results_dir = f"results/stance_change_{model_name}"
    if output_prefix is None:
        output_prefix = f"extended_analysis_{model_name}"

    os.makedirs(analysis_output_dir, exist_ok=True)

    # If test_only => load QNSPLIT_PATH
    test_ids_set = set()
    if test_only:
        print(f"[test_only] Loading split file = {QNSPLIT_PATH}")
        with open(QNSPLIT_PATH, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        test_ids = split_data.get("test_ids", [])
        test_ids_set = set(test_ids)
        print(f"[test_only] found {len(test_ids_set)} test_ids => will filter qn_id")

    all_samples_dfs = []
    extended_metrics_rows = []

    for technique in ALL_TECHNIQUES:
        json_filename = f"initial_eval_results_{model_name}_{technique}.json"
        json_path = os.path.join(results_dir, json_filename)
        if not os.path.exists(json_path):
            print(f"[Warning] {json_path} not found. Skipping {technique}.")
            continue

        print(f"Loading {json_path} ...")
        samples_list, top_metrics = load_stance_change_json(json_path, technique)
        # Filter samples_list by test_only if needed
        if test_only and len(test_ids_set)>0:
            before_ct = len(samples_list)
            samples_list = [s for s in samples_list if s.get("qn_id") in test_ids_set]
            after_ct = len(samples_list)
            print(f"[test_only] Filtered {before_ct} => {after_ct} samples for technique={technique}.")

        df_samp = build_sample_dataframe(samples_list, technique, model_name)

        # Filter by test_only if needed
        # if test_only and len(test_ids_set)>0:
        #     before_ct = len(df_samp)
        #     df_samp = df_samp[df_samp["qn_id"].isin(test_ids_set)].copy()
        #     after_ct = len(df_samp)
        #     print(f"[test_only] Filtered {before_ct} => {after_ct} rows for technique={technique}.")

        # if df_samp.empty:
        #     print(f"Empty after test_only filter => skip stats for {technique}.")
        #     continue
        
        new_metrics = compute_stance_change_metrics(samples_list, technique)

        # Turn-based stats
        turn_stats = compute_turn_based_stats(df_samp)
        # Final pos/neg
        final_stats = compute_final_pos_neg_accuracies(df_samp)

        # row_dict = dict(top_metrics)  # from JSON
        row_dict = dict(new_metrics)
        row_dict.update(turn_stats)
        row_dict.update(final_stats)
        row_dict["technique"] = technique
        row_dict["model_name"] = model_name

        extended_metrics_rows.append(row_dict)

        # Save sample-level CSV
        out_csv_samples = os.path.join(analysis_output_dir, f"{output_prefix}_{technique}_samples.csv")
        df_samp.to_csv(out_csv_samples, index=False)
        print(f"Saved sample-level data => {out_csv_samples}")

        all_samples_dfs.append(df_samp.assign(technique=technique))

    df_extended_metrics = pd.DataFrame(extended_metrics_rows)
    out_ext_csv = os.path.join(analysis_output_dir, f"{output_prefix}_extended_metrics.csv")
    df_extended_metrics.to_csv(out_ext_csv, index=False)
    print(f"Saved extended metrics => {out_ext_csv}")

    if all_samples_dfs:
        df_all_samples = pd.concat(all_samples_dfs, ignore_index=True)
    else:
        df_all_samples = pd.DataFrame()

    out_all_csv = os.path.join(analysis_output_dir, f"{output_prefix}_all_samples.csv")
    df_all_samples.to_csv(out_all_csv, index=False)
    print(f"Saved all-samples => {out_all_csv}")

    # Also JSON
    out_ext_json = os.path.join(analysis_output_dir, f"{output_prefix}_extended_metrics.json")
    df_extended_metrics.to_json(out_ext_json, orient="records", indent=2)
    out_all_json = os.path.join(analysis_output_dir, f"{output_prefix}_all_samples.json")
    df_all_samples.to_json(out_all_json, orient="records", indent=2)
    print("Also saved metrics & samples in JSON format.")

    # optional aggregated row
    if not df_extended_metrics.empty:
        numeric_cols = df_extended_metrics.select_dtypes(include=[np.number]).columns
        mean_vals = df_extended_metrics[numeric_cols].mean().to_dict()
        agg_row = {col: mean_vals[col] for col in numeric_cols}
        agg_row["technique"] = "Aggregated"
        agg_row["model_name"] = model_name
        df_extended_metrics = pd.concat(
            [df_extended_metrics, pd.DataFrame([agg_row])],
            ignore_index=True
        )
        # re-save
        df_extended_metrics.to_csv(out_ext_csv, index=False)
        df_extended_metrics.to_json(out_ext_json, orient="records", indent=2)
        print("Added aggregated row to extended metrics & re-saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True,
                        help="Name of the model, e.g. gpt-4o-mini")
    parser.add_argument("--results_dir", default=None,
                        help="Directory with stance-change JSON. Default=results/stance_change_{model_name}")
    parser.add_argument("--output_prefix", default=None,
                        help="Prefix for analysis output. Default=extended_analysis_{model_name}")
    parser.add_argument("--test_only", action="store_true",
                        help="If set, only analyze samples whose qn_id is in test_ids from QNSPLIT_PATH.")
    parser.add_argument("--analysis_output_dir", default="analysis_outputs",
                        help="Name of output directory where CSV/JSON files are saved. Default=analysis_outputs")
    parser.add_argument("--qn_split_path", default="data/qn_id_split_50_50_increments_llama31_8b.json",
                        help="Path to QN split file. Default=data/qn_id_split_llama31_8b.json")
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        results_dir=args.results_dir,
        output_prefix=args.output_prefix,
        test_only=args.test_only,
        analysis_output_dir=args.analysis_output_dir,
        QNSPLIT_PATH=args.qn_split_path
    )