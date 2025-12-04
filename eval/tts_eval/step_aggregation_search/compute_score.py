import os
import math
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter

import argparse

def is_answer_match(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a.lower() == b.lower(): return True
    try: return abs(float(a) - float(b)) < 1e-10
    except: return False

def load_json(file_path):
    """
    Load a JSON file and return its contents as a Python object (dict or list).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Extract Answer from candidates"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        # default="/home/ubuntu/porialab-us-midwest-1/Tej/mmr-eval/traces_data",
        required = True,
        help="Path to step traces",
    )

    args = parser.parse_args()

    root_dir = Path(args.data_dir)

    for idx, path in enumerate(root_dir.rglob("*.json")):

        print(f"{idx}: {path}")

        # filename = path.name

        data = load_json(path)

        maj_score = 0
        step_agg_score = 0
        total = 0

        for sidx, sample in enumerate(tqdm(data)):
            gt_answer = sample["annotation"]["answer"]
            # if len(sample['iteration_history']) < 1:
            #     continue
            # candidates = sample['iteration_history'][0]['candidates_info']
            # candidate_preds = [candidate["pred_answer"] for candidate in candidates]
            
            # maj_counter = Counter(candidate_preds)
            # majority_string, count = maj_counter.most_common(1)[0]

            # sample['majority'] = {
            #     "majority_string": majority_string,
            #     "majority_count": count
            # }

            # step_agg_chosen_idx = sample['step_agg']["chosen_candidate"]
            chosen_pred = sample["pred_answer"]

            # sample['step_agg']["chosen_pred_answer"] = step_agg_chosen_pred

            # if is_answer_match(gt_answer, majority_string):
            #     maj_score += 1

            if is_answer_match(gt_answer, chosen_pred):
                step_agg_score += 1

            total += 1
        if data and "step_agg" in data[0] and data[0]["step_agg"]:
            print(f"Step Aggregate Score: {step_agg_score}/{total}", step_agg_score/total)
        elif data and "non_greedy" in data[0] and data[0]["non_greedy"]:
            print(f"Non-Greedy Score: {step_agg_score}/{total}", step_agg_score/total)
        else:
            raise ValueError(f"No TTS type found for sample {sidx}")
        # print(f"Majority Score: {maj_score}/{total}", maj_score/total)

        # break


if __name__ == "__main__":
    main()

