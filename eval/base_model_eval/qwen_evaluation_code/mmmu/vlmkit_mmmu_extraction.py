import math
import random
import ast
import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import string

# Local imports from refactored files
from dataset_utils import load_dataset, dump_image, MMMU_preproc
from eval_utils import build_judge, eval_single_sample

# import sglang as sgl
# from sglang.test.test_utils import is_in_ci

from vllm import LLM, SamplingParams
from PIL import Image
import base64
from io import BytesIO

from transformers import AutoProcessor, AutoTokenizer
from utils import prepare_question_array_with_base64_image_strings, prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm
import dotenv

dotenv.load_dotenv()

# if not is_in_ci():
#     import nest_asyncio

#     nest_asyncio.apply()

if "MIT_SPIDER_TOKEN" in os.environ:
    print(os.environ["MIT_SPIDER_TOKEN"][:5])
else:
    print("MIT_SPIDER_TOKEN not found")
    exit()

os.environ["MIT_SPIDER_URL"] = "https://api.openai.com/v1/chat/completions"

def run_evaluation(args):
    """Run evaluation on inference results."""
    # Load results
    results = []
    with open(args.input_file, 'r') as f:
        data = json.load(f)
        for job in data:
            annotation = job["annotation"]
            annotation["prediction"] = job["raw_full_prediction"]
            results.append(annotation)
            
    data = pd.DataFrame.from_records(results)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    # If not choice label, then use lower case
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    # Load dataset
    meta = load_dataset(args.dataset)

    # data中data.iloc[i]中的index必须在results中存在，对应results中的results[i]['id']，并且data中data.iloc[i]中的question必须和results中results[i]['annotation']中的question完全一致
    print(f"len(data): {len(data)}")
    print(f"len(meta): {len(meta)}")
    meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
    data_map = {x: y for x, y in zip(data['index'], data['question'])}
    for k in data_map:
        assert k in meta_q_map, (
            "eval_file should be the same as or a subset of dataset MMMU_DEV_VAL"
        )

    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    data = MMMU_preproc(data)
    answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}
    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]
    items = []
    for i in range(len(data)):
        item = data.iloc[i]
        items.append(item)

    # Build judge model if needed
    model = None
    model = build_judge(args.eval_model, args.api_type)
    
    # Prepare evaluation tasks
    eval_tasks = []
    for item in items:
        eval_tasks.append((model, item))
    
    # Run evaluation
    eval_results = []
    
    # Debug mode: process single-threaded with first few samples
    debug = False
    if debug:
        print("Running in debug mode with first 5 samples...")
        # for task in tqdm(eval_tasks[:5], desc="Evaluating"):
        for task in eval_tasks[:5]:
            try:
                result = eval_single_sample(task)
                eval_results.append(result)
            except Exception as e:
                print(f"Error processing task: {e}")
                print(f"Task details: {task}")
                raise
    else:
        # Normal mode: process all samples with threading
        with ThreadPoolExecutor(max_workers=args.nproc) as executor:
            for result in tqdm(executor.map(eval_single_sample, eval_tasks), 
                             total=len(eval_tasks), desc="Evaluating"):
                eval_results.append(result)
    
    # Calculate overall accuracy
    accuracy = sum(r['hit'] for r in eval_results) / len(eval_results)
    
    # Calculate accuracy by split
    results_by_split = {}
    for result in eval_results:
        split = result.get('split', 'unknown')
        if split not in results_by_split:
            results_by_split[split] = []
        results_by_split[split].append(result)
    
    accuracy_by_split = {}
    for split, split_results in results_by_split.items():
        split_accuracy = sum(r['hit'] for r in split_results) / len(split_results)
        accuracy_by_split[split] = split_accuracy
        print(f"Accuracy for {split} split: {split_accuracy:.4f} ({sum(r['hit'] for r in split_results)}/{len(split_results)})")
    
    # Save results
    output_df = pd.DataFrame(eval_results)
    output_df.to_csv(args.output_file, index=False)
    
    # Save accuracy
    with open(args.output_file.replace('.csv', '_acc.json'), 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "accuracy_by_split": accuracy_by_split
        }, f, indent=2)
    
    # print(f"Evaluation completed. Overall accuracy: {accuracy:.4f}")
    print(f"Results saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="MMMU Evaluation Script")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, default="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/qwen-evaluation/mmmu/data", help="The absolute path of MMMU_DEV_VAL.tsv")
    eval_parser.add_argument("--input-file", type=str, required=True, help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    eval_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    eval_parser.add_argument("--eval-model", type=str, default="gpt-3.5-turbo-0125", 
                            choices=["gpt-3.5-turbo-0125","gpt-4-0125-preview", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"],
                            help="Model to use for evaluation")
    eval_parser.add_argument("--api-type", type=str, default="dash", choices=["dash", "mit"],
                            help="API type to use for evaluation")
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of processes to use")
    
    args = parser.parse_args()

    os.environ['LMUData'] = args.data_dir
    
    if args.mode == "infer":
        # run_inference(args)
        raise ValueError("Inference mode is not supported for this script")
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
