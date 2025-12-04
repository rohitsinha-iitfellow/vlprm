"""
MMMU-style judge evaluation for BoN (Best-of-N) search results.
Adapts the evaluation pipeline from qwen-evaluation/mmmu/ to work with BoN output format.

Can be run standalone:
    python qwen_collate_final_evaluation.py --input results.json --output evaluation_results.csv
    python qwen_collate_final_evaluation.py --input /scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation/reward_guided_search/outputs/VisualPRM-7B/sr_Q7B_mc0_visualprm_data_normal_tok_full_non_balanced_v2_p100_vit_trained_20250820_095203/MMMU_DEV_VAL-results/result-p0-0-225-20250822_135643.json --nproc 200
"""

import os
import requests
import time
import random
import string
import copy
import traceback
import pandas as pd
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import base64
import io
import argparse
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
# from qwen_evaluation.mmmu.common_utils import download_file, md5, toliststr, decode_base64_to_image_file

if "MIT_SPIDER_TOKEN" in os.environ:
    print(os.environ["MIT_SPIDER_TOKEN"][:5])
else:
    print("MIT_SPIDER_TOKEN not found")
    exit()  

os.environ["MIT_SPIDER_URL"] = "https://api.openai.com/v1/chat/completions"

def toliststr(s):
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError

def load_dataset(dataset_name='MMMU_DEV_VAL'):
    """Load the MMMU dataset."""
    # data_root = os.path.join(os.environ['LMUData'])
    # os.makedirs(data_root, exist_ok=True)
    
    # file_name = f"{dataset_name}.tsv"
    # data_path = os.path.join(data_root, file_name)
    
    # Download if not exists or MD5 doesn't match
    # if not os.path.exists(data_path) or md5(data_path) != MMMU_DATASET_MD5:
    #     print(f"Downloading {dataset_name} dataset...")
    #     download_file(MMMU_DATASET_URL, data_path)

    data_path = "/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/qwen-evaluation/mmmu/data/MMMU_DEV_VAL.tsv" 
    # Load the dataset
    data = pd.read_csv(data_path, sep="\t")
    data = data[data["split"] != "dev"]
    
    # Process the dataset
    data['index'] = [str(x) for x in data['index']]
    
    # Handle image data
    if 'image' in data:
        data['image'] = [str(x) for x in data['image']]
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]

        images = [toliststr(image_map[k]) for k in data['index']]
        data['image'] = [x[0] if len(x) == 1 else x for x in images]
    
    # Handle image paths
    if 'image_path' in data:
        paths = [toliststr(x) for x in data['image_path']]
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
    
    # Convert index to int if possible
    if np.all([isinstance(x, int) or x.isdigit() for x in data['index']]):
        data['index'] = [int(x) for x in data['index']]
    
    return data

# Utility functions copied from qwen-evaluation/mmmu/common_utils.py
def encode_image_to_base64(image, target_size=None):
    """Encode an image to base64 string."""
    if target_size is not None:
        width, height = image.size
        # Resize the image while maintaining the aspect ratio
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        image = image.resize((new_width, new_height))
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# Evaluation wrapper classes copied from qwen-evaluation/mmmu/eval_utils.py
class OpenAIWrapper:
    """Wrapper for OpenAI API."""
    
    def __init__(self, model, api_base, api_key, timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.retry = retry
        self.wait = wait
        self.fail_msg = 'Failed to obtain answer via API.'
    
    def generate(self, messages):
        """Generate a response from the API."""
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        
        # Format messages for API
        formatted_messages = []
        for msg in messages:
            if msg['type'] == 'text':
                formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg['value']}]})
            elif msg['type'] == 'image':
                # Load and encode the image
                image = Image.open(msg['value'])
                image_data = encode_image_to_base64(image)
                formatted_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                })
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": 4096,
            "temperature": 0
        }
        
        for i in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    return resp_json['choices'][0]['message']['content'].strip()
                
                time.sleep(self.wait)
            except Exception as e:
                print(f"API error: {e}")
                time.sleep(self.wait)
        
        return self.fail_msg


def build_judge(model, api_type):
    """Build a judge model for evaluation."""
    if api_type == 'mit':
        api_key = os.environ.get('MIT_SPIDER_TOKEN', '')
        api_base = os.environ.get('MIT_SPIDER_URL', '')
        return OpenAIWrapper(model, api_base, api_key)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")


def can_infer_option(answer, choices):
    """Rule-based extraction of answer option."""
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


def can_infer_text(answer, choices):
    """Extract answer by matching text content."""
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    """Combined approach to infer answer choice."""
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def build_choices(item):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret


def build_option_str(option_dict):
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s


def build_prompt(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)


def extract_answer_from_item(model, item, wait=5):
    """Extract answer from model prediction using rule-based and model-based approaches."""
    # Build choices dictionary
    choices = build_choices(item)
    option_str = build_option_str(choices)
    prompt = build_prompt(item['question'], option_str, item['prediction'])
    
    # Try rule-based extraction first
    prediction = item['prediction']
    ret = can_infer(prediction, choices)
    
    if ret:
        if ret == 'Z':
            extract_flag = False
            log = f"Rule extract failed with rule result: {ret} prediction: {prediction}"
        else:
            extract_flag = True
            log = f"Rule extract success with rule result: {ret} prediction: {prediction}"
        return dict(opt=ret, log=log, extract_model='rule', extract_flag=extract_flag)
    
    # If rule-based extraction fails, use model-based extraction
    print("Rule extract failed. Use model-based extraction.")
    if model is None:
       assert model is not None, 'Judge model is None for MMMU_DEV_VAL !!!'
    
    # Try model-based extraction with retries
    retry = 25
    while retry:
        ans = model.generate([{"type": "text", "value": prompt}])
        if 'Failed to obtain answer via API' in ans:
            print('API failed to answer.')
        else:
            ret = can_infer(ans, choices)
            if ret and ret != 'Z':
                log = f'{model.model} extract Succeed. {model.model}:{ans}\n'
                return dict(opt=ret, log=log, extract_model=model.model, extract_flag=True)
            else:
                print(f'Output includes 0 / > 1 letter among candidates {set(choices)} and Z: {ans}')
        retry -= 1
        T = random.random() * wait * 2
        time.sleep(T)
        
        if retry == 0:
            options = list(choices) + ['Z'] if 'Z' not in choices else list(choices)
            log = f'{model.model} extract failed. randomly generate one. {model.model} response:{ans}\n'
            return dict(opt=random.choice(options), log=log, extract_model=model.model, extract_flag=False)


def eval_single_sample(args):
    """Evaluate a single sample."""
    model, item = args
        
    # Extract answer using the combined approach
    result = extract_answer_from_item(model, item)
    
    # Determine if the answer is correct
    hit = 1 if result['opt'] == item['GT'] else 0
    
    return {
        "index": item['index'],
        "split": item['split'],
        "question": item['question'],
        "prediction": item['prediction'],
        "extracted_answer": result['opt'],
        "extraction_method": result['extract_model'],
        "extraction_success": result['extract_flag'],
        "extraction_log": result['log'],
        "gt": item['GT'],
        "hit": hit
    }


def MMMU_preproc(data):
    """
    Preprocess MMMU dataset to reformulate open questions to multi-choice ones.
    This aligns with the implementation in multiple_choice.py
    """
    print("Preprocessing MMMU dataset...")
    cnt = 0
    As, Bs, Ans = list(data['A']), list(data['B']), list(data['answer'])
    lt = len(data)
    for i in range(lt):
        if pd.isna(As[i]):
            As[i] = Ans[i]
            Bs[i] = 'Other Answers'
            cnt += 1
    print(f'During MMMU_preproc in Evaluation, {cnt} open questions are re-formulated to multi-choice ones.')
    data['A'] = As
    data['B'] = Bs
    return data


def calculate_qwen_judge_evaluation_score(input_file, eval_model="gpt-4.1", api_type="mit", nproc=200):
    """
    Run MMMU-style judge evaluation on BoN (Best-of-N) search results.
    
    Args:
        input_file: Path to JSON file containing BoN search results
        eval_model: Judge model to use for evaluation
        api_type: API type ('mit')
        nproc: Number of processes for threading
        
    Returns:
        dict: Evaluation results with overall accuracy, per-split accuracy, and detailed results
    """
    print(f"Running judge evaluation on: {input_file}")
    
    try:
        # Load BoN JSON results
        with open(input_file, 'r') as f:
            bon_results = json.load(f)
            # Load only first row for manual debugging
            # if isinstance(bon_results, list) and len(bon_results) > 0:
            #     bon_results = [bon_results[0]]
        
        print(f"Loaded {len(bon_results)} BoN results")
        
        
        print("First filter out results where len(pred_answer)=1, which is means a successful \\boxed extraction was done. We can collate a temporary score here before running the full evaluation with judge") 
        simple_cases = []
        complex_cases = []
        for result in bon_results:
            pred_answer = result.get("pred_answer", "")
            if isinstance(pred_answer, str) and len(pred_answer.strip()) == 1:
                simple_cases.append(result)
            else:
                complex_cases.append(result)
        
        print(f"Simple cases (len=1 pred_answer): {len(simple_cases)}")
        print(f"Complex cases (requiring judge evaluation): {len(complex_cases)}")
        
        # Evaluate simple cases with direct string matching
        simple_eval_results = []
        simple_correct = 0
        for result in simple_cases:
            pred_answer = result.get("pred_answer", "").strip()
            gt_answer = result.get("gt_answer", "").strip()
            
            # Case-insensitive comparison
            hit = 1 if pred_answer.lower() == gt_answer.lower() else 0
            simple_correct += hit
            
            # Create evaluation result in same format as judge evaluation
            eval_result = {
                "index": result.get("index", "unknown"),
                "split": result.get("annotation", {}).get("split", "unknown"),
                "question": result.get("annotation", {}).get("question", ""),
                "prediction": pred_answer,
                "extracted_answer": pred_answer,
                "extraction_method": "simple_string_match",
                "extraction_success": True,
                "extraction_log": f"Simple string match: {pred_answer} vs {gt_answer}",
                "gt": gt_answer,
                "hit": hit
            }
            simple_eval_results.append(eval_result)
        
        if simple_cases:
            simple_accuracy = simple_correct / len(simple_cases)
            print(f"Simple evaluation accuracy: {simple_accuracy:.4f} ({simple_correct}/{len(simple_cases)})")
        
        # Transform BoN format to MMMU evaluation format (only complex cases)
        mmmu_results = []
        for result in complex_cases:
            # Join final_steps into single prediction string, with fallback to pred_answer
            if result.get("final_steps") and len(result["final_steps"]) > 0:
                prediction_text = "\n\n".join(result["final_steps"])
            elif result.get("pred_answer"):
                prediction_text = str(result["pred_answer"])
            else:
                prediction_text = ""  # Empty prediction
            
            # print("result:", result)
            # print("prediction_text:", prediction_text)
            # Create MMMU-compatible annotation
            annotation = result["annotation"].copy()
            annotation["prediction"] = {"text": prediction_text}
            
            mmmu_results.append(annotation)
        
        print(f"Transformed {len(mmmu_results)} complex cases to MMMU format")

        # Initialize judge_eval_results as empty list
        judge_eval_results = []

        # Only process complex cases if they exist
        if complex_cases:
            # Convert to DataFrame and apply preprocessing
            data = pd.DataFrame.from_records(mmmu_results)
            data = data.sort_values(by="index")
            data["prediction"] = [str(x) for x in data["prediction"]]

            # Normalize column names (lowercase except uppercase letters) - matches original
            for k in data.keys():
                data[k.lower() if k not in list(string.ascii_uppercase) else k] = (
                    data.pop(k)
                )

            meta = load_dataset("MMMU_DEV_VAL")
            print(f"len(data): {len(data)}")
            print(f"len(meta): {len(meta)}")
            meta_q_map = {x: y for x, y in zip(meta["index"], meta["question"])}
            data_map = {x: y for x, y in zip(data["index"], data["question"])}
            for k in data_map:
                assert k in meta_q_map, (
                    "eval_file should be the same as or a subset of dataset MMMU_DEV_VAL"
                )

            answer_map = {i: c for i, c in zip(meta["index"], meta["answer"])}
            data = MMMU_preproc(data)
            answer_map = {
                k: (v if v in list(string.ascii_uppercase) else "A")
                for k, v in answer_map.items()
            }
            data = data[data["index"].isin(answer_map)]
            data["GT"] = [answer_map[idx] for idx in data["index"]]
            items = []
            for i in range(len(data)):
                item = data.iloc[i]
                items.append(item)

            # Build judge model if needed
            model = None
            model = build_judge(eval_model, api_type)

            # Prepare evaluation tasks
            eval_tasks = []
            for item in items:
                eval_tasks.append((model, item))

            print(f"Running judge evaluation with {nproc} processes...")

            # Run threaded evaluation
            with ThreadPoolExecutor(max_workers=nproc) as executor:
                for result in tqdm(
                    executor.map(eval_single_sample, eval_tasks),
                    total=len(eval_tasks),
                    desc="Judge Evaluation",
                ):
                    judge_eval_results.append(result)
        else:
            print(
                "No complex cases requiring judge evaluation - all samples handled by simple string matching"
            )
        
        # Merge simple and judge evaluation results
        eval_results = simple_eval_results + judge_eval_results
        print(f"Merged results: {len(simple_eval_results)} simple + {len(judge_eval_results)} judge = {len(eval_results)} total")
        
        # Calculate overall accuracy
        accuracy = sum(r['hit'] for r in eval_results) / len(eval_results)
        
        # Calculate per-split accuracy
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
            print(f"Overall accuracy for {split} split: {split_accuracy:.4f} ({sum(r['hit'] for r in split_results)}/{len(split_results)})")
        
        print(f"Overall accuracy: {accuracy:.4f} ({sum(r['hit'] for r in eval_results)}/{len(eval_results)})")
        
        # Show breakdown by evaluation method
        simple_results = [r for r in eval_results if r.get('extraction_method') == 'simple_string_match']
        judge_results = [r for r in eval_results if r.get('extraction_method') != 'simple_string_match']
        
        if simple_results:
            simple_acc = sum(r['hit'] for r in simple_results) / len(simple_results)
            print(f"Simple string matching accuracy: {simple_acc:.4f} ({sum(r['hit'] for r in simple_results)}/{len(simple_results)})")
        
        if judge_results:
            judge_acc = sum(r['hit'] for r in judge_results) / len(judge_results)
            print(f"Judge model evaluation accuracy: {judge_acc:.4f} ({sum(r['hit'] for r in judge_results)}/{len(judge_results)})")
        
        return {
            "overall_accuracy": accuracy,
            "accuracy_by_split": accuracy_by_split,
            "eval_results": eval_results,
            "num_correct_samples_after_llm_judgement": sum(r['hit'] for r in eval_results),
            "num_samples_after_llm_judgement": len(eval_results)
        }
        
    except Exception as e:
        print(f"Error in judge evaluation: {e}")
        print(traceback.format_exc())
        return None


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Run MMMU-style judge evaluation on BoN search results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qwen_collate_final_evaluation.py --input results.json
  python qwen_collate_final_evaluation.py --input results.json --output eval_results.csv --eval-model gpt-4.1-nano --api-type mit --nproc 8
        """
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to JSON file containing BoN search results"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path to save detailed evaluation results as CSV (optional)"
    )
    parser.add_argument(
        "--eval-model", 
        type=str, 
        default="gpt-4.1",
        help="Judge model to use for evaluation (default: gpt-4.1)"
    )
    parser.add_argument(
        "--api-type", 
        type=str, 
        default="mit",
        choices=["mit"],
        help="API type to use (default: mit)"
    )
    parser.add_argument(
        "--nproc", 
        type=int, 
        default=200,
        help="Number of processes for parallel evaluation (default: 4)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print(f"Starting judge evaluation...")
    print(f"Input file: {args.input}")
    print(f"Judge model: {args.eval_model}")
    print(f"API type: {args.api_type}")
    print(f"Parallel processes: {args.nproc}")
    print("-" * 50)
    
    # Run the evaluation
    try:
        results = calculate_qwen_judge_evaluation_score(
            input_file=args.input,
            eval_model=args.eval_model,
            api_type=args.api_type,
            nproc=args.nproc
        )
        
        if results is None:
            print("Evaluation failed!")
            return 1
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        overall_accuracy = results["overall_accuracy"]
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy:.2%})")
        
        if results["accuracy_by_split"]:
            print("\nAccuracy by Split:")
            for split, accuracy in results["accuracy_by_split"].items():
                print(f"  {split}: {accuracy:.4f} ({accuracy:.2%})")
        
        eval_results = results["eval_results"]
        total_samples = len(eval_results)
        correct_samples = sum(r['hit'] for r in eval_results)
        
        print(f"\nCorrect samples: {correct_samples}/{total_samples}")
        
        # Count extraction methods
        extraction_methods = {}
        successful_extractions = 0
        for r in eval_results:
            method = r.get('extraction_method', 'unknown')
            if method not in extraction_methods:
                extraction_methods[method] = 0
            extraction_methods[method] += 1
            if r.get('extraction_success', False):
                successful_extractions += 1
        
        print(f"Successful extractions: {successful_extractions}/{total_samples} ({successful_extractions/total_samples:.2%})")
        print(f"Extraction methods used:")
        for method, count in extraction_methods.items():
            print(f"  {method}: {count} samples")
        
        # Save detailed results if requested
        if args.output:
            try:
                results_df = pd.DataFrame(eval_results)
                results_df.to_csv(args.output, index=False)
                print(f"\nDetailed results saved to: {args.output}")
            except Exception as e:
                print(f"Warning: Failed to save detailed results: {e}")
        
        # Save accuracy summary
        input_base = os.path.splitext(args.input)[0]
        summary_file = f"{input_base}_judge_accuracy.json"
        try:
            summary = {
                "overall_accuracy": overall_accuracy,
                "accuracy_by_split": results["accuracy_by_split"],
                "total_samples": total_samples,
                "correct_samples": correct_samples,
                "successful_extractions": successful_extractions,
                "extraction_methods": extraction_methods,
                "eval_model": args.eval_model,
                "api_type": args.api_type
            }
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Accuracy summary saved to: {summary_file}")
        except Exception as e:
            print(f"Warning: Failed to save accuracy summary: {e}")
        
        print(f"\nEvaluation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if args.verbose:
            print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())