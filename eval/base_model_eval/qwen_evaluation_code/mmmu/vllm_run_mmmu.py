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


def run_inference(args):
    """Run inference on the MMMU dataset."""
    # Set NUMEXPR_MAX_THREADS to avoid threading issues in parallel execution
    os.environ['NUMEXPR_MAX_THREADS'] = '64'
    
    # Handle GPU selection if specified
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f"Using GPU {args.gpu_id} (CUDA_VISIBLE_DEVICES={args.gpu_id})")
    
    # Load dataset
    data = load_dataset(args.dataset)

    # Set up image root directory
    img_root = os.path.join(os.environ["LMUData"], "images", "MMMU")
    os.makedirs(img_root, exist_ok=True)

    # Handle dataset slicing for parallel execution
    if args.data_end is None:
        args.data_end = len(data)
    
    # Validate range parameters
    if args.data_begin < 0 or args.data_begin >= len(data):
        raise ValueError(f"Invalid data_begin: {args.data_begin} (dataset size: {len(data)})")
    if args.data_end <= args.data_begin:
        raise ValueError(f"data_end ({args.data_end}) must be greater than data_begin ({args.data_begin})")
    
    # Slice the dataset
    original_size = len(data)
    data = data.iloc[args.data_begin:min(args.data_end, len(data))]
    
    if args.partition_id is not None:
        print(f"Partition {args.partition_id}: Processing samples [{args.data_begin}, {min(args.data_end, original_size)})")
    print(f"Dataset subset size: {len(data)} (from original {original_size})")
    
    # Set up dump_image function
    def dump_image_func(line):
        return dump_image(line, img_root)

    # Create output directory and adjust output filename for partitions
    if args.partition_id is not None:
        # Modify output filename to include partition info
        base_name = os.path.basename(args.output_file)
        dir_name = os.path.dirname(args.output_file)
        name_parts = os.path.splitext(base_name)
        partition_filename = f"{name_parts[0]}-p{args.partition_id}-{args.data_begin}-{args.data_end}{name_parts[1]}"
        output_file = os.path.join(dir_name, partition_filename)
        print(f"Partition output file: {output_file}")
    else:
        output_file = args.output_file
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Set up CoT prompt if enabled
    cot_prompt = ""
    if args.use_cot:
        cot_prompt = (
            args.cot_prompt
            if args.cot_prompt
            else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        )
        print(f"Using CoT prompt: {cot_prompt}")

    # Initialize VLLM model
    print(f"Loading VLLM model from {args.model_path}")

    # Ensure CUDA device is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    # Initialize VLLM engine
    if "32b" in args.model_path.lower():
        model = LLM(
            model=args.model_path,
            max_num_seqs=32,
            limit_mm_per_prompt={"image": 24},
            gpu_memory_utilization=0.90,
            mm_processor_kwargs={
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            )
    elif args.model_path == "openbmb/MiniCPM-V-2_6":
            # policy_model = AutoModel.from_pretrained(args.policy_model_path, trust_remote_code=True)
            # policy_model = policy_model.to(dtype=torch.bfloat16)
            # policy_model.eval().cuda()
            model = LLM(
                model=args.model_path,
                max_num_seqs=128,
                limit_mm_per_prompt={"image": 24},
                gpu_memory_utilization=0.80,
                trust_remote_code=True,
            )
    else:
        model = LLM(
            model=args.model_path,
            max_num_seqs=128,  # equivalent to max_running_requests
            limit_mm_per_prompt={"image": 24},  # max images per prompt
            gpu_memory_utilization=0.80,  # equivalent to mem_fraction_static
            mm_processor_kwargs={
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            # disable_mm_preprocessor_cache=True,
            # device="cuda",
            # base_gpu_id=gpu_id,
        )

    if "minicpm" in args.model_path.lower():
        processor = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
    else:
        processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=5120*28*28, min_pixels=1280*28*28)

    # data = data[data["id"].isin(["validation_Chemistry_12"])]

    # Run inference
    results = []
    for i in tqdm(range(len(data)), desc="Running inference"):
        line = data.iloc[i]
        dump_image(line, img_root)
        index = line["index"]

        # Convert line to dict and ensure all values are JSON serializable
        line_dict = line.to_dict()
        for k, v in line_dict.items():
            if isinstance(v, np.integer):
                line_dict[k] = int(v)
            elif isinstance(v, np.floating):
                line_dict[k] = float(v)

        # Extract base64 image data directly from dataset (images are already base64 encoded)
        image_data_base64_list = []
        if "image" in line_dict and line_dict["image"]:
            # Handle single image case
            if isinstance(line_dict["image"], str):
                image_data_base64_list.append(line_dict["image"])
            # Handle multiple images case
            elif isinstance(line_dict["image"], list):
                image_data_base64_list.extend(
                    [img for img in line_dict["image"] if img]
                )

        print(f"Found {len(image_data_base64_list)} base64 images in line id: {line['id']}, index: {line['index']}")

        # Build prompt text (similar to _build_mmmu_prompt but adapted for SGLang)
        question = line["question"]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        prompt = ""
        if "hint" in line and not pd.isna(line["hint"]):
            prompt += f"Hint: {line['hint']}\n"

        prompt += f"Question: {question}\n"

        if len(options):
            prompt += "Options:\n"
            for key, item in options.items():
                prompt += f"{key}. {item}\n"
            prompt += "Please select the correct answer from the options above.\n"
        
        # Add CoT prompt if enabled
        if args.use_cot:
            prompt += cot_prompt

        prompt = prompt.rstrip()
        print(prompt)
        # Build messages array for SGLang (images are already in base64 format)
        if "internvl" in args.model_path.lower() or "minicpm" in args.model_path.lower():
                question_in_messages_array_format, question_corresponding_image_data_base64_list = (
                    prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm(
                        prompt,
                        image_data_base64_list,
                        interleave_image_tokens=True,
                        model_type="internvl" if "internvl" in args.model_path.lower() else "minicpm"
                    )
                )
        else:
            # TODO: For now we use the same function for both MMMU and PuzzleVQA, but be careful when using for MathVista
            question_in_messages_array_format, question_corresponding_image_data_base64_list = (
                prepare_question_array_with_base64_image_strings(
                    prompt,
                    image_data_base64_list,
                    interleave_image_tokens=True,
                )
            )
        
        print(f"length of question_corresponding_image_data_base64_list: {len(question_corresponding_image_data_base64_list)}")
        # (
        #     question_in_messages_array_format,
        #     question_corresponding_image_data_base64_list,
        # ) = prepare_question_array_with_base64_image_strings(
        #     prompt,
        #     image_data_base64_list,  # Use base64 images directly from dataset
        #     interleave_image_tokens=True,  # MMMU uses interleaved images
        # )

        # if len(question_corresponding_image_data_base64_list) == 1:
        # if len(question_corresponding_image_data_base64_list) > 1:
        #     print(question_in_messages_array_format)
        #     print(len(question_corresponding_image_data_base64_list))
            # print(question_corresponding_image_data_base64_list)
            # exit()

        sys_and_user_prompt_messages_array = (
            [{"role": "system", "content": "You are a helpful assistant."}]
            + question_in_messages_array_format
            # + [{"role": "assistant", "content": ""}]
        )

        # Override for Gemma models due to incorrect tokenizer tokens
        if "gemma" in args.model_path.lower():
            # Set the tokenizer's special tokens directly (official method)
            processor.tokenizer.eos_token = "<end_of_turn>"
            processor.tokenizer.bos_token = "[BOS]"
            print(
                f"Overriding EOS token for Gemma model {args.model_path}: {processor.tokenizer.eos_token}"
            )
            print(
                f"Overriding BOS token for Gemma model {args.model_path}: {processor.tokenizer.bos_token}"
            )
        elif "cpm" in args.model_path.lower() or "internvl" in args.model_path.lower(): 
            print(
                f"Using EOS token for policy model {args.model_path}: {processor.eos_token}"
            )
            print(
                f"Using BOS token for policy model {args.model_path}: {processor.bos_token}"
            )
        else:
            print(
                f"Using EOS token for model {args.model_path}: {processor.tokenizer.eos_token}"
            )
            print(
                f"Using BOS token for model {args.model_path}: {processor.tokenizer.bos_token}"
            )

        user_input_prompt = processor.apply_chat_template(
            sys_and_user_prompt_messages_array,
            tokenize=False,
            add_generation_prompt=True,
        )

        if "minicpm" in args.model_path.lower():
            sys_prompt = sys_and_user_prompt_messages_array[0]['content']
            user_prompt = sys_and_user_prompt_messages_array[1]['content']
            combined_user_prompt = sys_prompt + "\n" + user_prompt
            user_input_prompt = processor.apply_chat_template(
                [{"role": "user", "content": combined_user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            user_input_prompt = user_input_prompt.replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>", "")
            print(f"Policy prompt: {user_input_prompt}")

        # user_input_prompt = user_input_prompt.replace(
        #     processor.tokenizer.eos_token, ""
        # ).strip()
        # print(f"user_input_prompt: {user_input_prompt}")
        # print(f"len(question_corresponding_image_data_base64_list): {len(question_corresponding_image_data_base64_list)}")
        # if len(question_corresponding_image_data_base64_list) > 1:
        #     print(f"user_input_prompt: {user_input_prompt}")
        #     exit()

        # Generate response using VLLM
        sampling_params = SamplingParams(
            n=1,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            include_stop_str_in_output=True,
            max_tokens=32768,
            # logprobs=20,
        )

        images = []
        for base64_img in question_corresponding_image_data_base64_list:
            img_data = base64.b64decode(base64_img)
            if "minicpm" in args.model_path.lower():
                img = Image.open(BytesIO(img_data)).convert('RGB')
                img_width, img_height = img.width, img.height # resizing https://github.com/OpenBMB/MiniCPM-V/blob/main/eval_mm/vlmevalkit/vlmeval/vlm/minicpm_v.py#L260
                if (img_width * img_height) >= (1344 * 1344):
                    print(f"Image is too large, skipping resize: {img_width}x{img_height}")
                    img = img
                else:
                    ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                    max_img_width = int(img_width * ratio)
                    new_img_width = random.randint(img_width, max_img_width)
                    new_img_height = int(new_img_width / img_width * img_height)
                    print(f"Resizing image from {img_width}x{img_height} to {new_img_width}x{new_img_height}")
                    img = img.resize((new_img_width, new_img_height))
            elif "internvl" in args.model_path.lower():
                img = base64_img
            else:
                img = Image.open(BytesIO(img_data))
                img.load()
            images.append(img)

        llm_outputs = model.generate(
            {
                "prompt": user_input_prompt,
                "multi_modal_data": {"image": images} if images else None,
            },
            sampling_params,
        )

        completion_output = llm_outputs[0].outputs[0]
        request_output = llm_outputs[0]

        # Extract response with metadata according to vLLM documentation
        response = {
            "text": completion_output.text,
            "meta_info": {
                "id": request_output.request_id,
                "finish_reason": {
                    "type": str(completion_output.finish_reason),
                    "length": len(completion_output.token_ids)
                    if completion_output.token_ids
                    else None,
                },
                "prompt_tokens": len(request_output.prompt_token_ids)
                if request_output.prompt_token_ids
                else None,
                "completion_tokens": len(completion_output.token_ids)
                if completion_output.token_ids
                else None,
                "cached_tokens": getattr(request_output.metrics, "cached_tokens", 0)
                if hasattr(request_output, "metrics")
                else 0,
                "cumulative_logprob": completion_output.cumulative_logprob,
                "e2e_latency": getattr(request_output.metrics, "finished_time", 0)
                - getattr(request_output.metrics, "arrival_time", 0)
                if hasattr(request_output, "metrics")
                else None,
            },
        }
            
        print(f"response: {response}")
        # exit()
        print(f"annotation answer: {line['answer']}")
        print('-' * 50)
        
        # Save result
        result = {
            "index": int(index) if isinstance(index, np.integer) else index,
            "annotation": line_dict,
            "task": args.dataset,
            "result": {"gen": response},
            "messages": sys_and_user_prompt_messages_array[
                :-1
            ],  # Exclude the empty assistant message
        }
        results.append(result)
        
        # Write intermediate results
        if i % 10 == 0:
            with open(output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')
            
    # Write final results
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"Inference completed. Results saved to {output_file}")

def run_evaluation(args):
    """Run evaluation on inference results."""
    # Load results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            job = json.loads(line)
            annotation = job["annotation"]
            annotation["prediction"] = job["result"]["gen"]
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
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    infer_parser.add_argument("--data-dir", type=str, default="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/qwen-evaluation/mmmu/data", help="The absolute path of MMMU_DEV_VAL.tsv")
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought prompting")
    infer_parser.add_argument("--cot-prompt", type=str, default="", help="Custom Chain-of-Thought prompt")
    infer_parser.add_argument("--data-begin", type=int, default=0, help="Starting index for dataset subset")
    infer_parser.add_argument("--data-end", type=int, default=None, help="Ending index for dataset subset")
    infer_parser.add_argument("--partition-id", type=int, default=None, help="Partition ID for parallel execution")
    infer_parser.add_argument("--gpu-id", type=int, default=None, help="Specific GPU to use (sets CUDA_VISIBLE_DEVICES)")
    
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
        run_inference(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
