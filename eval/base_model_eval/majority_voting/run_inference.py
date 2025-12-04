import glob
import os
import argparse
import json
import datetime
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from collections import Counter
import sglang as sgl
import ast
from typing import List, Dict

# Import utility functions from the bon_greedy_search module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reward_guided_search.prompts import (
    MCQ_SUFFIX_PROMPT_NO_TEMPLATE,
    POLICY_QWEN_SYSTEM_PROMPT_DYNAMIC_VARS,
    MCQ_ANSWER_FORMAT_INSTRUCTIONS,
    SHORT_ANSWER_FORMAT_INSTRUCTIONS,
    POLICY_USER_PROMPT,
)
from reward_guided_search.utils import sample_to_images_list, convert_images_to_base64, prepare_question_array_with_base64_image_strings, extract_boxed
from reward_guided_search.qwenvl_utils import process_vision_info
from common.logger import log_info

# Data paths for different evaluation datasets
DATA_PATHS = {
    # "mathvista_testmini": "../eval_datasets/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet", 
    "mmmu_dev": "eval_datasets/MMMU/data/dev-00000-of-00001.parquet", 
}

validation_files = glob.glob("./eval_datasets/MMMU/data/validation*.parquet")
if not validation_files:
    raise ValueError("No validation*.parquet files found in ./eval_datasets/MMMU/data/ directory")
log_info(f"Found {len(validation_files)} MMMU validation parquet files: {validation_files}")
DATA_PATHS["mmmu_validation"] = validation_files

class MajorityVotingPRM:
    """Majority Voting Policy Reward Model using SGLang for batch inference."""
    
    def __init__(self, model_path: str, num_votes: int = 7, temperature: float = 0.7):
        """
        Initialize the MajorityVotingPRM with SGLang engine.
        
        Args:
            model_path: Path to the VLM model
            num_votes: Number of votes for majority voting
            temperature: Sampling temperature
        """
        self.model_path = model_path
        self.num_votes = num_votes
        self.temperature = temperature
        
        num_gpus = torch.cuda.device_count()
        log_info(f"Detected {num_gpus} GPUs available")
        
        
        self.engine = sgl.Engine(model_path=model_path)
            
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Sampling parameters for generation
        self.sampling_params = {
            "n": num_votes,  # Generate n samples for majority voting
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "temperature": temperature,
            "max_new_tokens": 8192,
            "stop": ["<|im_end|>"],
        }
        
        log_info(f"Initialized MajorityVotingPRM with model: {model_path}")
        log_info(f"Sampling parameters: {self.sampling_params}")
    
    def generate(self, prompts: List[str], image_data_list: List[List[Dict]], 
                 return_logprobs: bool = False) -> List[Dict]:
        """
        Generate responses for a batch of prompts with images.
        
        Args:
            prompts: List of text prompts
            image_data_list: List of image data (base64 encoded) for each prompt
            return_logprobs: Whether to return logprobs for token IDs
        
        Returns:
            List of dictionaries containing generated texts and majority vote results
        """
        results = []
        
        # Process each prompt sequentially
        for prompt, images in zip(prompts, image_data_list):
            # Generate multiple samples for majority voting
            sampling_params = self.sampling_params.copy()
            
            print("running inference for prompt:")
            print(prompt)
            print(len(images))
            outputs = self.engine.generate(
                prompt=prompt,
                image_data=[images] if len(images) > 1 else images,
                sampling_params=sampling_params,
            )
            
            # Extract generated texts
            generated_texts = []
            for output in outputs:
                text = output["text"].replace("\\n", "\n").strip()
                generated_texts.append(text)
            
            # Perform majority voting
            vote_result = self._majority_vote(generated_texts)
            
            results.append({
                "prompt": prompt,
                "generated_texts": generated_texts,
                "majority_vote": vote_result["majority"],
                "vote_counts": vote_result["counts"],
                "raw_outputs": outputs if return_logprobs else None
            })
        
        return results
    
    def _majority_vote(self, texts: List[str]) -> Dict:
        """
        Perform majority voting on generated texts.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Dictionary with majority answer and vote counts
        """
        # Extract answers from texts (looking for boxed answers or MCQ choices)
        answers = []
        for text in texts:
            # Try to extract boxed answer
            boxed = extract_boxed(text)
            if boxed:
                answers.append(boxed)
            else:
                # Try to find MCQ answer (A, B, C, D, E) at the end
                import re
                match = re.search(r'\b([A-E])\b(?!.*\b[A-E]\b)', text)
                if match:
                    answers.append(match.group(1))
                else:
                    answers.append(text)  # Use full text if no clear answer
        
        # Count votes
        vote_counts = Counter(answers)
        
        # Get majority answer
        if vote_counts:
            majority_answer, _ = vote_counts.most_common(1)[0]
        else:
            majority_answer = None
        
        return {
            "majority": majority_answer,
            "counts": dict(vote_counts),
            "total_votes": len(answers)
        }


def prepare_prompt_for_evaluation(data: Dict, dataset_name: str, processor) -> tuple:
    """
    Prepare prompt and images for evaluation based on dataset format.
    
    Args:
        data: Single data sample from dataset
        dataset_name: Name of the dataset (e.g., "mmmu_dev", "mathvista_testmini")
        processor: Model processor for chat template
    
    Returns:
        Tuple of (formatted_prompt, image_data_base64_list)
    """
    # Extract images
    image_data_list = sample_to_images_list(data)
    if len(image_data_list) > 0:
        image_data_base64_list = convert_images_to_base64(image_data_list)
    else:
        image_data_base64_list = []
    
    # Process images for Qwen format
    if image_data_base64_list:
        processed_image_data_pil_list, _ = process_vision_info([{"content": image_data_base64_list}])
        processed_image_data_qwen_format_base64_list = [
            image["image"] for image in convert_images_to_base64(processed_image_data_pil_list)
        ]
    else:
        processed_image_data_qwen_format_base64_list = []
    
    # Build prompt
    question = data["question"]
    user_prompt = POLICY_USER_PROMPT.replace("{{QUESTION}}", question)
    
    # Handle multiple choice questions
    question_type = data.get("question_type", "")
    if question_type in ["multi-choice", "multiple-choice"]:
        options = data["options"] if "mmmu" in dataset_name else data.get("choices", [])
        
        # Parse options if they're in string format
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                try:
                    options = ast.literal_eval(options)
                except Exception:
                    log_info(f"Failed to parse options: {repr(options)}")
                    raise
        
        # Format options
        if "mmmu" in dataset_name:
            options_str = "\n".join([
                f"{chr(65 + i)}: {option}"
                for i, option in enumerate(options)
            ])
        else:
            options_str = "\n".join([f"{option}" for option in options]) + "\n"
        
        user_prompt += MCQ_SUFFIX_PROMPT_NO_TEMPLATE.replace("{{OPTIONS}}", options_str)
        system_prompt = POLICY_QWEN_SYSTEM_PROMPT_DYNAMIC_VARS.replace(
            "{{FORMAT_INSTRUCTIONS}}", MCQ_ANSWER_FORMAT_INSTRUCTIONS
        )
    else:
        system_prompt = POLICY_QWEN_SYSTEM_PROMPT_DYNAMIC_VARS.replace(
            "{{FORMAT_INSTRUCTIONS}}", SHORT_ANSWER_FORMAT_INSTRUCTIONS
        )
    
    # Determine interleave setting based on dataset
    interleave_image_tokens = True if "mmmu" in dataset_name else False
    
    # Prepare question in messages array format
    question_in_messages_array_format, corresponding_image_data_base64_list = prepare_question_array_with_base64_image_strings(
        user_prompt, 
        processed_image_data_qwen_format_base64_list, 
        interleave_image_tokens=interleave_image_tokens
    )
    
    # Build full message array
    sys_and_user_prompt_messages_array = (
        [{"role": "system", "content": system_prompt}] +
        question_in_messages_array_format +
        [{"role": "assistant", "content": ""}]
    )
    
    # Apply chat template
    formatted_prompt = processor.apply_chat_template(
        sys_and_user_prompt_messages_array,
        tokenize=False,
        add_generation_prompt=False,
    )
    formatted_prompt = formatted_prompt.replace("<|im_end|>", "").strip()
    
    return formatted_prompt, corresponding_image_data_base64_list


def evaluate_majority_voting(
    model_path: str,
    dataset_name: str,
    num_votes: int = 7,
    temperature: float = 0.7,
    data_begin: int = 0,
    data_end: int = 10,
    output_dir: str = "./results",
    development_mode: bool = False
):
    """
    Run majority voting evaluation on a dataset.
    
    Args:
        model_path: Path to the VLM model
        dataset_name: Name of dataset to evaluate
        num_votes: Number of votes for majority voting
        temperature: Sampling temperature
        data_begin: Starting index of dataset
        data_end: Ending index of dataset
        output_dir: Directory to save results
        development_mode: Whether to run in development mode (small subset)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = MajorityVotingPRM(model_path, num_votes, temperature)
    
    # Load dataset
    log_info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset("parquet", data_files=DATA_PATHS[dataset_name])
    dataset = dataset[list(dataset.keys())[0]]
    
    # Filter dataset if in development mode
    if development_mode:
        target_ids = ["dev_Electronics_5, dev_Art_Theory_3"] if dataset_name == "mmmu_dev" else ["3", "5"]
        if dataset_name == "mmmu_dev":
            dataset = dataset.filter(lambda x: x["id"] in target_ids)
        else:
            dataset = dataset.filter(lambda x: x["pid"] in target_ids)
    
    # Select data range
    start_idx = data_begin
    end_idx = min(data_end, len(dataset))
    dataset = dataset.select(range(start_idx, end_idx))
    log_info(f"Evaluating {len(dataset)} samples")
    
    # Prepare prompts and images for batch processing
    prompts = []
    image_data_lists = []
    ground_truths = []
    
    for data in tqdm(dataset, desc="Preparing prompts"):
        prompt, corresponding_image_data_base64_list = prepare_prompt_for_evaluation(data, dataset_name, model.processor)
        prompts.append(prompt)
        image_data_lists.append(corresponding_image_data_base64_list)
        
        # Get ground truth answer
        if "mmmu" in dataset_name:
            ground_truths.append(data["answer"])
        else:
            ground_truths.append(data.get("answer", data.get("solution", "")))
    
    # Run batch inference with majority voting
    log_info("Running batch inference with majority voting...")
    results = model.generate(prompts, image_data_lists, return_logprobs=True)
    
    # Compile results
    evaluation_results = []
    correct = 0
    total = len(results)
    
    for i, (result, gt) in enumerate(zip(results, ground_truths)):
        pred_answer = result["majority_vote"]
        
        # Check if answer is correct
        is_correct = str(pred_answer).upper() == str(gt).upper()
        if is_correct:
            correct += 1
        
        evaluation_results.append({
            "index": i,
            "prompt": result["prompt"][:200] + "...",  # Truncate for readability
            "ground_truth": gt,
            "predicted_answer": pred_answer,
            "is_correct": is_correct,
            "vote_counts": result["vote_counts"],
            "all_votes": result["generated_texts"]
        })
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Save results
    run_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir,
        f"majority_voting_{dataset_name}_{data_begin}-{data_end}_{run_datetime}.json"
    )
    
    final_results = {
        "model": model_path,
        "dataset": dataset_name,
        "num_samples": total,
        "num_votes_per_sample": num_votes,
        "temperature": temperature,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": evaluation_results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    log_info("Evaluation complete!")
    log_info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    log_info(f"Results saved to: {output_file}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Majority Voting evaluation for VLMs")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the VLM model (e.g., Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mathvista_testmini", "mmmu_dev", "mmmu_validation"],
                        help="Dataset to evaluate on")
    parser.add_argument("--num_votes", type=int, default=7,
                        help="Number of votes for majority voting")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--data_begin", type=int, default=0,
                        help="Starting index of dataset")
    parser.add_argument("--data_end", type=int, default=10,
                        help="Ending index of dataset")
    parser.add_argument("--output_dir", type=str, default="./results/majority_voting",
                        help="Directory to save results")
    parser.add_argument("--development_mode", action="store_true",
                        help="Run in development mode (small subset)")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_majority_voting(
        model_path=args.model_path,
        dataset_name=args.dataset,
        num_votes=args.num_votes,
        temperature=args.temperature,
        data_begin=args.data_begin,
        data_end=args.data_end,
        output_dir=args.output_dir,
        development_mode=args.development_mode
    )


if __name__ == "__main__":
    main()