#!/usr/bin/env python3
"""
PuzzleVQA LLM Judge Evaluation

This module implements the LLM Judge fallback mechanism for PuzzleVQA evaluation,
following the exact reference implementation from LLM-PuzzleTest/PuzzleVQA.

The judge handles cases where regex-based answer extraction fails by using
the ChainThoughtMultiExtractPrompter format to guide the model toward
explicit answer selection.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import dotenv
dotenv.load_dotenv()

if "OPENAI_API_KEY" in os.environ:
    print(os.environ["OPENAI_API_KEY"][:5])
else:
    print("OPENAI_API_KEY not found")
    exit()

# Import from same directory
try:
    from utils.logger import log_info
except ImportError:
    # Fallback to basic print if logger not available
    def log_info(message):
        print(f"INFO - {message}")

try:
    from puzzleTest_helpers import extract_puzzlevqa_answer
except ImportError:
    from eval.tts_eval.reward_guided_search.puzzleTest_helpers import extract_puzzlevqa_answer


class PuzzleVQAJudgeModel:
    """
    LLM Judge model for PuzzleVQA answer extraction fallback.
    
    Implements the exact ChainThoughtMultiExtractPrompter logic from the
    reference PuzzleVQA implementation.
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", temperature: float = 0.0):
        """
        Initialize the PuzzleVQA judge model.
        
        Args:
            model_path: Path to the policy model for judge inference
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model_path = model_path
        self.temperature = temperature
        self.model = None
        
    def load_model(self):
        """Load the LLM model for judge inference."""
        if self.model is None:
            # print("Loading OpenAI client for GPT-4.1")
            self.model = OpenAI()
            # print("OpenAI client loaded successfully")

    # def build_extraction_prompt(self, original_prompt: str, reasoning_trajectory: str, options: List[str]) -> str:
    #     """
    #     Build the extraction prompt using PuzzleVQA's ChainThoughtMultiExtractPrompter format.

    #     This replicates the exact format from the reference implementation:
    #     [original_prompt]
    #     [reasoning_trajectory]

    #     Therefore, among (A) (B) (C) (D), the answer is:

    #     Args:
    #         original_prompt: Original question prompt with options
    #         reasoning_trajectory: The model's reasoning steps
    #         options: List of answer options

    #     Returns:
    #         Formatted extraction prompt
    #     """
    #     # Handle 3-option size puzzles
    #     size_options = {"small", "medium", "large"}
    #     is_size_puzzle = len(options) == 3 and set(options) == size_options

    #     # Build the extraction instruction
    #     if is_size_puzzle:
    #         extraction_instruction = "Therefore, among (A) (B) (C), the answer is:"
    #     else:
    #         extraction_instruction = "Therefore, among (A) (B) (C) (D), the answer is:"

    #     # Combine into final extraction prompt
    #     parts = [
    #         original_prompt.strip(),
    #         reasoning_trajectory.strip(),
    #         "",
    #         extraction_instruction
    #     ]

    #     return "\n".join(parts)
    
    def judge_single_sample(self, sample_data: Dict) -> Tuple[str, bool]:
        """
        Run LLM judge on a single sample to extract the answer.
        
        Args:
            sample_data: Sample data dictionary from results file
            
        Returns:
            Tuple of (extracted_answer, success_flag)
        """
        self.load_model()
        
        try:
            # Extract information from sample data
            original_question = sample_data.get("question", "")
            final_steps = sample_data.get("final_steps", [])
            annotation = sample_data.get("annotation", {})
            reasoning_trajectory = sample_data.get("prediction_full_text", "")
            
            # Get options from annotation
            options = annotation.get("options", [])
            if not options:
                return "Cannot extract: No options found", False
            
            # Get the final reasoning step (last step in trajectory)
            if not final_steps:
                return "Cannot extract: No reasoning steps found", False
            
            # reasoning_trajectory = final_steps[-1] if isinstance(final_steps, list) else str(final_steps)
            
            # Reconstruct original prompt (approximation of PuzzleVQA format)
            judge_prompt = self._construct_judge_prompt(
                original_question, options, reasoning_trajectory
            )
            
            # Build extraction prompt
            # extraction_prompt = self.build_extraction_prompt(original_prompt, reasoning_trajectory, options)
            
            # Run judge inference
            print(f"Running judge inference with prompt: {judge_prompt}")
            response = self.model.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "text"},
                temperature=0,
                max_completion_tokens=256,
            )
            judge_output = response.choices[0].message.content.strip()

            print(f"returned Judge output: {judge_output}")

            alphabetical_options = [f"{chr(65 + i)}" for i in range(len(options))]
            valid_responses = alphabetical_options + ["Z"]
            if judge_output in valid_responses:
                extracted_answer = judge_output
                success = True
            else:
                extracted_answer = judge_output
                success = False
            
            return extracted_answer, success
            
        except Exception as e:
            print(f"Error in judge_single_sample: {e}")
            return f"Judge error: {str(e)}", False

    def _construct_judge_prompt(
        self, question: str, options: List[str], reasoning_trajectory: str
    ) -> str:
        """
        Reconstruct the original PuzzleVQA prompt format.
        
        Args:
            question: The question text
            options: List of answer options
            
        Returns:
            Reconstructed prompt in PuzzleVQA format
        """
        judge_prompt = (
            "You are an AI assistant who will help me to match "
            "an answer with several options of a single-choice question. "
            "You are provided with a question, several options, and an answer, "
            "and you need to find which option is most similar to the answer. "
            "If the meaning of all options are significantly different from the answer, output Z. "
            "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
            "Example 1: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
            "Answer: a cute teddy bear\nYour output: A\n"
            "Example 2: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
            "Answer: Spider\nYour output: Z\n"
            "Example 3: \n"
            "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
        )
        formatted_options = " ".join(
            [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        )
        judge_prompt = judge_prompt.format(
            question, formatted_options, reasoning_trajectory
        )

        return judge_prompt

def eval_single_sample(item):
    """Evaluate a single sample."""
    # model, item = args

    judge_model = PuzzleVQAJudgeModel()

    try:
        extracted_answer, success = judge_model.judge_single_sample(item)
    except Exception as e:
        print(f"Error in judge_single_sample: {e}")
        return f"Judge error: {str(e)}", False

    # Extract answer using the combined approach
    # result = extract_answer_from_item(model, item)
    
    # Determine if the answer is correct
    hit = 1 if extracted_answer == item['gt_answer'] else 0
    print(f"extracted_answer: {extracted_answer}, item['gt_answer']: {item['gt_answer']}, hit: {hit}")
    
    return {
        "index": item['index'],
        "split": item['task'],
        "question": item['question'],
        "prediction": item['prediction_full_text'],
        "extracted_answer": extracted_answer,
        "extraction_method": "gpt-4.1-judge",
        "extraction_success": success,
        "extraction_log": "Judge output: " + extracted_answer,
        "gt": item['gt_answer'],
        "hit": hit
    }

def calculate_puzzlevqa_judge_evaluation_score(results_file: str) -> Optional[Dict]:
    """
    Calculate PuzzleVQA evaluation scores with LLM Judge fallback mechanism.
    
    This function implements the two-stage evaluation approach:
    1. Count initial regex-based extraction results
    2. Apply LLM Judge to failed extractions
    3. Calculate and return both sets of statistics
    
    Args:
        results_file: Path to JSON results file from evaluation
        
    Returns:
        Dictionary with evaluation statistics matching MMMU judge interface:
        {
            "overall_accuracy": float,
            "num_correct_samples_after_llm_judgement": int,
            "num_samples_after_llm_judgement": int,
            "initial_correct_count": int,
            "initial_total_count": int,
            "judge_improved_count": int,
            "judge_processed_count": int
        }
    """
    print("Starting LLM Judge evaluation...")

    try:
        
        # Load results file
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            print("Warning: Empty results file")
            return None
        
        # Initialize counters
        # initial_correct = 0
        initial_total = len(results)
        # judge_processed = 0
        # judge_improved = 0
        
        # TODO: Change to OpenAI
        # judge_model = PuzzleVQAJudgeModel()
        
        # Process each sample
        print(f"Processing {initial_total} samples for judge evaluation...")
        print("First filter out results where len(pred_answer)=1, which is means a successful \\boxed extraction was done. We can collate a temporary score here before running the full evaluation with judge") 
        simple_cases = []
        complex_cases = []

        for i, sample in enumerate(results):
            if i % 50 == 0:  # Progress logging
                print(f"Processing sample {i+1}/{initial_total}")
            
            pred_answer = sample.get("pred_answer", "")
            if isinstance(pred_answer, str) and len(pred_answer.strip()) == 1:
                simple_cases.append(sample)
            else:
                complex_cases.append(sample)
            
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

        # Transform BoN format to judge evaluation format (only complex cases)
        judge_samples = []
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
            judge_item = result.copy()
            judge_item["prediction_full_text"] = prediction_text.strip()
            
            judge_samples.append(judge_item)
        
        # print(f"Transformed {len(judge_samples)} complex cases to MMMU format")

        # Initialize judge_eval_results as empty list
        judge_eval_results = []

        # print(f"Judge samples: {len(judge_samples)}")
        # print(f"Judge sample[0] {judge_samples[0]}")
        # print(f"Judge sample[0].keys() {judge_samples[0].keys()}")
        # exit()
        # Only process complex cases if they exist
        if judge_samples:
            # Run threaded evaluation
            with ThreadPoolExecutor(max_workers=200) as executor:
                for result in tqdm(
                    executor.map(eval_single_sample, judge_samples),
                    total=len(judge_samples),
                    desc="Judge Evaluation",
                ):
                    judge_eval_results.append(result)
        else:
            print("No complex cases requiring judge evaluation - all samples handled by simple string matching")
            
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
            # pred_answer = sample.get("pred_answer", "")
            # gt_answer = sample.get("gt_answer", "")
            
            # options = sample.get("annotation", {}).get("options", [])
            # valid_alpha_options = [f"{chr(65+i)}" for i in range(len(options))]
            # # Count initial correct answers (before judge)

            # initial_correct_this_sample = pred_answer.lower() == gt_answer.lower()
            # if initial_correct_this_sample:
            #     initial_correct += 1
            #     print(
            #         f"pred_answer: {pred_answer} is a single letter and is correct in valid_alpha_options: {valid_alpha_options}"
            #     )
            #     continue

            # # try to extract answer between \boxed{}
            
            # if pred_answer and len(pred_answer) != 1:
            #     print(f"pred_answer: {pred_answer} is not a single letter")
            #     print(f"options: {options}")
            #     print()
            #     print(f"pred_answer: {pred_answer}")
            #     print(f"gt_answer: {gt_answer}")

            #     print("Running LLM Judge workflow for this sample")
            
            # # Apply judge if initial extraction failed
            #     final_pred_answer = pred_answer
            #     judge_applied = True
            #     judge_processed += 1
                
                # Run LLM judge
        #         extracted_answer, success = judge_model.judge_single_sample(sample)
                
        #         if success:
        #             final_pred_answer = extracted_answer
        #             # Check if judge improved the result
        #             if extracted_answer == gt_answer:
        #                 judge_improved += 1
        #         else:
        #             final_pred_answer = sample.get("pred_answer", "")

        #         # Create updated sample with judge result
        #         updated_sample = sample.copy()
        #         updated_sample["pred_answer_after_judge"] = final_pred_answer
        #         updated_sample["judge_applied"] = judge_applied
        #         updated_results.append(updated_sample)
        
        # # Calculate final statistics
        # final_correct = 0
        # for sample in updated_results:
        #     final_pred = sample.get("pred_answer_after_judge", sample.get("pred_answer", ""))
        #     gt_answer = sample.get("gt_answer", "")
        #     if final_pred == gt_answer:
        #         final_correct += 1
        
        # final_accuracy = final_correct + initial_correct / initial_total if initial_total > 0 else 0.0
        # initial_accuracy = initial_correct / initial_total if initial_total > 0 else 0.0
        
        # Save updated results with judge annotations
        # judge_results_file = results_file.replace(".json", "_with_judge.json")
        # with open(judge_results_file, 'w', encoding='utf-8') as f:
        #     json.dump(updated_results, f, ensure_ascii=False, indent=2)
        
        # # Log detailed results
        # print("\nPuzzleVQA LLM Judge Evaluation Results:")
        # print(f"Initial accuracy (regex only): {initial_correct}/{initial_total} = {initial_accuracy:.2%}")
        # print(f"Final accuracy (with judge): {final_correct + initial_correct}/{initial_total} = {final_accuracy:.2%}")
        # print(f"Judge processed: {judge_processed} failed extractions")
        # print(f"Judge improved: {judge_improved} samples")
        # print(f"Judge results saved to: {judge_results_file}")
        
        # Return results in same format as MMMU judge
        # return {
        #     "overall_accuracy": final_accuracy,
        #     "num_correct_samples_after_llm_judgement": final_correct + initial_correct,
        #     "num_samples_after_llm_judgement": initial_total,
        #     "initial_correct_count": initial_correct,
        #     "initial_total_count": initial_total,
        #     "judge_improved_count": judge_improved,
        #     "judge_processed_count": judge_processed
        # }
        
    except Exception as e:
        print(f"Error in judge evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """Test the PuzzleVQA judge evaluation on a results file."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python puzzlevqa_judge_evaluation.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    print(f"Testing PuzzleVQA judge evaluation on: {results_file}")
    judge_results = calculate_puzzlevqa_judge_evaluation_score(results_file)
    
    if judge_results:
        print("\nJudge evaluation completed successfully!")
        # for key, value in judge_results.items():
        #     print(f"  {key}: {value}")
    else:
        print("Judge evaluation failed!")
        sys.exit(1)