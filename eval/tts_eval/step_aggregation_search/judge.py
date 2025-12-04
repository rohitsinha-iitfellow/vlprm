# try:
#     from .logger import log_info
# except ImportError:
#     # Handle case when run as script
#     from logger import log_info
# from datasets import load_dataset
# import pandas as pd
import json
# import re
# from Levenshtein import distance
# from utils import extract_boxed
import os
import sys
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import dotenv
dotenv.load_dotenv()

if "OPENAI_API_KEY" in os.environ:
    print(os.environ["OPENAI_API_KEY"][:5])
else:
    print("OPENAI_API_KEY not found")
    exit()

class MathVistaJudgeModel:
    """
    LLM Judge model for MathVista answer extraction fallback.
    
    Implements the exact ChainThoughtMultiExtractPrompter logic from the
    reference MathVista implementation.
    """

    def __init__(self, model_path: str = "gpt-4.1", temperature: float = 0.0):
        """
        Initialize the MathVista judge model.
        
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
    
    def judge_single_sample(self, question, options, reasoning_trajectory) -> Tuple[str, bool]:
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
            # original_question = sample_data.get("question", "")
            # final_steps = sample_data.get("final_steps", [])
            # annotation = sample_data.get("annotation", {})
            # reasoning_trajectory = sample_data.get("prediction_full_text", "")
            
            # Get options from annotation
            # options = annotation.get("choices", [])
            # if not options:
            #     return "Cannot extract: No choices found", False
            
            # Get the final reasoning step (last step in trajectory)
            # if not final_steps:
            #     return "Cannot extract: No reasoning steps found", False
            
            # reasoning_trajectory = final_steps[-1] if isinstance(final_steps, list) else str(final_steps)
            
            # Reconstruct original prompt (approximation of MathVista format)
            judge_prompt = self._construct_judge_prompt(
                question, options, reasoning_trajectory
            )
            
            # Build extraction prompt
            # extraction_prompt = self.build_extraction_prompt(original_prompt, reasoning_trajectory, options)
            
            # Run judge inference
            print(f"Running judge inference with prompt: {judge_prompt}")
            response = self.model.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "text"},
                temperature=self.temperature,
                max_completion_tokens=256,
            )
            judge_output = response.choices[0].message.content.strip()

            print(f"returned Judge output: {judge_output}")

            if options:
                alphabetical_options = [f"{chr(65 + i)}" for i in range(len(options))]
                valid_responses = alphabetical_options + ["Z"]
                if judge_output in valid_responses:
                    extraction_method = "Valid alphabetical option"
                else:
                    extraction_method = "Invalid alphabetical option"
            elif not options:
                extraction_method = "Open-ended answer extraction"

            output = {
                "judge_output": judge_output,
                "extraction_method": extraction_method,
                "judge_prompt": judge_prompt,
            }
            return output
            
        except Exception as e:
            print(f"Error in judge_single_sample: {e}")
            return {
                "judge_output": f"Judge error: {str(e)}",
                "extraction_method": "Judge error",
                "judge_prompt": "",
            }

    def _construct_judge_prompt(
        self, question: str, options: List[str], reasoning_trajectory: str
    ) -> str:
        """
        Reconstruct the original MathVista prompt format.
        
        Args:
            question: The question text
            options: List of answer options
            
        Returns:
            Reconstructed prompt in MathVista format
        """
        if options and len(options) > 0:
            print(f"MCQ choices provided: {options}")

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
        elif not options:
            print("No MCQ choices provided, running open-ended answer extraction")
            judge_prompt = (
                "You are an AI assistant who will help me to extract "
                "the final answer from a detailed response. "
                "You are provided with a question and a response that may contain reasoning, explanations, or calculations, "
                "and you need to extract only the final answer. "
                "IMPORTANT: For numeric answers, extract ONLY the number WITHOUT any units, symbols, or text. "
                "For non-numeric answers, extract the most direct and concise form. "
                "Your output should be brief and to the point. \n"
                "Example 1: \n"
                "Question: What is 15 + 27?\n"
                "Answer: Let me calculate this step by step. First, I'll add 15 + 27. 15 + 20 = 35, then 35 + 7 = 42. Therefore, the answer is 42.\n"
                "Your output: 42\n"
                "Example 2: \n"
                "Question: How old is the tree?\n"
                "Answer: Based on the rings I can count in the cross-section, the tree appears to be approximately 20 years old.\n"
                "Your output: 20\n"
                "Example 3: \n"
                "Question: What is the speed of the car?\n"
                "Answer: Looking at the speedometer in the image, the car is traveling at 65 mph.\n"
                "Your output: 65\n"
                "Example 4: \n"
                "Question: What color is the sky in the image?\n"
                "Answer: Looking at the image, I can see a clear day with a bright blue sky. The sky appears to be blue.\n"
                "Your output: blue\n"
                "Example 5: \n"
                "Question: How many apples are there?\n"
                "Answer: I need to count the apples. I see 3 on the left side and 2 on the right side. In total, there are 5 apples.\n"
                "Your output: 5\n"
                "Example 6: \n"
                "Question: {}?\n"
                "Answer: {}\n"
                "Your output: "
            )
            judge_prompt = judge_prompt.format(question, reasoning_trajectory)
        else:
            raise ValueError(f"Invalid number of options: {len(options)}")

        return judge_prompt

class MiniCPMJudgeModel:
    """
    LLM Judge model for MathVista answer extraction fallback.
    
    Implements the exact ChainThoughtMultiExtractPrompter logic from the
    reference MathVista implementation.
    """

    def __init__(self, model_path: str = "gpt-4.1", temperature: float = 0.0):
        """
        Initialize the MathVista judge model.
        
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
    
    def judge_single_sample(self, question, options, reasoning_trajectory) -> Tuple[str, bool]:
        """
        Run LLM judge on a single sample to extract the answer.
        
        Args:
            sample_data: Sample data dictionary from results file
            
        Returns:
            Tuple of (extracted_answer, success_flag)
        """
        self.load_model()
        
        try:
            # Reconstruct original prompt (approximation of MathVista format)
            judge_prompt = self._construct_judge_prompt(
                question, options, reasoning_trajectory
            )
             
            # Run judge inference
            print(f"Running judge inference with prompt: {judge_prompt}")
            response = self.model.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "text"},
                temperature=self.temperature,
                max_completion_tokens=256,
            )
            judge_output = response.choices[0].message.content.strip()

            print(f"returned Judge output: {judge_output}")

            if options:
                alphabetical_options = [f"{chr(65 + i)}" for i in range(len(options))]
                valid_responses = alphabetical_options + ["Z"]
                if judge_output in valid_responses:
                    extraction_method = "Valid alphabetical option"
                else:
                    extraction_method = "Invalid alphabetical option"
            elif not options:
                extraction_method = "Open-ended answer extraction"

            output = {
                "judge_output": judge_output,
                "extraction_method": extraction_method,
                "judge_prompt": judge_prompt,
            }
            return output
            
        except Exception as e:
            print(f"Error in judge_single_sample: {e}")
            return {
                "judge_output": f"Judge error: {str(e)}",
                "extraction_method": "Judge error",
                "judge_prompt": "",
            }

    def _construct_judge_prompt(
        self, question: str, options: List[str], reasoning_trajectory: str
    ) -> str:
        """
        Reconstruct the original MathVista prompt format.
        
        Args:
            question: The question text
            options: List of answer options
            
        Returns:
            Reconstructed prompt in MathVista format
        """
        if options and len(options) > 0:
            print(f"MCQ choices provided: {options}")

            judge_prompt = (
                "You are an AI assistant who will help me to match "
                "an answer with several options of a single-choice question. "
                "You are provided with a question, several options, and an answer response, "
                "and you need to find which option is most similar to the response. "
                "If the meaning of all options are significantly different from the given response, output Z. "
                "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
                "Example 1: \n"
                "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
                "Response: a cute teddy bear\nYour output: A\n"
                "Example 2: \n"
                "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
                "Response: Spider\nYour output: Z\n"
                "Example 3: \n"
                "Question: What fraction of the shape is blue?\nOptions: A. 3/11 B. 8/11 C. 6/11 D. 3/5\n"
                "Response: Answer: (B) 8/11.\nYour output: B\n"
                "Question: {}?\nOptions: {}\nResponse: {}\nYour output: "
            )
            if ''.join(options) != 'ABCDE':
                formatted_options = " ".join(
                    [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
                )
            else:
                formatted_options = " ".join(
                    [f"{option}" for i, option in enumerate(options)]
                )

            judge_prompt = judge_prompt.format(
                question, formatted_options, reasoning_trajectory
            )
        elif not options:
            print("No MCQ choices provided, running open-ended answer extraction")
            judge_prompt = (
                "You are an AI assistant who will help me to extract "
                "the final answer from a detailed response. "
                "You are provided with a question and a response that may contain reasoning, explanations, or calculations, "
                "and you need to extract only the final answer. "
                "IMPORTANT: For numeric answers, extract ONLY the number WITHOUT any units, symbols, or text. "
                "For non-numeric answers, extract the most direct and concise form. "
                "Your output should be brief and to the point. \n"
                "Example 1: \n"
                "Question: What is 15 + 27?\n"
                "Response: Let me calculate this step by step. First, I'll add 15 + 27. 15 + 20 = 35, then 35 + 7 = 42. Therefore, the answer is 42.\n"
                "Your output: 42\n"
                "Example 2: \n"
                "Question: How old is the tree?\n"
                "Response: Based on the rings I can count in the cross-section, the tree appears to be approximately 20 years old.\n"
                "Your output: 20\n"
                "Example 3: \n"
                "Question: What is the speed of the car?\n"
                "Response: Looking at the speedometer in the image, the car is traveling at 65 mph.\n"
                "Your output: 65\n"
                "Example 4: \n"
                "Question: What color is the sky in the image?\n"
                "Response: Looking at the image, I can see a clear day with a bright blue sky. The sky appears to be blue.\n"
                "Your output: blue\n"
                "Example 5: \n"
                "Question: How many apples are there?\n"
                "Response: I need to count the apples. I see 3 on the left side and 2 on the right side. In total, there are 5 apples.\n"
                "Your output: 5\n"
                "Example 6: \n"
                "Question: {}?\n"
                "Response: {}\n"
                "Your output: "
            )
            judge_prompt = judge_prompt.format(question, reasoning_trajectory)
        else:
            raise ValueError(f"Invalid number of options: {len(options)}")

        return judge_prompt

def is_answer_match(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a.lower() == b.lower(): return True
    try: return abs(float(a) - float(b)) < 1e-10
    except: return False

def eval_single_sample(item):
    """Evaluate a single sample."""

    judge_model = MathVistaJudgeModel(model_path="gpt-4.1", temperature=0.0)

    try:
        output = judge_model.judge_single_sample(item)
        extracted_answer = output["judge_output"]
        extraction_method = output["extraction_method"]
        judge_prompt = output["judge_prompt"]
    except Exception as e:
        print(f"Error in judge_single_sample: {e}")
        return {
            "judge_output": f"Judge error: {str(e)}",
            "extraction_method": "Judge error",
            "judge_prompt": "",
        }

    # Extract answer using the combined approach
    # result = extract_answer_from_item(model, item)
    
    # Determine if the answer is correct
    # hit = 1 if extracted_answer == item['gt_answer'] else 0
    hit = 1 if is_answer_match(extracted_answer, item['gt_answer']) else 0
    print(f"extracted_answer: {extracted_answer}, item['gt_answer']: {item['gt_answer']}, hit: {hit}")
    print(f"judge_prompt for above extracted answer: {judge_prompt}")
    
    return {
        "index": item["index"],
        "split": "Judge Evaluation (MCQ and Open-Ended Float/Integer)", # Mathvista open-ended is ONLY float or Integer
        "question": item["question"],
        "prediction": item["prediction_full_text"],
        "extracted_answer": extracted_answer,
        "extraction_model": judge_model.model_path
        + "-temperature-"
        + str(judge_model.temperature)
        if judge_model.model_path and judge_model.temperature is not None
        else "gpt-4.1-judge-temperature-0.0",
        # "extraction_success": success,
        "extraction_method": extraction_method,
        "extraction_log": "Judge output: " + extracted_answer,
        "gt": item["gt_answer"],
        "hit": hit,
    }

def calculate_mathvista_judge_evaluation_score(results_file: str) -> Optional[Dict]:
    """
    Calculate MathVista evaluation scores with LLM Judge fallback mechanism.
    
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

        # Process each sample
        print(f"Processing {initial_total} samples for judge evaluation...")
        simple_mcq_cases = []
        simple_string_matching_cases = []
        complex_cases = []

        # filter out null results
        filtered_results = [result for result in results if result.get("pred_answer") is not None]
        # null_results = [
        #     result for result in results if result.get("pred_answer") is None
        # ] # TODO: Null results are caused not by failed \boxed extraction but by LLM just not answering (reason still to be investigated, so we can skip for now)

        # print(f"Null evaluation results: {len(null_results)}")
        # null_eval_results = evaluate_null_results(null_results)
        # exit()
        # print(f"After filtering out null results: {len(filtered_results)}")

        print("First filter out results where simple matching does not work")
        for i, sample in enumerate(filtered_results):
            if i % 50 == 0:  # Progress logging
                print(f"Processing sample {i+1}/{initial_total}")

            # Should have filtered out null results by now
            pred_answer = sample.get("pred_answer", "").strip()
            gt_answer = sample.get("gt_answer", "").strip()

            choices = sample.get("annotation", {}).get("choices", [])
            if choices and len(choices) > 0:
                valid_alpha_options = [f"{chr(65 + i)}" for i in range(len(choices))]

            # handle pred_answer for MCQ and Open-Ended separately
            if len(pred_answer) == 1 and pred_answer in valid_alpha_options:
                simple_mcq_cases.append(sample) # filters out all MCQ cases, correct or wrong, assume format of choices and answer is consistent**
            elif len(pred_answer) > 0 and pred_answer.lower() == gt_answer.lower(): # filters out all open-ended cases that already match and are correct
                simple_string_matching_cases.append(sample)
            else:
                complex_cases.append(sample) # contains MCQ letter and open-ended cases, have to manage separately (MCQ extract and match with options, Open-Ended extract seems to all be integer/float) - Extract and Judge?

            
        # print(f"Simple cases (len=1 pred_answer): {len(simple_mcq_cases)}")
        # print(f"Simple cases (len=1 pred_answer): {len(simple_string_matching_cases)}")
        # print(f"Complex cases (requiring judge evaluation): {len(complex_cases)}")
        # print([result.get("pred_answer") for result in simple_mcq_cases])
        # print([result.get("pred_answer") for result in simple_string_matching_cases])
        # print([result.get("pred_answer") for result in complex_cases])
        # exit()
        # Evaluate simple cases with direct string matching
        simple_cases = simple_mcq_cases + simple_string_matching_cases

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
                "split": "simple cases (mcq and string matching)",
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
        prediction_text = ""
        for result in complex_cases:
            # Join final_steps into single prediction string, with fallback to pred_answer
            if result.get("final_steps") and len(result["final_steps"]) > 0:
                prediction_text = "\n\n".join(result["final_steps"])
            elif result.get("pred_answer") and not prediction_text:
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
        null_results = [r for r in results if r.get('pred_answer') is None]
        print(f"Merged results: {len(simple_eval_results)} simple + {len(judge_eval_results)} judge + {len(null_results)} null = {len(results)} total")
        
        # Calculate overall accuracy
        accuracy = sum(r['hit'] for r in eval_results) / len(results)
        
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
        
        print(f"Overall accuracy: {accuracy:.4f} ({sum(r['hit'] for r in eval_results)}/{len(results)})")
        
        # # Show breakdown by evaluation method
        # simple_results = [r for r in eval_results if r.get('extraction_method') == 'simple_string_match']
        # judge_results = [r for r in eval_results if r.get('extraction_method') != 'simple_string_match']
        
        # if simple_results:
        #     simple_acc = sum(r['hit'] for r in simple_results) / len(simple_results)
        #     print(f"Simple string matching accuracy: {simple_acc:.4f} ({sum(r['hit'] for r in simple_results)}/{len(simple_results)})")
        
        # if judge_results:
        #     judge_acc = sum(r['hit'] for r in judge_results) / len(judge_results)
        #     print(f"Judge model evaluation accuracy: {judge_acc:.4f} ({sum(r['hit'] for r in judge_results)}/{len(judge_results)})")
        
        return {
            "overall_accuracy": accuracy,
            "accuracy_by_split": accuracy_by_split,
            "eval_results": eval_results,
            "num_correct_samples_after_llm_judgement": sum(r['hit'] for r in eval_results),
            "num_samples_after_llm_judgement": len(results) # careful with this, make sure check how NULL values are handled and sums to full dataset
        }
    except Exception as e:
        print(f"Error in judge evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """Test the MathVista judge evaluation on a results file."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python mathvista_helper_functions.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    print(f"Testing MathVista judge evaluation on: {results_file}")
    judge_results = calculate_mathvista_judge_evaluation_score(results_file)
    
    if judge_results:
        print("\nJudge evaluation completed successfully!")
    else:
        print("Judge evaluation failed!")
        sys.exit(1)