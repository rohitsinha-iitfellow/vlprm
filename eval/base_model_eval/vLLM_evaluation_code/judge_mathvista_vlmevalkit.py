try:
    from .logger import log_info
except ImportError:
    # Handle case when run as script
    from logger import log_info
# from evaluation.common.logger import log_info

# from datasets import load_dataset
# import pandas as pd
import json

# import re
# from Levenshtein import distance
# from utils import extract_boxed
import os
import sys
from typing import Dict, List, Optional, Tuple
from vlmkit_judge_extraction import post_check
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import dotenv

dotenv.load_dotenv()

if "OPENAI_API_KEY" in os.environ:
    log_info(os.environ["OPENAI_API_KEY"][:5])
else:
    log_info("OPENAI_API_KEY not found")
    exit()


import math


def is_nan_value(value):
    """Check if value is any form of NaN/null"""
    if value is None:
        return True
    if isinstance(value, str) and value.lower() in ["nan", "null", ""]:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


class MathVistaJudgeModelVLMEvalKit:
    """
    LLM Judge model for MathVista answer extraction fallback.

    Implements the exact ChainThoughtMultiExtractPrompter logic from the
    reference MathVista implementation.
    """

    def __init__(self, model_path: str = "gpt-4.1-mini", temperature: float = 0.0):
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
            # final_steps = sample_data.get("final_steps", [])
            annotation = sample_data.get("annotation", {})
            reasoning_trajectory = sample_data.get("prediction_full_text", "")

            # Get options from annotation
            choices_raw = annotation.get("choices", [])

            # Parse choices from JSON string format
            if isinstance(choices_raw, str) and not is_nan_value(choices_raw):
                try:
                    import ast

                    options = ast.literal_eval(choices_raw) # parse the string to an actual list of options
                except (ValueError, SyntaxError):
                    raise ValueError(f"Failed to parse a valid list of options: {choices_raw}")
            elif isinstance(choices_raw, list):
                options = choices_raw
            elif is_nan_value(choices_raw):  # Check for NaN
                options = []
            else:
                raise ValueError(f"choices_raw is not a list or a string and not NaN: {choices_raw}")
            # if not options:
            #     return "Cannot extract: No choices found", False

            # Get the final reasoning step (last step in trajectory)
            # if not final_steps:
            #     return "Cannot extract: No reasoning steps found", False

            # reasoning_trajectory = final_steps[-1] if isinstance(final_steps, list) else str(final_steps)

            # Reconstruct original prompt (approximation of MathVista format)
            judge_prompt = self._construct_judge_prompt(
                original_question, options, reasoning_trajectory
            )

            # Build extraction prompt
            # extraction_prompt = self.build_extraction_prompt(original_prompt, reasoning_trajectory, options)

            # Run judge inference
            # log_info(f"Running judge inference with prompt: {judge_prompt}")
            response = self.model.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "text"},
                temperature=self.temperature,
                max_completion_tokens=256,
            )
            judge_output = response.choices[0].message.content.strip()

            # log_info(f"returned Judge output: {judge_output}")

            if options:
                alphabetical_options = [f"{chr(65 + i)}" for i in range(len(options))]
                valid_responses = alphabetical_options + ["Z"]
                if judge_output in valid_responses:
                    extraction_method = "Valid alphabetical option"
                else:
                    extraction_method = "Invalid alphabetical option"
            elif not options:
                extraction_method = "Open-ended answer extraction"

            # if judge_output in valid_responses:
            #     extracted_answer = judge_output
            #     extraction_method = True
            # else:
            #     extracted_answer = judge_output
            #     extraction_method = False
            output = {
                "judge_output": judge_output,
                "extraction_method": extraction_method,
                "reasoning_trajectory": reasoning_trajectory,
            }

            return output

        except Exception as e:
            log_info(f"Error in judge_single_sample: {e}")
            return {
                "judge_output": f"Judge error: {str(e)}",
                "extraction_method": "Judge error",
                "reasoning_trajectory": "",
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
        example_1 = """
    Hint: Please answer the question requiring an integer answer and provide the final value,
    e.g., 1, 2, 3, at the end.\n
    Question: Which number is missing?\n
    Model response: The number missing in the sequence is 14.\n
    Extracted answer: 14
    """

        example_2 = """
    Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
    e.g., 1.2, 1.3, 1.4, at the end.\n
    Question: What is the fraction of females facing the camera?\n
    Model response: The fraction of females facing the camera is 0.6,
    which means that six out of ten females in the group are facing the camera.\n
    Extracted answer: 0.6
    """

        example_3 = """
    Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
    e.g., 1.23, 1.34, 1.45, at the end.\n
    Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
    Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
    Extracted answer: 1.45
    """

        example_4 = """
    Hint: Please answer the question requiring a Python list as an answer and provide the final list,
    e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
    Question: Between which two years does the line graph saw its maximum peak?\n
    Model response: The line graph saw its maximum peak between 2007 and 2008.\n
    Extracted answer: [2007, 2008]
    """

        example_5 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: What fraction of the shape is blue?\n
    Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
    Model response: The correct answer is (B) 8/11.\n
    Extracted answer: B
    """

        ice_examples = [example_1, example_2, example_3, example_4, example_5]

        task_description = """
    Please read the following example.
    Then extract the answer from the model response and type it at the end of the prompt.\n
    """
        # question = line['question']
        # prediction = str(line['prediction'])
        judge_prompt = task_description
        # examples = get_gpt4_ICE()
        for example in ice_examples:
            judge_prompt += example + '\n'
        judge_prompt += question + '\n'
        judge_prompt += 'Model response: ' + reasoning_trajectory
        judge_prompt += 'Extracted answer:'
        return judge_prompt

def is_answer_match(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a.lower() == b.lower():
        return True
    try:
        return abs(float(a) - float(b)) < 1e-10
    except:
        return False


def eval_single_sample(item):
    """Evaluate a single sample."""

    judge_model = MathVistaJudgeModelVLMEvalKit(model_path="gpt-4o-mini", temperature=0.0)

    try:
        output = judge_model.judge_single_sample(item)
        extracted_answer = output["judge_output"]
        extraction_method = output["extraction_method"]
    except Exception as e:
        log_info(f"Error in judge_single_sample: {e}")
        return {
            "judge_output": f"Judge error: {str(e)}",
            "extraction_method": "Judge error",
            "reasoning_trajectory": "",
        }

    # Extract answer using the combined approach
    # result = extract_answer_from_item(model, item)

    # Determine if the answer is correct
    hit = 1 if extracted_answer == item['gt_answer'] else 0
    # hit = 1 if is_answer_match(extracted_answer, item["gt_answer"]) else 0
    log_info(
        f"extracted_answer: {extracted_answer}, item['gt_answer']: {item['gt_answer']}, hit: {hit}"
    )
    log_info(f"reasoning_trajectory: {output['reasoning_trajectory']}")

    return {
        "index": item["index"],
        "split": item.get("split")
        or "Judge Evaluation (MCQ and Open-Ended Float/Integer)",  # Mathvista open-ended is ONLY float or Integer
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


def evaluate_null_results(null_result_items):
    simple_mcq_cases = []
    simple_string_matching_cases = []
    complex_cases = []
    additional_correct_cases_array = []
    number_of_additional_correct_cases = 0
    number_of_additional_incorrect_cases = 0

    for i, sample in enumerate(null_result_items):
        if i % 50 == 0:  # Progress logging
            log_info(f"Processing sample {i + 1}/{len(null_result_items)}")

        # Should have filtered out null results by now
        pred_answer = sample.get("raw_full_prediction", "").strip()
        question_type = sample.get("annotation", {}).get("question_type", "")

        # Debug: Print question_type to see what we're actually getting
        # if i < 5:  # Only print for first 5 samples to avoid spam
        #     log_info(
        #         f"Sample {i}: question_type = '{question_type}' (repr: {repr(question_type)})"
        #     )

        if question_type == "multi_choice":
            gt_answer = sample.get("annotation", {}).get("answer_option", "")
            # if i < 5:
            #     log_info(
            #         f"Sample {i}: Using multi_choice path, answer_option = '{gt_answer}'"
            #     )
        elif question_type == "free_form":
            gt_answer = sample.get("annotation", {}).get("answer", "")
            # if i < 5:
            #     log_info(f"Sample {i}: Using else path, answer = '{gt_answer}'")
        # if is_nan_value(gt_answer):
        #     print(f"gt_answer is NaN, using answer instead")
        #     gt_answer = sample.get("annotation", {}).get("answer", "")
        #     print(f"gt_answer: {gt_answer}")
        # exit()

        # valid_alpha_options = []
        choices_raw = sample.get("annotation", {}).get("choices", [])
        print(f"choices_raw: {choices_raw}")

        # Parse choices from JSON string format
        if isinstance(choices_raw, str) and not is_nan_value(choices_raw):
            try:
                import ast

                choices = ast.literal_eval(choices_raw)
            except (ValueError, SyntaxError):
                raise ValueError(f"Failed to parse a valid list of options: {choices_raw}")
        elif isinstance(choices_raw, list):
            choices = choices_raw
        elif is_nan_value(choices_raw):  # Check for NaN
            choices = []
        else:
            raise ValueError(f"choices_raw is not a list or a string and not NaN: {choices_raw}")

        print(f"parsed choices: {choices}")
        # if choices and len(choices) > 0:
        #     valid_alpha_options = [f"{chr(65 + i)}" for i in range(len(choices))]
        complex_cases.append(sample)
        # handle pred_answer for MCQ and Open-Ended separately
        # log_info(f"pred_answer: {pred_answer}")
        # log_info(f"gt_answer: {gt_answer}")
        # if post_check(sample, prefetch=True) is not None:
        #     post_check_processing_answer = post_check(sample, prefetch=True)
        #     if post_check_processing_answer == gt_answer:
        #         number_of_additional_correct_cases += 1
        #         additional_correct_cases_array.append(sample)
        #     else:
        #         number_of_additional_incorrect_cases += 1
        #         # additional_incorrect_cases_array.append(sample)
        # else:
            # complex_cases.append(
            #     sample
            # )  # contains MCQ letter and open-ended cases, have to manage separately (MCQ extract and match with options, Open-Ended extract seems to all be integer/float) - Extract and Judge?

    # print(f"Simple cases (len=1 pred_answer): {len(simple_mcq_cases)}")
    # print(f"Simple cases (len=1 pred_answer): {len(simple_string_matching_cases)}")
    # print(f"Complex cases (requiring judge evaluation): {len(complex_cases)}")
    # print([result.get("pred_answer") for result in simple_mcq_cases])
    # print([result.get("pred_answer") for result in simple_string_matching_cases])
    # print([result.get("pred_answer") for result in complex_cases])
    # exit()
    # Evaluate simple cases with direct string matching
    # print("number_of_additional_correct_cases: ", number_of_additional_correct_cases)
    # print("number_of_additional_incorrect_cases: ", number_of_additional_incorrect_cases)
    print("number of complex cases: ", len(complex_cases))
    print("total number of cases processed: ", len(complex_cases))
    # exit()
    # print(additional_correct_cases_array[0])
    # exit()
    # simple_cases = simple_mcq_cases + simple_string_matching_cases

    # simple_eval_results = []
    # simple_correct = 0
    # for result in simple_cases:
    #     pred_answer = result.get(
    #         "raw_full_prediction", ""
    #     ).strip()  # sometimes raw_full_prediction answers the letter directly, resulting in None from \boxed extraction
    #     # gt_answer = result.get("gt_answer", "").strip()
    #     # exit()

    #     # Case-insensitive comparison
    #     hit = 1 if is_answer_match(pred_answer, gt_answer) else 0
    #     simple_correct += hit

    #     # Create evaluation result in same format as judge evaluation
    #     eval_result = {
    #         "index": result.get("index", "unknown"),
    #         "split": "null result cases (mcq and string matching)",
    #         "question": result.get("annotation", {}).get("question", ""),
    #         "prediction": result.get("raw_full_prediction", "").strip(),
    #         "extracted_answer": result.get("pred_answer", ""),  # should be null
    #         "extraction_method": "simple_string_match",
    #         "extraction_success": True,
    #         "extraction_log": f"Simple string match: {pred_answer} vs {gt_answer}",
    #         "gt": gt_answer,
    #         "hit": hit,
    #     }
    #     simple_eval_results.append(eval_result)

    # if simple_cases:
    #     simple_accuracy = simple_correct / len(simple_cases)
    #     log_info(
    #         f"Simple evaluation accuracy: {simple_accuracy:.4f} ({simple_correct}/{len(simple_cases)})"
    #     )

    # Transform BoN format to judge evaluation format (only complex cases)
    judge_samples = []
    for result in complex_cases:
        # Join final_steps into single prediction string, with fallback to pred_answer
        # if result.get("final_steps") and len(result["final_steps"]) > 0:
        #     prediction_text = "\n\n".join(result["final_steps"])
        if result.get("raw_full_prediction"):
            prediction_text = str(result["raw_full_prediction"])
        else:
            raise ValueError(f"No prediction text found: {result}")

        question_type = result.get("annotation", {}).get("question_type", "")
        gt_answer = None
        if question_type == "multi_choice":
            gt_answer = result.get("annotation", {}).get("answer_option", "")
        elif question_type == "free_form":
            gt_answer = result.get("annotation", {}).get("answer", "")
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        # print("result:", result)
        # print("prediction_text:", prediction_text)
        # Create MMMU-compatible annotation
        judge_item = result.copy()
        judge_item["prediction_full_text"] = prediction_text.strip()
        judge_item["gt_answer"] = gt_answer
        judge_item["split"] = "null result cases (using judge extraction)"
    
        # print(judge_item)
        # exit()
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
        log_info(
            "No complex cases requiring judge evaluation - all samples handled by simple string matching"
        )
    # print(f"Number of additional correct cases: {number_of_additional_correct_cases}")
    # print(f"Number of additional incorrect cases: {number_of_additional_incorrect_cases}")
    # print(f"Number of additional cases: {number_of_additional_correct_cases + number_of_additional_incorrect_cases}")
    # eval_results = simple_eval_results + judge_eval_results
    return judge_eval_results

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
    log_info("Starting LLM Judge evaluation...")

    try:
        # Load results file
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        if not results:
            log_info("Warning: Empty results file")
            return None

        # Initialize counters
        # initial_correct = 0
        initial_total = len(results)
        # judge_processed = 0
        # judge_improved = 0

        # Process each sample
        log_info(f"Processing {initial_total} samples for judge evaluation...")
        null_results = [
            result for result in results if result.get("pred_answer") is None
        ]

        null_eval_results = evaluate_null_results(null_results)
        log_info(f"Null judge evaluation results: {len(null_eval_results)}")
        split_accuracy = sum(r["hit"] for r in null_eval_results) / len(null_eval_results)
        log_info(f"Null judge evaluation split accuracy: {split_accuracy:.4f} ({sum(r['hit'] for r in null_eval_results)}/{len(null_eval_results)})")
    except Exception as e:
        log_info(f"Error in judge evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    """Test the MathVista judge evaluation on a results file."""
    import sys

    if len(sys.argv) != 2:
        log_info("Usage: python mathvista_helper_functions.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        log_info(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    log_info(f"Testing MathVista judge evaluation on: {results_file}")
    judge_results = calculate_mathvista_judge_evaluation_score(results_file)
    
    if judge_results:
        log_info("\nJudge evaluation completed successfully!")
        # for key, value in judge_results.items():
        #     print(f"  {key}: {value}")
    else:
        log_info("Judge evaluation failed!")
        sys.exit(1)