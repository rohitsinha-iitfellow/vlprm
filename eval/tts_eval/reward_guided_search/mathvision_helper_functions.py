try:
    from .logger import log_info
except ImportError:
    # Handle case when run as script
    from utils.logger import log_info
from datasets import load_dataset
import pandas as pd
import re

import json
from tqdm import tqdm
# from latex2sympy2 import latex2sympy # TODO: remember to activate when using MathVision as Judge
import time 
from math import *

import json
from tqdm import tqdm  # Assuming tqdm is imported to show progress bar
import time

import os
import math
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import dotenv
import json
dotenv.load_dotenv()

if "OPENAI_API_KEY" in os.environ:
    log_info(os.environ["OPENAI_API_KEY"][:5])
else:
    log_info("OPENAI_API_KEY not found")
    exit()  

def load_mathvision_dataset(dataset_name='MathLLMs/MathVision', test_split_name='testmini'):
    """Load MathVista dataset from HuggingFace and convert to DataFrame format compatible with existing pipeline."""
    log_info(f"Loading MathVision dataset {dataset_name}, split {test_split_name}...")
    
    # Load from HuggingFace (following MathVista generate_response.py:108-113)
    data_list = load_dataset(dataset_name, split=test_split_name)
    
    # Convert to DataFrame format compatible with existing pipeline
    data_records = []
    for item in data_list:

        # Convert HF dataset item to record format
        record = {
            "index": item["id"],  # Use pid as index
            "id": item["id"],  # Also store as id for consistency
            "question": item["question"],
            "answer": item['answer'],
            "options": item.get("options", []),
            "level": item.get("level", ""),
            "subject": item.get("subject", ""),
            "solution": item.get("solution", ""),
        }
        
        # Convert PIL image to base64 string
        if 'decoded_image' in item and item['decoded_image'] is not None:
            # Convert PIL image to base64 string
            import io
            import base64
            buffered = io.BytesIO()
            item['decoded_image'].save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            record['image'] = img_base64
        else:
            record['image'] = None
            
        data_records.append(record)
    
    # Convert to DataFrame
    data_df = pd.DataFrame(data_records)
    log_info(f"Loaded {len(data_df)} MathVista samples")
    
    return data_df

# Define the function to process each example
def build_mathvision_prompt_original(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    return input

def build_mathvision_prompt_minicpm(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    input = 'Please solve the problem step by step and then output the final answer in the format of "Answer: single number or single word or phrase"\n\nWhen working out the reasoning / intermediate steps, 请一步步推理, especially when that helps clarity.\n\n' + f"{question}\n{options}"
    return input

def build_mathvision_prompt(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    user_prompt = ""

    user_prompt += f"Question: {question}\n"

    if options and len(options) > 0:
        user_prompt += f"Options:{options}"
        user_prompt += "Please select the correct answer from the options above.\n"

    # user_prompt += "If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering. If you do need to think step by step, first, carefully observe and describe everything you see: the image itself and how it connects to the problem being presented. Then, work through your reasoning step by step to arrive at your answer. **Your teacher will review your descriptions of visual elements to ensure you're observing all relevant details accurately, and will critique each step of your reasoning to provide guidance and ensure you're on the right track.** Put your final answer within \\boxed{}. If multiple-choice options for answers to the question are provided, select the alphabetical letter of the corresponding best option within \\boxed{} when you are ready to provide your final answer."

    # user_prompt = user_prompt.rstrip()
    return user_prompt

# Define the function to process each example
def build_mathvision_prompt_notags(example):
    question = example['question']
    cleaned_question = re.sub(r"<image\d+>", "", question)
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{cleaned_question}\n{options}"
    return input

# start copying from here
def is_nan_value(value):
    """Check if value is any form of NaN/null"""
    if value is None:
        return True
    if isinstance(value, str) and value.lower() in ["nan", "null", ""]:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


class JudgeExtractorVLMEvalKit:
    """
    LLM Judge model for MathVision answer extraction fallback.

    Implements the exact ChainThoughtMultiExtractPrompter logic from the
    reference MathVista implementation.
    """

    # def __init__(self, model_path: str = "gpt-4.o-mini", temperature: float = 0.0):
    def __init__(self, model_path: str = "gpt-4.1-mini", temperature: float = 0.0):
    # def __init__(self, model_path: str = "gpt-4.1", temperature: float = 0.0):
        """
        Initialize the MathVision judge model.

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
            self.model = OpenAI()

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
            choices_raw = annotation.get("options", [])

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

            judge_prompt = self._construct_judge_prompt(
                original_question, options, reasoning_trajectory
            )
            response = self.model.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "text"},
                temperature=self.temperature,
                max_completion_tokens=256,
            )
            judge_output = response.choices[0].message.content.strip()

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

        example_6 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: What are the equivalent units for conversion costs for each quarter using the weighted-average method? Assume that the quarters are independent.\n
    Choices: (A) 132,625 (B) 134,485 (C) 135,332 (D) 132,685\n
    Model response: Therefore, the correct option is 132,685.\n\nAnswer: D.\n
    Extracted answer: D
    """

        example_7 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: Calculate the missing values of company 2.\n
    Choices: (A) $1,620 (B) $12,000 (C) $51,180 (D) $0\n
    Model response: Therefore, the missing value for Co.2 is $51,180.\n\nAnswer: C.\n
    Extracted answer: C
    """

        example_8 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: The primary disaccharide digestion product of starch is\n
    Choices: (A) <image 1> (B) <image 2> (C) <image 3> (D) <image 4>\n
    Model response: Based on this analysis, the correct answer is the one that shows the structure of maltose.\n\nAnswer: A.\n
    Extracted answer: A
    """
        
        example_9 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?\n
    Choices: (A) $3 (B) $4 (C) $5 (D) $6\n
    Model response: Therefore, the per unit manufacturing overhead costs incurred when 30,000 units are produced is $5.\n\nAnswer: C.\n
    Extracted answer: C
    """

        ice_examples = [
            example_1,
            example_2,
            example_3,
            example_4,
            example_5,
            example_6,
            example_7,
            example_8,
            example_9,
        ]

        task_description = """
    Please read the following example.
    Then extract the answer from the model response and type it at the end of the prompt.\n
    """
        # Notice above examples have hints in them, not always true. Also they have Choices in their questions, not always true so we have to add the options string manually if so below, TODO: To check for every dataset how options work.
        judge_prompt = task_description
        for example in ice_examples:
            judge_prompt += example + '\n'
        judge_prompt += f"Question: {question}\n"

        if options and ''.join(options) != 'ABCDE':
            options_str = ""
            for i, option in enumerate(options):
                letter = chr(ord("A") + i)
                options_str += f"({letter}) {option} " # format for MathVision
            if len(options_str) > 0:
                judge_prompt += f"Choices: {options_str} \n"
                print(f"judge_prompt with choices: {judge_prompt}")

        judge_prompt += f"Model response: {reasoning_trajectory}\n"
        judge_prompt += 'Extracted answer:'
        return judge_prompt


def eval_single_sample_vlmevalkit(item):
    """Evaluate a single sample."""

    judge_model = JudgeExtractorVLMEvalKit(model_path="gpt-4.1-mini", temperature=0.0)

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
    complex_cases = []

    for i, sample in enumerate(null_result_items):
        if i % 50 == 0:  # Progress logging
            log_info(f"Processing sample {i + 1}/{len(null_result_items)}")

        # pred_answer = sample.get("raw_full_prediction", "").strip()
        # question_type = sample.get("annotation", {}).get("question_type", "")

        # if question_type == "multi_choice":
        #     gt_answer = sample.get("annotation", {}).get("answer_option", "")
        # elif question_type == "free_form":
        #     gt_answer = sample.get("annotation", {}).get("answer", "")
        # else:
        #     raise ValueError(f"Invalid question type: {question_type}")
        gt_answer = sample.get("gt_answer", "").strip()

        choices_raw = sample.get("annotation", {}).get("options", [])
        print(f"choices_raw: {choices_raw}")

        # Parse choices from JSON string format
        if isinstance(choices_raw, str) and not is_nan_value(choices_raw):
            try:
                import ast

                choices = ast.literal_eval(choices_raw)
            except (ValueError, SyntaxError):
                raise ValueError(f"Failed to parse a valid list of options: {choices_raw}")
        elif isinstance(choices_raw, list) and len(choices_raw) > 0:
            choices = choices_raw
        elif any(
            sample.get("annotation", {}).get(key)
            for key in ["A", "B", "C", "D", "E", "F"]
        ):
            choices = []
            for key in ["A", "B", "C", "D", "E", "F"]:
                value = sample.get("annotation", {}).get(key)
                if (
                    value and str(value).strip() and not is_nan_value(value)
                ):  # Check if non-empty and not just whitespace
                    choices.append(value)
        elif is_nan_value(choices_raw):  # Check for NaN
            choices = []
        else:
            raise ValueError(f"choices_raw is not a list or a string and not NaN: {choices_raw}")
        print(f"sample.annotation.keys(): {sample['annotation'].keys()}")
        print(f"parsed choices: {choices}")
        complex_cases.append(sample)
    print("number of complex cases: ", len(complex_cases))
    print("total number of cases processed: ", len(complex_cases))
    judge_samples = []
    for result in complex_cases:
        if result.get("raw_full_prediction"):
            prediction_text = str(result["raw_full_prediction"])
        else:
            raise ValueError(f"No prediction text found: {result}")

        # question_type = result.get("annotation", {}).get("question_type", "")
        # gt_answer = None
        # if question_type == "multi_choice":
        #     gt_answer = result.get("annotation", {}).get("answer_option", "")
        # elif question_type == "free_form":
        #     gt_answer = result.get("annotation", {}).get("answer", "")
        # else:
        #     raise ValueError(f"Invalid question type: {question_type}")
        gt_answer = result.get("gt_answer", "").strip()

        judge_item = result.copy()
        judge_item["prediction_full_text"] = prediction_text.strip()
        judge_item["gt_answer"] = gt_answer
        judge_item["split"] = "null result cases (using judge extraction)"
    
        judge_samples.append(judge_item)

    # Initialize judge_eval_results as empty list
    judge_eval_results = []

    # Only process complex cases if they exist
    if judge_samples:
        # Run threaded evaluation
        with ThreadPoolExecutor(max_workers=200) as executor:
            for result in tqdm(
                executor.map(eval_single_sample_vlmevalkit, judge_samples),
                total=len(judge_samples),
                desc="Judge Evaluation",
            ):
                judge_eval_results.append(result)
    else:
        log_info(
            "No complex cases requiring judge evaluation - all samples handled by simple string matching"
        )
    return judge_eval_results

def calculate_mathvision_judge_evaluation_score_all_sample_gpt_extraction(results_file: str) -> Optional[Dict]:
    """
    Calculate MathVision evaluation scores with LLM Judge fallback mechanism.

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
        initial_total = len(results)

        # Process each sample
        log_info(f"Processing {initial_total} samples for judge evaluation...")
        null_results = [
            result for result in results if result.get("pred_answer") is None
        ]

        null_eval_results = evaluate_null_results(null_results)
        log_info(f"Null judge evaluation results: {len(null_eval_results)}")
        split_accuracy = sum(r["hit"] for r in null_eval_results) / len(null_eval_results)
        log_info(f"Null judge evaluation split accuracy: {split_accuracy:.4f} ({sum(r['hit'] for r in null_eval_results)}/{len(null_eval_results)})")
        
        return {
            "overall_accuracy": split_accuracy,
            "accuracy_by_split": {
                "null_result_cases_using_judge_extraction": split_accuracy
            },
            "eval_results": null_eval_results,
            "num_correct_samples_after_llm_judgement": sum(r['hit'] for r in null_eval_results),
            "num_samples_after_llm_judgement": len(results) # careful with this, make sure check how NULL values are handled and sums to full dataset
        }
    except Exception as e:
        log_info(f"Error in judge evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    """Test the MathVision judge evaluation on a results file."""
    # Note: This judge is built for MiniCPM models because all pred_answer is None, and we use a specific judge to extract the answer with the Answer: format.
    # for MathVision, we should use Tej extractor in general. Otherwise we use MathVista Judge for failed boxed{} format MCQ extraction
    import sys

    if len(sys.argv) != 2:
        print("Usage: python mathvision_helper_functions.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    print(f"Testing MathVision judge evaluation on: {results_file}")
    judge_results = (
        calculate_mathvision_judge_evaluation_score_all_sample_gpt_extraction(
            results_file
        )
    )

    if judge_results:
        print("\nJudge evaluation completed successfully!")
        # for key, value in judge_results.items():
        #     print(f"  {key}: {value}")
    else:
        print("Judge evaluation failed!")
        sys.exit(1)

# #######################
# # Eval Helper Functions
# #######################
# def timestamp() -> str:
#     nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
#     print(nowtime)  
#     return nowtime  

# def is_number(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         return False

# def save_jsonl(path: str, data: list, t_stamp=True) -> None:
#     if t_stamp:
#         file_name = f"{path.replace('.jsonl','')}{timestamp()}.jsonl"
#     else:
#         file_name = path
#     with open(file_name, 'w', encoding='utf-8') as f:
#         for line in tqdm(data, desc='save'):
#             f.write(json.dumps(line, ensure_ascii=False) + '\n')


# def load_jsonl(path: str):
#     with open(path, "r", encoding='utf-8') as fh:
#         return [json.loads(line) for line in fh.readlines() if line]



# def eval_tuple(s):
#     """
#     Evaluates the mathematical expressions within tuples or lists represented as strings.
    
#     Args:
#         s (str): The string representation of a tuple or list.
#                  E.g., "(a,b,c,...)" or "[a,b,c,...]"
    
#     Returns:
#         str: A string representation of the tuple or list with evaluated expressions.
#              Returns the original string if it doesn't match the expected format or if an error occurs.
    
#     Example:
#         eval_tuple("(2*3, 5+2)") -> "(6,7)"
    
#     Note:
#         This function relies on the latex2sympy function which is assumed to be defined elsewhere in the code.
#     """
#     # Split the string by commas to get individual elements
#     sl = s[1:-1].split(',')
    
#     try:
#         # Check if string is a tuple representation and has more than one element
#         if s[0] == '(' and s[-1] == ')' and len(sl) > 1:
#             # Evaluate each element using latex2sympy and round the result to 2 decimal places
#             # Skip evaluation if element is 'infty', 'a', or '-a'
#             s = ','.join([str(round(eval(str(latex2sympy(sub))),2)) 
#                           if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
#             return f"({s})"
        
#         # Check if string is a list representation and has more than one element
#         elif s[0] == '[' and s[-1] == ']' and len(sl) > 1:
#             # Same evaluation process as for tuples
#             s = ','.join([str(round(eval(str(latex2sympy(sub))),2)) 
#                           if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
#             return f"[{s}]"
    
#     except Exception:  # Catch any exceptions and return the original string
#         return s
    
#     # Return original string if it doesn't match tuple or list format
#     return s


# def is_equal(asw: str, gt_asw: str) -> bool:
#     """
#     Judge if `asw` is equivalent to `gt_asw`.

#     This function checks if the given answers are equivalent, considering
#     various scenarios such as tuples, lists separated by commas, and
#     mathematical equivalence in LaTeX format.

#     Args:
#         asw (str): The answer string to be checked.
#         gt_asw (str): The ground truth answer string to be matched against.

#     Returns:
#         bool: True if the answers are equivalent, otherwise False.

#     """

#     # return gt_asw == asw

#     # Check for empty strings after removing spaces and return False if any of them is empty.
#     asw = asw.lower()
#     gt_asw = gt_asw.lower()
    
#     if asw.replace(' ', '') == '' or gt_asw.replace(' ', '') == '':
#         return False

#     if gt_asw.strip() == asw.strip():
#         return True
   
#     # Convert the string to a tuple format.
#     asw = eval_tuple(asw)
#     gt_asw = eval_tuple(gt_asw)

#     # Check for simple tuple containment. Return True if one tuple is contained in the other.
#     if gt_asw == asw:
#         return True

    

#     try:
#         # Convert LaTeX format to a sympy expression and evaluate both expressions.
#         # If the evaluated results are close enough (up to 2 decimal places), return True.
#         if round(eval(str(latex2sympy(gt_asw))), 2) == round(eval(str(latex2sympy(asw))), 2):
#             return True

#         else:
#             return False
#     except:
#         # If any error occurs during comparison, return False.
#         return False


# def in_area(id: str, area: str) -> bool:
#     """Determine if a given ID falls within a specified area.

#     This function checks if a provided ID contains the specified area string
#     or if the ID matches the pattern for a test CSV related to that area.

#     Args:
#         id (str): The ID to be checked.
#         area (str): The area string or 'all'. If 'all', the function always
#                     returns True.

#     Returns:
#         bool: True if the ID is within the specified area or the area is 'all',
#               False otherwise.

#     Examples:
#         >>> in_area("abstract_algebra_test.csv_1", "algebra")
#         True
#         >>> in_area("test/precalculus/244.json", "precalculus")
#         True
#         >>> in_area("abstract_algebra_test.csv_1", "precalculus")
#         False
#     """

#     # If the area is 'all', always return True
#     if area == 'all':
#         return True
    
#     # Check if the ID contains the specified area or if it matches the pattern 
#     # for a test CSV related to that area
#     if f'/{area}/' in id or f'{area}_test.csv' in id:
#         return True

#     # If none of the above conditions are met, return False
#     else:
#         return False


# def extract_nums(s):
#     s = s.replace(",", "")
#     nums = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)
#     return_list = []
#     for i in range(len(nums)):
#         try:
#             return_list.append(eval(nums[i].strip().lstrip(" 0")))
#         except:
#             pass
#     return return_list

# def find_formula(step):
#     assert step.count("<<") == step.count(">>") == 1
#     left, right = step.find("<<")+2, step.find(">>")
#     return step[left: right]


# def extract_answer(completion):
#     ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
#     match = ANS_RE.search(completion)
#     if match:
#         match_str = match.group(1).strip()
#         match_str = match_str.replace(",", "")
#         return match_str
#     else:
#         assert False


# def delete_extra_zero(n):
#     try:
#         n = float(n)  # Try to convert the input to a float
#     except ValueError:  # If conversion fails
#         print("None {}".format(n))  # Print the error message
#         return n  # Return the original string
        
#     # If n is an integer after conversion, return its string representation
#     if isinstance(n, int):
#         return str(n)
    
#     # If n is a float after conversion
#     if isinstance(n, float):
#         n = str(n).rstrip('0')  # Remove trailing zeros after the decimal point
#         # If number ends with a dot after removing zeros, convert to int
#         # Otherwise, keep it as float and return its string representation
#         n = int(n.rstrip('.')) if n.endswith('.') else float(n)
#         return str(n)




# def _fix_fracs(string):
#     # Split the string based on occurrences of '\frac'.
#     substrs = string.split("\\frac")
#     new_str = substrs[0]

#     # Check if there are any occurrences of '\frac' in the string.
#     if len(substrs) > 1:
#         # Exclude the part of the string before the first '\frac'.
#         substrs = substrs[1:]

#         for substr in substrs:
#             new_str += "\\frac"
#             # If the current substring already starts with a brace, 
#             # it's likely formatted correctly.
#             if len(substr) > 0 and substr[0] == "{":
#                 new_str += substr
#             else:
#                 # Ensure that the substring has at least 2 characters 
#                 # for numerator and denominator.
#                 try:
#                     assert len(substr) >= 2
#                 except:
#                     return string

#                 a = substr[0]  # Potential numerator.
#                 b = substr[1]  # Potential denominator.

#                 # Check if the denominator (b) is already braced.
#                 if b != "{":
#                     if len(substr) > 2:
#                         post_substr = substr[2:]
#                         new_str += "{" + a + "}{" + b + "}" + post_substr
#                     else:
#                         new_str += "{" + a + "}{" + b + "}"
#                 else:
#                     if len(substr) > 2:
#                         post_substr = substr[2:]
#                         new_str += "{" + a + "}" + b + post_substr
#                     else:
#                         new_str += "{" + a + "}" + b

#     # Update the string to the newly formatted version.
#     string = new_str
#     return string


# def _fix_a_slash_b(string):
#     # Check if the string contains exactly one slash, which may indicate it's a fraction.
#     if len(string.split("/")) != 2:
#         return string

#     # Split the string by slash to extract potential numerator and denominator.
#     a, b = string.split("/")

#     try:
#         # Try to convert the parts to integers.
#         a = int(a)
#         b = int(b)

#         # Check if the string is in the expected format after conversion.
#         assert string == "{}/{}".format(a, b)

#         # Convert the fraction to LaTeX representation.
#         new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
#         return new_string

#     # Handle exceptions for non-integer fractions or other unexpected formats.
#     except:
#         return string

# def _remove_right_units(string):
#     # Split the string using "\\text{ " as the delimiter.
#     splits = string.split("\\text{ ")
    
#     # Return the part of the string before the last occurrence of "\\text{ ".
#     return splits[0]



# def _fix_sqrt(string):
#     # Check if "\sqrt" is not in the string. If not, return the string as is.
#     if "\\sqrt" not in string:
#         return string

#     # Split the string based on the "\sqrt" substring.
#     splits = string.split("\\sqrt")
    
#     # The initial portion of the string before the first occurrence of "\sqrt".
#     new_string = splits[0]

#     # Loop through each split portion (after the initial one).
#     for split in splits[1:]:
#         # If the split portion is non-empty and the first character isn't a '{',
#         # then it means the argument of the sqrt is not enclosed in braces.
#         if len(split) > 0 and split[0] != "{":
#             a = split[0]
#             # Add braces around the first character and append the rest of the split portion.
#             new_substr = "\\sqrt{" + a + "}" + split[1:]
#         else:
#             # If the split portion starts with a '{', then it's already correct.
#             new_substr = "\\sqrt" + split
#         # Add the new substring to our result string.
#         new_string += new_substr

#     return new_string



# def _strip_string(string):
#     # Remove linebreaks
#     string = string.replace("\n", "")

#     # Remove inverse spaces
#     string = string.replace("\\!", "")

#     # Replace double backslashes with a single backslash
#     string = string.replace("\\\\", "\\")

#     # Replace tfrac and dfrac with frac
#     string = string.replace("tfrac", "frac")
#     string = string.replace("dfrac", "frac")

#     # Remove \left and \right LaTeX commands
#     string = string.replace("\\left", "")
#     string = string.replace("\\right", "")

#     # Remove degree notation
#     string = string.replace("^{\\circ}", "")
#     string = string.replace("^\\circ", "")

#     # Remove dollar signs (potentially used for inline math in LaTeX)
#     string = string.replace("\\$", "")
#     string = string.replace("$", "")

#     # Remove units (assumed to be on the right). Note: The function _remove_right_units is not provided.
#     string = _remove_right_units(string)

#     # Remove percentage notations
#     string = string.replace("\\%", "")
#     string = string.replace("\%", "")

#     # Handle floating numbers starting with "."
#     string = string.replace(" .", " 0.")
#     string = string.replace("{.", "{0.")
#     if len(string) == 0:
#         return string
#     if string[0] == ".":
#         string = "0" + string

#     # If there are equalities or approximations, only consider the value after them
#     if len(string.split("=")) == 2:
#         string = string.split("=")[-1]
#     if len(string.split("\\approx")) == 2:
#         string = string.split("\\approx")[-1]

#     # Fix sqrt values not wrapped in curly braces. Note: The function _fix_sqrt is not provided.
#     if 'sqrt' in string:
#         string = _fix_sqrt(string)

#     # Remove all spaces
#     string = string.replace(" ", "")

#     # Transform certain fraction notations to the desired format. Note: The function _fix_fracs is not provided.
#     if 'sqrt' in string:
#         string = _fix_fracs(string)

#     # Convert 0.5 to its fraction representation
#     if string == "0.5":
#         string = "\\frac{1}{2}"

#     # Fix fractions represented with a slash. Note: The function _fix_a_slash_b is not provided.
#     string = _fix_a_slash_b(string)

#     return string



# def find_math_answer(s: str) -> str:
#     s = s.lower()
#     if '{}' in s:
#         s = s.replace('{}', '')

#     try:
#         pattern = re.compile('oxed{(.*)}', flags=re.S)
#         ans = pattern.findall(s)[-1]
#     except:     
#         ans = s  # If the pattern is not found, consider the entire string as the answer.

#     # If there's a closing bracket without an opening bracket before it, consider everything before it.
#     if ans.find('}') != -1 and (ans.find('{') == -1 or  ans.find('}') < ans.find('{')):
#         ans = ans.split('}')[0]

#     # Extract the value after the equals sign or approx symbol.
#     ans = ans.split('=')[-1]
#     ans = ans.split('\\approx')[-1]

#     # Clean the string from various LaTeX formatting.
#     ans = ans.replace(" ", "").replace("\\,", "").replace('∞', '\\infty')
#     ans = ans.replace("+\infty", "\infty").replace("\\\\", "\\").replace("\n", "")
#     ans = ans.replace('\\text', '').replace('\\mbox', '').replace('bmatrix', 'pmatrix')
#     ans = ans.replace("\\left", "").replace('\\right', '').replace("^{\\circ}", "")
#     ans = ans.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
#     ans = ans.replace("{units}", "").replace("units", "").replace("{km}", "").replace("km", "")

#     return _strip_string(ans) 

# if __name__ == "__main__":
#     # Test cases
#     test_cases = [
#         ("0.5", "\\frac{1}{2}", True),
#         ("2/3", "\\frac{2}{3}", True),
#         # ("(1, 2, 3)", "[1, 2, 3]", True), # to add to is_equal later
#         # ("x^2 + 2x + 1", "(x + 1)^2", True),
#         ("3.14159", "3.14", True),  # Within tolerance
#         ("", "answer", False),
#         ("42", "42.0", True),
#         ("$\\frac{\\sqrt{6}}{6}$", "\\frac{\\sqrt{6}}{6}", True),
#     ]
    
#     for ans, gt, expected in test_cases:
#         result = is_equal(ans, gt)
#         status = "✓" if result == expected else "✗"
#         print(f"{status} is_equal_simplified('{ans}', '{gt}') = {result} (expected: {expected})")