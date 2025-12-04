try:
    from .logger import log_info
except ImportError:
    # Handle case when run as script
    from utils.logger import log_info
from datasets import load_dataset
import pandas as pd
import json
import numpy as np
# import re
# from Levenshtein import distance
# from utils import extract_boxed
import os
import sys
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math
import dotenv
dotenv.load_dotenv()

if "OPENAI_API_KEY" in os.environ:
    print(os.environ["OPENAI_API_KEY"][:5])
else:
    print("OPENAI_API_KEY not found")
    exit()

def load_mathvista_dataset(dataset_name='AI4Math/MathVista', test_split_name='testmini'):
    """Load MathVista dataset from HuggingFace and convert to DataFrame format compatible with existing pipeline."""
    log_info(f"Loading MathVista dataset {dataset_name}, split {test_split_name}...")
    
    # Load from HuggingFace (following MathVista generate_response.py:108-113)
    data_list = load_dataset(dataset_name, split=test_split_name)
    
    # Convert to DataFrame format compatible with existing pipeline
    data_records = []
    for item in data_list:
        # check if question_type is multi_choice
        if item["question_type"] == "multi_choice":
            # if so, convert gt_answer to corresponding option letter (A, B, C, D)
            answer_str = str(item["answer"]).strip()
            options_str = [str(opt).strip() for opt in item["choices"]]

            answer_index = options_str.index(answer_str)
            answer_letter = [chr(ord("A") + i) for i in range(len(options_str))][
                answer_index
            ]
        # Convert HF dataset item to record format
        record = {
            "index": item["pid"],  # Use pid as index
            "id": item["pid"],  # Also store as id for consistency
            "question": item["question"],
            "answer": answer_letter
            if item["question_type"] == "multi_choice"
            else item["answer"],  # Ground truth answer
            "question_type": item["question_type"],
            "answer_type": item["answer_type"],
            "choices": item.get("choices", []),  # May be empty for non-MC questions
            "unit": item.get("unit", ""),
            "precision": item.get("precision", None),
            "grade": item["metadata"].get("grade", ""),
            "subject": item["metadata"].get("subject", ""),
            "category": item["metadata"].get("category", ""),
            "skills": item["metadata"].get("skills", []),
            "source": item["metadata"].get("source", ""),
            "context": item["metadata"].get("context", ""),
            "task": item["metadata"].get("task", ""),
            "language": item["metadata"].get("language", ""),
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

def build_mathvista_prompt(line_dict):
    """Build MathVista prompt following the original MathVista query building logic.
    
    Ports logic from MathVista build_query.py:create_one_query() function.
    """
    question = line_dict['question']
    unit = line_dict.get('unit', '')
    choices = line_dict.get('choices', [])
    question_type = line_dict['question_type']
    answer_type = line_dict['answer_type'] 
    precision = line_dict.get('precision', None)
    
    # Build question text
    question_text = f"Question: {question}"
    if unit:
        question_text += f" (Unit: {unit})"
    
    # Build choices text (for multiple choice questions)
    choices_text = ""
    if choices:
        texts = ["Choices:"]
        for i, choice in enumerate(choices):
            texts.append(f"({chr(ord('A')+i)}) {choice}")
        choices_text = "\n".join(texts)
    
    # Build hint text based on question/answer type (following MathVista logic)
    if question_type == "multi_choice":
        assert answer_type == "text"
        hint_text = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
    else:
        if answer_type == "integer":
            hint_text = "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."
        elif answer_type == "float" and precision == 1:
            hint_text = "Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."
        elif answer_type == "float" and precision == 2:
            hint_text = "Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."
        elif answer_type == "list":
            hint_text = "Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
        else:
            hint_text = "Hint: Please answer the question and provide the final answer at the end."
    
    # Build reasoning instructions
    reasoning_text = ("If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. "
                     "Avoid repeating steps indefinitely—provide your best guess even if unsure. "
                     "Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering. "
                     "If you do need to think step by step, first, carefully observe and describe everything you see: the image itself and how it connects to the problem being presented. "
                     "Then, work through your reasoning step by step to arrive at your answer. "
                     "**Your teacher will review your descriptions of visual elements to ensure you're observing all relevant details accurately, and will critique each step of your reasoning to provide guidance and ensure you're on the right track.** "
                     "Put your final answer within \\boxed{}. "
                     "If multiple-choice options for answers to the question are provided, select the alphabetical letter of the corresponding best option within \\boxed{} when you are ready to provide your final answer.")
    
    # Combine all elements
    elements = [question_text, choices_text, hint_text, reasoning_text]
    user_prompt = "\n".join([e for e in elements if e]).strip()
    
    return user_prompt

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
            options = annotation.get("choices", [])
            # if not options:
            #     return "Cannot extract: No choices found", False
            
            # Get the final reasoning step (last step in trajectory)
            if not final_steps:
                return "Cannot extract: No reasoning steps found", False
            
            # reasoning_trajectory = final_steps[-1] if isinstance(final_steps, list) else str(final_steps)
            
            # Reconstruct original prompt (approximation of MathVista format)
            judge_prompt = self._construct_judge_prompt(
                original_question, options, reasoning_trajectory
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

            # if judge_output in valid_responses:
            #     extracted_answer = judge_output
            #     extraction_method = True
            # else:
            #     extracted_answer = judge_output
            #     extraction_method = False
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

def toliststr(s):
    """Convert string representation of list to actual list"""
    if isinstance(s, str):
        try:
            import ast
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [s]
    return s if isinstance(s, list) else [s]

def build_minicpm_mathvista_cot_prompt(line_dict): # no CoT for MathVista for MiniCPM
    # short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
    #                                  '''then output the final answer in the format of "Answer: single number '''
    #                                  '''or single word or phrase".\n\n''')
    short_ans_cot_prompt = ('''Read the following question carefully and analyze all problem-relevant visual elements through a step-by-step perceptual process. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering. If you do need to think step by step, describe what you see step by step, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')

    short_ans_cot_prompt += "When working out the reasoning / intermediate steps, 请一步步推理, especially when that helps clarity.\n\n"
    # short_ans_cot_prompt += "When reasoning through intermediate steps, always consider whether thinking in Mandarin Chinese (普通话) would improve clarity or accuracy. If it would, explicitly carry out those reasoning steps in Mandarin.\n\n"
    
    prompt = short_ans_cot_prompt + line_dict["question"] 
    # return line_dict["question"]
    return prompt

def load_mathvista_dataset_from_file(data_dir, dataset_name='MathVista_MINI'):
    """Load the MathVista dataset from TSV file."""
    data_path = os.path.join(data_dir, f"{dataset_name}.tsv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    data = pd.read_csv(data_path, sep="\t")
    
    data['index'] = [str(x) for x in data['index']]
    
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


def eval_single_sample_vlmevalkit(item):
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
                executor.map(eval_single_sample_vlmevalkit, judge_samples),
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

def calculate_mathvista_judge_evaluation_score_all_sample_gpt_extraction(results_file: str) -> Optional[Dict]:
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
    judge_results = calculate_mathvista_judge_evaluation_score_all_sample_gpt_extraction(results_file)
    
    if judge_results:
        print("\nJudge evaluation completed successfully!")
        # for key, value in judge_results.items():
        #     print(f"  {key}: {value}")
    else:
        print("Judge evaluation failed!")
        sys.exit(1)