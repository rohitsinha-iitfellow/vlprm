import os
import numpy as np
import json
import base64
import math
from tqdm import tqdm

# from openai import OpenAI
from PIL import Image
from io import BytesIO

import argparse
import re
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# VLLM imports
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# from prompts import POLICY_USER_PROMPT, MCQ_SUFFIX_PROMPT_NO_TEMPLATE
POLICY_USER_PROMPT = r"""Here is the question you need to answer: {{QUESTION}}"""
MCQ_SUFFIX_PROMPT_NO_TEMPLATE = """

Multiple-choice options:
{{OPTIONS}}
"""

def safe_isnan(val):
    try:
        return math.isnan(val)
    except (TypeError, ValueError):
        return False

def get_token_prob(logprobs_dict, target_token):
    """
    logprobs_dict: dict[token_id -> Logprob]
    target_token: string (decoded token to look for)
    """
    for tok_id, logprob_obj in logprobs_dict.items():
        if logprob_obj.decoded_token == target_token:
            return float(np.exp(logprob_obj.logprob))
    return 0  # token not found in top-k

PRM_SYSTEM_PROMPT_NORMAL_TOK = """**You are a process supervision model for visual reasoning tasks. You will receive an image and an image-based problem statement, followed by solution steps to evaluate.**

First round: problem statement and first solution step.  
Subsequent rounds: one new step per round.

Assess the cumulative correctness of the entire solution up to each step.

## Evaluation Criteria:

1. **Visual Accuracy**: Are visual elements from the image correctly identified (shapes, colors, positions, quantities, spatial relationships)?

2. **Logical Validity**: Do all inferences and calculations follow correctly from the image and previous steps?

## Response:
- **"+"** if correct up to this step
- **"-"** if any error exists up to this step

Only respond with "+" or "-". No explanations.

An error in any step invalidates all subsequent steps."""

def load_json(file_path):
    """
    Load a JSON file and return its contents as a Python object (dict or list).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Save to bytes buffer
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.read()).decode("utf-8")

def save_json(data, filepath, indent=2):
    """
    Save Python object as JSON file, creating parent directories if needed.

    Args:
        data: Python object (dict, list, etc.)
        filepath: path to save the JSON file
        indent: indentation level for pretty printing (default=2)
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

class VisualPRM:
    def __init__(self, args):
        self.llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=True,  # Often needed for vision models
            enable_prefix_caching=True,  # Enable prefix caching for better performance
            seed = 42,
            # use_tqdm=False
            # enable_progress_bar=False #=True
        )

        self.pos_token = "+"
        self.neg_token = "-"
        self.wanted_tokens = ["+", "-"]

        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            stop=None,
            guided_decoding=GuidedDecodingParams(choice = self.wanted_tokens),
            logprobs=10
        )
        

    def get_judge_reward(self, image_strs, question, solution):
        image_contents = []
        
        if isinstance(image_strs, list):
            for image_str in image_strs:
                image_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_str}"},
                    }
                )
        elif isinstance(image_strs, str):
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_strs}"},
                }
            )
        else:
            raise TypeError("Image Strings are not of type list or string")
        
        messages = []

        messages.append({"role": "system", "content": PRM_SYSTEM_PROMPT_NORMAL_TOK})

        content = image_contents + [
            {
                "type": "text",
                "text": "### Question:\n" + question + "\n\n### Solution Process:\n" + solution + "\n\n",
            }
        ]
        messages.append({"role": "user", "content": content})
        
        # Use chat interface with proper message formatting
        chat_response = self.llm.chat(
            messages,
            sampling_params=self.sampling_params,
        )
            
        # Extract judgment from chat response
        response = chat_response[0].outputs[0].text.strip().lower()
        logprobs  = chat_response[0].outputs[0].logprobs[0]
            
        if not (response.startswith("+") or response.startswith("-")):
            raise ValueError(f"Invalid Generation: {response}")
            
        positive_prob = get_token_prob(logprobs, "+")

        return positive_prob

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Vision Language Model on accuracy with LLM As Judge based Reward"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the vision language model",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for VLLM",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for model weights",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        # default="outputs/vision-prm",
        required = True,
        help="Output path for results",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required = True,
        help="Path to step traces",
    )
    parser.add_argument(
        "--num-workers", type=int, default=32, help="Number of parallel workers"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a Math Teacher evaluating student solutions. Given a question with visual elements and a student's step-by-step solution, evaluate the mathematical correctness and logic consistency of each step. Respond with '+' if the step is correct or '-' if it is incorrect.",
        help="System prompt for the model",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results during processing",
    )

    args = parser.parse_args()

    data = load_json(args.data_path)

    model = VisualPRM(args)

    file_name = os.path.basename(args.data_path)

    for sample in tqdm(data):
        question = sample['annotation']['question']
        user_prompt = POLICY_USER_PROMPT.replace("{{QUESTION}}", question)
        image_str = sample['annotation']['image']

        if "puzzleVQA" in file_name or "AlgoPuzzleVQA" in file_name:
            options = sample['annotation']['options']
            options_str = "\n".join(
                        [f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
                    )
            user_prompt += MCQ_SUFFIX_PROMPT_NO_TEMPLATE.replace(
                        "{{OPTIONS}}", options_str
                    )
        elif "MMMU" in file_name:
            # raise NotImplementedError("MMMU Data processing TODO")
            if sample['annotation']['question_type'] in [
                "multi-choice",
                "multiple-choice",
            ]:
                # options = []
                option_chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
                option_strs = []
                for option_char in option_chars:
                    option_val = sample['annotation'][option_char]
                    if option_val != None and not safe_isnan(option_val):
                        option_strs.append(f"{option_char}: {option_val}")
                
                options_str = "\n".join(option_strs)
                user_prompt += MCQ_SUFFIX_PROMPT_NO_TEMPLATE.replace(
                        "{{OPTIONS}}", options_str
                    )
                
        elif "mathvista" in file_name:
            if sample['annotation']['question_type'] in [
                "multi-choice",
                "multiple-choice",
            ]:
                options = sample['annotation']['choices']
                options_str = "\n".join([f"{option}" for option in options]) + "\n"
                user_prompt += MCQ_SUFFIX_PROMPT_NO_TEMPLATE.replace(
                            "{{OPTIONS}}", options_str
                        )
        else:
            raise ValueError(
                f"Check MCQ formatting for dataset before running: {args.data_path}"
            )

        max_score = -1
        best_idx = -1
        if len(sample['iteration_history']) < 1:
            continue
        
        for cidx, candidate in enumerate(sample['iteration_history'][0]['candidates_info']):
            solution = candidate['candidate_step'].replace("<|im_end|>", "").replace("<|end_of_turn|>", "")
            reward = model.get_judge_reward(image_str, user_prompt, solution)

            if reward > max_score:
                max_score = reward
                best_idx = cidx
            
            candidate["judge_reward"] = reward

        sample['judge_reward'] = {
            "chosen_candidate": best_idx,
            "aggregate_reward": max_score
        }
        # break

    save_json(data, args.output_path)



if __name__ == "__main__":
    main()
        