"""
PuzzleVQA Helper Functions for PRM-guided Evaluation Pipeline

This module contains all PuzzleVQA-specific functionality including:
- Dataset loading from JSON format
- Prompt building with exact PuzzleVQA ChainOfThought format
- Answer extraction using PuzzleVQA regex patterns
- Image processing and scoring functions

Designed to maintain exact compatibility with the reference PuzzleVQA implementation
while integrating seamlessly with the PRM-guided search pipeline.
"""

import os
import json
import re
import base64
from io import BytesIO
from typing import List, Dict, Tuple
import pandas as pd
from PIL import Image
# from .logger import log_info
# from logger import log_info
try: # TODO: tech debt for now, revisit later, since this file is used both as module and script
    from utils.logger import log_info
except ImportError:
    # Handle case when run as script
    from logger import log_info


# All 20 PuzzleVQA dataset types
PUZZLE_DATASET_TYPES = [
    "triangle", "color_grid", "color_hexagon", "color_number_hexagon",
    "color_overlap_squares", "color_size_circle", "grid_number_color", 
    "grid_number", "polygon_sides_color", "polygon_sides_number",
    "rectangle_height_color", "rectangle_height_number", "shape_morph",
    "shape_reflect", "shape_size_grid", "shape_size_hexagon", 
    "size_cycle", "size_grid", "circle_size_number", "venn"
]

ALGOPUZZLE_DATASET_TYPES = [
    "board_tile",
    "calendar",
    "chain_link",
    "checker_move",
    "clock",
    "colour_hue",
    "map",
    "maze",
    "move_box",
    "n_queens",
    "number_slide",
    "rotting_kiwi",
    "rubiks_cube",
    "think_dot",
    "tower_of_hanoi",
    "water_jugs",
    "wheel_of_fortune",
    "wood_slide",
]


def load_puzzle_dataset(puzzle_data_dir: str, puzzle_type: str, puzzle_test_dataset_type: str) -> pd.DataFrame:
    """
    Load PuzzleVQA dataset from JSON format and convert to compatible DataFrame.

    Args:
        puzzle_data_dir: Path to PuzzleVQA or AlgoPuzzleVQA data directory
        puzzle_type: Name of puzzle type (e.g., "triangle", "color_grid")

    Returns:
        pandas DataFrame with columns: question, options, answer, image (base64), index

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If puzzle_type not in supported types
    """
    if (
        puzzle_type not in PUZZLE_DATASET_TYPES
        and puzzle_type not in ALGOPUZZLE_DATASET_TYPES
    ):
        raise ValueError(
            f"Unsupported puzzle type: {puzzle_type}. Must be one of: {PUZZLE_DATASET_TYPES + ALGOPUZZLE_DATASET_TYPES}"
        )

    json_path = os.path.join(puzzle_data_dir, f"{puzzle_type}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Puzzle dataset file not found: {json_path}")

    log_info(f"Loading Puzzle dataset: {json_path}")
    
    # Load JSON data
    samples = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                sample = json.loads(line.strip())
                
                # Convert local image path to base64
                image_path = os.path.join(puzzle_data_dir, sample["image"])
                if not os.path.exists(image_path):
                    log_info(f"Warning: Image file not found: {image_path}")
                    continue
                    
                # Load and convert image to base64
                with Image.open(image_path) as img:
                    img_rgb = img.convert("RGB")
                    buffer = BytesIO()
                    img_rgb.save(buffer, format="JPEG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                # Convert answer to corresponding option letter (A, B, C, D)
                try:
                    # Ensure type consistency by converting both to strings for comparison
                    answer_str = str(sample["answer"])
                    options_str = [str(opt) for opt in sample["options"]]

                    answer_index = options_str.index(answer_str)
                    answer_letter = ["A", "B", "C", "D"][answer_index]
                except (ValueError, IndexError) as e:
                    log_info(
                        f"Warning: Answer '{sample['answer']}' not found in options {sample['options']} at line {line_idx}: {e}"
                    )
                    continue

                # Create sample record
                sample_record = {
                    "puzzle_type": puzzle_type,
                    "question": sample["question"],
                    "options": sample[
                        "options"
                    ],  # List of 4 options or 3 for size puzzles
                    "answer": answer_letter,  # The correct option letter (A, B, C, D)
                    "image": img_base64,  # Base64 encoded image
                    "index": line_idx,  # Sequential index
                    # Optional fields from PuzzleVQA (for potential future use)
                    "caption": sample.get("caption", ""),
                    "explanation": sample.get("explanation", ""),
                    "deduction": sample.get("deduction", ""),
                }
                samples.append(sample_record)

            except (json.JSONDecodeError, KeyError, IOError) as e:
                log_info(f"Error processing line {line_idx} in {json_path}: {e}")
                continue

    if not samples:
        raise ValueError(f"No valid samples loaded from {json_path}")

    df = pd.DataFrame(samples)
    log_info(
        f"Loaded {len(df)} samples from Puzzle dataset: {puzzle_type} from {puzzle_test_dataset_type}"
    )
    return df


def build_puzzlevqa_prompt(question: str, options: List[str]) -> str:
    """
    Build prompt using exact PuzzleVQA ChainThoughtMultiChoicePrompter format.

    This replicates the exact prompt structure from PuzzleVQA's reference implementation:
    - "Do not directly give the final answer" instruction
    - Options formatted as (A), (B), (C), (D)
    - "Let's describe the image first and think step by step" ending

    Args:
        question: The puzzle question text
        options: List of answer options (3 or 4 options)

    Returns:
        Formatted prompt string matching PuzzleVQA format
    """
    # Handle special case for 3-option puzzles (size puzzles)
    size_options = {"small", "medium", "large"}
    is_size_puzzle = len(options) == 3 and set(options) == size_options

    # Build prompt parts following exact PuzzleVQA format
    parts = [
        f"Question: {question.rstrip()} Options:",
        f"(A) {options[0]}",
        f"(B) {options[1]}",
        f"(C) {options[2]}",
    ]

    # Add (D) option if not a size puzzle
    if not is_size_puzzle and len(options) >= 4:
        parts.append(f"(D) {options[3]}")

    # parts.extend([
    #     "",
    #     "Answer: Let's describe the image first and think step by step."
    # ])
    parts.extend(
        [
            "Please select the correct answer from the options above.",
            "If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering. If you do need to think step by step, first, carefully observe and describe everything you see: the image itself and how it connects to the problem being presented. Then, work through your reasoning step by step to arrive at your answer. **Your teacher will review your descriptions of visual elements to ensure you're observing all relevant details accurately, and will critique each step of your reasoning to provide guidance and ensure you're on the right track.** Put your final answer within \\boxed{}. If multiple-choice options for answers to the question are provided, select the alphabetical letter of the corresponding best option within \\boxed{} when you are ready to provide your final answer.",
        ]
    )
    
    return "\n".join(parts)

def build_puzzlevqa_prompt_minicpm(question: str, options: List[str]) -> str:
    """
    Build prompt using exact PuzzleVQA ChainThoughtMultiChoicePrompter format.

    This replicates the exact prompt structure from PuzzleVQA's reference implementation:
    - "Do not directly give the final answer" instruction
    - Options formatted as (A), (B), (C), (D)
    - "Let's describe the image first and think step by step" ending

    Args:
        question: The puzzle question text
        options: List of answer options (3 or 4 options)

    Returns:
        Formatted prompt string matching PuzzleVQA format
    """
    # Handle special case for 3-option puzzles (size puzzles)
    size_options = {"small", "medium", "large"}
    is_size_puzzle = len(options) == 3 and set(options) == size_options

    # Build prompt parts following exact PuzzleVQA format
    parts = [
        f"Question: {question.rstrip()} Options:",
        f"(A) {options[0]}",
        f"(B) {options[1]}",
        f"(C) {options[2]}",
    ]

    # Add (D) option if not a size puzzle
    if not is_size_puzzle and len(options) >= 4:
        parts.append(f"(D) {options[3]}")

    # parts.extend([
    #     "",
    #     "Answer: Let's describe the image first and think step by step."
    # ])
    parts.extend(
        [
            "Please select the correct answer from the options above.",
            'Please describe what you see step by step, then solve the problem step by step and then output the final answer in the format of "Answer: single number or single word or phrase"\n\nWhen working out your reasoning / intermediate steps, 请一步步推理, especially when doing so helps with clarity in your reasoning in order to arrive at the correct final answer.',
        ]
    )
    
    return "\n".join(parts)


def build_algopuzzlevqa_prompt(question: str, options: List[str]) -> str:
    """
    Build prompt using exact PuzzleVQA ChainThoughtMultiChoicePrompter format.

    This replicates the exact prompt structure from PuzzleVQA's reference implementation:
    - "Do not directly give the final answer" instruction
    - Options formatted as (A), (B), (C), (D)
    - "Let's describe the image first and think step by step" ending

    Args:
        question: The puzzle question text
        options: List of answer options (2 or 4 options)

    Returns:
        Formatted prompt string matching PuzzleVQA format
    """
    size_options = {"small", "medium", "large"}
    binary_options = {"Yes", "No"}
    assert (
        len(options) == 4
        or set(options) == size_options
        or set(options) == binary_options
    )

    parts: list[str] = [
        f"Question: {question.rstrip()} Options:",
        f"(A) {options[0]}",
        f"(B) {options[1]}",
        f"(C) {options[2]}"
        if len(options) >= 3
        else "",  # TODO: will result in empty newlines
        f"(D) {options[3]}" if len(options) == 4 else "",
    ]

    if set(options) == size_options:
        parts.pop(-3)

    parts.extend(
        [
            "Please select the correct answer from the options above.",
            "If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering. If you do need to think step by step, first, carefully observe and describe everything you see: the image itself and how it connects to the problem being presented. Then, work through your reasoning step by step to arrive at your answer. **Your teacher will review your descriptions of visual elements to ensure you're observing all relevant details accurately, and will critique each step of your reasoning to provide guidance and ensure you're on the right track.** Put your final answer within \\boxed{}. If multiple-choice options for answers to the question are provided, select the alphabetical letter of the corresponding best option within \\boxed{} when you are ready to provide your final answer.",
        ]
    )

    return "\n".join(part for part in parts if part)

def build_algopuzzlevqa_prompt_minicpm(question: str, options: List[str]) -> str:
    """
    Build prompt using exact PuzzleVQA ChainThoughtMultiChoicePrompter format.

    This replicates the exact prompt structure from PuzzleVQA's reference implementation:
    - "Do not directly give the final answer" instruction
    - Options formatted as (A), (B), (C), (D)
    - "Let's describe the image first and think step by step" ending

    Args:
        question: The puzzle question text
        options: List of answer options (2 or 4 options)

    Returns:
        Formatted prompt string matching PuzzleVQA format
    """
    size_options = {"small", "medium", "large"}
    binary_options = {"Yes", "No"}
    assert (
        len(options) == 4
        or set(options) == size_options
        or set(options) == binary_options
    )

    parts: list[str] = [
        f"Question: {question.rstrip()} Options:",
        f"(A) {options[0]}",
        f"(B) {options[1]}",
        f"(C) {options[2]}"
        if len(options) >= 3
        else "",  # TODO: will result in empty newlines
        f"(D) {options[3]}" if len(options) == 4 else "",
    ]

    if set(options) == size_options:
        parts.pop(-3)

    parts.extend(
        [
            "Please select the correct answer from the options above.",
            'Please describe what you see step by step, then solve the problem step by step and then output the final answer in the format of "Answer: single number or single word or phrase"\n\nWhen working out your reasoning / intermediate steps, 请一步步推理, especially when doing so helps with clarity in your reasoning in order to arrive at the correct final answer.',
        ]
    )

    return "\n".join(part for part in parts if part)


def extract_puzzlevqa_answer(text: str, options: List[str]) -> str:
    """
    Extract answer using exact PuzzleVQA regex patterns and matching logic.
    
    This replicates PuzzleVQA's ChainThoughtMultiExtractPrompter.get_answer() method:
    1. Try to find (A), (B), (C), (D) patterns first
    2. Fallback to A, B, C, D patterns
    3. Take the LAST match found (critical detail)
    4. Map back to the actual option string
    
    Args:
        text: Generated text to extract answer from
        options: List of answer options to map back to
        
    Returns:
        Extracted answer string, or error message if extraction fails
    """
    # Create mapping from letters to options
    mapping = {letter: option for letter, option in zip("ABCD", options)}
    
    # Primary pattern: look for (A), (B), (C), (D) 
    matches = re.findall(r"\(([ABCD])\)", text)
    if matches:
        # Take LAST match (critical detail from PuzzleVQA)
        last_match = matches[-1]
        return mapping.get(last_match, options[0])
    
    # Fallback pattern: look for standalone A, B, C, D
    matches = re.findall(r"[ABCD]", text)
    if matches:
        # Take LAST match
        last_match = matches[-1]  
        return mapping.get(last_match, options[0])
    
    # Extraction failed - return error message like PuzzleVQA
    return f"Cannot get_answer: {text}"


# def prepare_puzzlevqa_question_array(question: str, image_base64: str) -> Tuple[List[Dict], List[str]]:
#     """
#     Prepare PuzzleVQA question in message array format for the reward model.
    
#     PuzzleVQA questions don't have interleaved image tokens, so we use the
#     non-interleave format with image first, then text.
    
#     Args:
#         question: The formatted question prompt
#         image_base64: Base64 encoded image string
        
#     Returns:
#         Tuple of (messages_array, image_data_list)
#     """
#     # PuzzleVQA uses non-interleaved format
#     user_messages_array = [{
#         "role": "user",
#         "content": [{"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"}] +
#                   [{"type": "text", "text": question}]
#     }]
    
#     return user_messages_array, [image_base64]


# def calculate_puzzlevqa_score(results_file: str) -> Tuple[float, int, int]:
#     """
#     Calculate evaluation score using PuzzleVQA's exact matching approach.
    
#     PuzzleVQA uses simple exact string matching:
#     - 1.0 if predicted answer exactly matches ground truth
#     - 0.0 otherwise
    
#     Args:
#         results_file: Path to JSON results file
        
#     Returns:
#         Tuple of (accuracy_score, correct_count, total_count)
#     """
#     try:
#         with open(results_file, 'r', encoding='utf-8') as f:
#             results = json.load(f)
        
#         if not results:
#             log_info("Warning: Empty results file")
#             return None, 0, 0
        
#         correct_count = 0
#         total_count = len(results)
        
#         for result in results:
#             pred_answer = result.get("pred_answer", "")
#             gt_answer = result.get("gt_answer", "")
            
#             # Handle the case where extraction failed
#             if pred_answer and not pred_answer.startswith("Cannot get_answer:"):
#                 if pred_answer == gt_answer:
#                     correct_count += 1
        
#         accuracy = correct_count / total_count if total_count > 0 else 0.0
        
#         log_info(f"PuzzleVQA evaluation: {correct_count}/{total_count} = {accuracy:.2%}")
#         return accuracy, correct_count, total_count
        
#     except Exception as e:
#         log_info(f"Error calculating PuzzleVQA score: {e}")
#         return None, 0, 0


# def calculate_puzzlevqa_1k_subset_detailed_score(results_file: str) -> Tuple[float, int, int, Dict]:
#     """
#     Calculate detailed evaluation scores for PuzzleVQA 1K subset with per-puzzle-type breakdown.
    
#     Args:
#         results_file: Path to JSON results file
        
#     Returns:
#         Tuple of (overall_accuracy, total_correct, total_count, per_puzzle_scores)
#         where per_puzzle_scores is a dict mapping puzzle_type -> (accuracy, correct, total)
#     """
#     try:
#         with open(results_file, 'r', encoding='utf-8') as f:
#             results = json.load(f)
        
#         if not results:
#             log_info("Warning: Empty results file")
#             return None, 0, 0, {}
        
#         # Overall counters
#         overall_correct = 0
#         overall_total = len(results)
        
#         # Per-puzzle-type counters
#         puzzle_type_stats = {}
        
#         for result in results:
#             pred_answer = result.get("pred_answer", "")
#             gt_answer = result.get("gt_answer", "")
#             puzzle_type = result.get("puzzle_type", "unknown")
            
#             # Initialize puzzle type stats if not seen before
#             if puzzle_type not in puzzle_type_stats:
#                 puzzle_type_stats[puzzle_type] = {"correct": 0, "total": 0}
            
#             # Update puzzle type totals
#             puzzle_type_stats[puzzle_type]["total"] += 1
            
#             # Check if answer is correct
#             is_correct = False
#             if pred_answer and not pred_answer.startswith("Cannot get_answer:"):
#                 if pred_answer == gt_answer:
#                     is_correct = True
#                     overall_correct += 1
#                     puzzle_type_stats[puzzle_type]["correct"] += 1
        
#         # Calculate overall accuracy
#         overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        
#         # Calculate per-puzzle-type accuracies
#         per_puzzle_scores = {}
#         for puzzle_type, stats in puzzle_type_stats.items():
#             accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
#             per_puzzle_scores[puzzle_type] = (accuracy, stats["correct"], stats["total"])
        
#         # Log detailed results
#         log_info(f"PuzzleVQA 1K subset overall: {overall_correct}/{overall_total} = {overall_accuracy:.2%}")
#         log_info("Per-puzzle-type breakdown:")
#         for puzzle_type in sorted(per_puzzle_scores.keys()):
#             acc, correct, total = per_puzzle_scores[puzzle_type]
#             log_info(f"  {puzzle_type}: {correct}/{total} = {acc:.2%}")
        
#         return overall_accuracy, overall_correct, overall_total, per_puzzle_scores
        
#     except Exception as e:
#         log_info(f"Error calculating PuzzleVQA 1K subset detailed score: {e}")
#         return None, 0, 0, {}


def load_puzzle_subset(
    puzzle_data_dir: str, puzzle_test_dataset_type: str
) -> pd.DataFrame:
    """
    Load PuzzleVQA 1K subset: 50 samples from each of the 20 puzzle types.
    
    This creates a comprehensive evaluation dataset with 1000 total samples
    (50 samples × 20 puzzle types) for unified evaluation across all puzzle types.
    
    Args:
        puzzlevqa_data_dir: Path to PuzzleVQA data directory
        
    Returns:
        pandas DataFrame with 1000 samples from all puzzle types combined
        Includes additional 'puzzle_type' column for analysis
        
    Raises:
        FileNotFoundError: If any puzzle dataset files are missing
    """
    log_info("Loading PuzzleVQA 1K subset: 50 samples from each of 20 puzzle types")
    
    all_samples = []
    global_index = 0

    for puzzle_type in (
        PUZZLE_DATASET_TYPES
        if puzzle_test_dataset_type == "puzzleVQA_1K_subset"
        else ALGOPUZZLE_DATASET_TYPES
    ):
        log_info(f"Loading {puzzle_type} dataset (taking first 50/100 samples)")
        
        try:
            # Load full puzzle dataset
            puzzle_df = load_puzzle_dataset(
                puzzle_data_dir, puzzle_type, puzzle_test_dataset_type
            )
            
            # Take first 50 samples for consistency and reproducibility
            sampled_df = puzzle_df.head(50).copy()
            
            # Add puzzle type information for analysis
            sampled_df['puzzle_type'] = puzzle_type
            
            # Update global indices to ensure uniqueness across all puzzles
            sampled_df['global_index'] = range(global_index, global_index + len(sampled_df))
            global_index += len(sampled_df)
            
            all_samples.append(sampled_df)
            
        except Exception as e:
            log_info(f"Error loading {puzzle_type}: {e}")
            raise FileNotFoundError(f"Failed to load puzzle type {puzzle_type}: {e}")
    
    # Combine all puzzle datasets
    combined_df = pd.concat(all_samples, ignore_index=True)
    
    log_info(f"Successfully loaded PuzzleVQA 1K subset: {len(combined_df)} samples")
    log_info(f"Puzzle type distribution: {combined_df['puzzle_type'].value_counts().to_dict()}")
    
    return combined_df


def get_puzzle_dataset_info(puzzle_test_dataset_type: str) -> Dict:
    """
    Get dataset-specific configuration for PuzzleVQA puzzle types.

    Args:
        puzzle_test_dataset_type: Name of puzzle test dataset type or "puzzleVQA_1K_subset" or "AlgoPuzzleVQA_900_subset"

    Returns:
        Dictionary with dataset configuration
    """
    if puzzle_test_dataset_type == "puzzleVQA_1K_subset":
        return {
            "name": "puzzleVQA_1K_subset",
            "total_samples": 1000,  # 50 samples × 20 puzzle types
            "has_multiple_choice": True,
            "answer_key": "answer",
            "interleave_image_tokens": False,
            "system_prompt": "You are a helpful assistant.",
            "scoring_method": "exact_match"
        }
    elif puzzle_test_dataset_type == "AlgoPuzzleVQA_900_subset":
        return {
            "name": "AlgoPuzzleVQA_900_subset",
            "total_samples": 900,  # Individual puzzle datasets have 100 samples
            "has_multiple_choice": True,
            "answer_key": "answer",
            "interleave_image_tokens": False,  # PuzzleVQA doesn't use interleaved tokens
            "system_prompt": "You are a helpful assistant.",
            "scoring_method": "exact_match",
        }
    else:
        raise ValueError(
            f"Unsupported puzzle test dataset type: {puzzle_test_dataset_type}. Must be one of: ['puzzleVQA_1K_subset', 'AlgoPuzzleVQA_900_subset']"
        )


# def validate_puzzle_type(puzzle_type: str) -> str:
#     """
#     Validate and normalize puzzle type string.
    
#     Args:
#         puzzle_type: Raw puzzle type string (may include "puzzle_" prefix or be "puzzleVQA_1K_subset")
        
#     Returns:
#         Normalized puzzle type string
        
#     Raises:
#         ValueError: If puzzle type is not supported
#     """
#     # Handle special case for 1K subset
#     if puzzle_type == "puzzleVQA_1K_subset":
#         return puzzle_type
    
#     # Remove "puzzle_" prefix if present
#     if puzzle_type.startswith("puzzle_"):
#         puzzle_type = puzzle_type[7:]  # Remove "puzzle_" prefix
    
#     if puzzle_type not in PUZZLE_DATASET_TYPES:
#         valid_options = PUZZLE_DATASET_TYPES + ["puzzleVQA_1K_subset"]
#         raise ValueError(f"Unsupported puzzle type: {puzzle_type}. Must be one of: {valid_options}")
    
#     return puzzle_type