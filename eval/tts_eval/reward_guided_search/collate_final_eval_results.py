import json
from math_verify import parse, verify
import argparse


def verify_math_ans(gold, answer):
    try:
        gold = parse("$" + gold + "$")
        answer = parse("$" + answer + "$")
    except Exception as e:
        print("Error Parsing: ", e)
        print(gold, answer)
        return False
    try:
        output = verify(gold, answer)
        return output
    except BaseException as e:
        print("Error Comparing: ", e)
        print(gold, answer)
        return False


def calculate_evaluation_score_direct(file_path):
    """
    Calculate evaluation score using direct JSON loading (avoids PyArrow issues).
    
    Args:
        file_path: Path to the JSON results file
        
    Returns:
        tuple: (score, correct_count, total_count) or (None, None, None) on error
    """
    print(f"Using result file: {file_path}")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON with {len(data)} records")
        
        if len(data) > 0:
            print("Sample record keys:", list(data[0].keys()))
            print("First record gt_answer:", data[0].get("gt_answer"))
            print("First record pred_answer:", data[0].get("pred_answer"))
            
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file {file_path}: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None, None, None

    data_filtered = [item for item in data if item.get("pred_answer") is not None]
    null_predictions = len(data) - len(data_filtered)

    print(f"Total samples: {len(data)}, Filtered samples: {len(data_filtered)}")
    if null_predictions > 0:
        print(f"Warning: {null_predictions} samples have null pred_answer and were excluded")

    if len(data) == 0:
        return 0.0, 0, 0

    if "mmmu" in file_path.lower() or "mathvista" in file_path.lower():
        correct_count = sum(
            [
                verify_math_ans(
                    str(data_filtered[idx]["gt_answer"]),
                    data_filtered[idx]["pred_answer"]
                )
                if str(data_filtered[idx]["gt_answer"]).isnumeric()
                else data_filtered[idx]["gt_answer"].lower()
                == data_filtered[idx]["pred_answer"].lower()
                for idx in range(len(data_filtered))
            ]
        )
    else:
        correct_count = sum(
            [
                data_filtered[idx]["gt_answer"].lower() == data_filtered[idx]["pred_answer"].lower()
                for idx in range(len(data_filtered))
            ]
        )

    full_score = correct_count / len(data)
    filtered_score = correct_count / len(data_filtered)
    
    print(f"With {len(data_filtered)} data filtered (non-null answer) samples: {filtered_score}")
    # print(f"With {len(data)} all samples: {full_score}")
    print(f"Correct samples: {correct_count}")
    
    return full_score, correct_count, len(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BON Greedy Search results evaluation collate")
    parser.add_argument("file_path", type=str, help="Path to the JSON results file")
    args = parser.parse_args()

    file_path = args.file_path
    
    # Use the extracted function for standalone execution
    score, correct_count, total_count = calculate_evaluation_score_direct(file_path)
    
    if score is not None:
        print(f"Final results on full dataset (not filtered for null answers): Score={score:.4f}, Correct={correct_count}/{total_count}")
    else:
        print("Evaluation failed due to errors above.")