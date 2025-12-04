#!/usr/bin/env python3
"""
Simple script to merge partition results from parallel evaluation runs.
Combines multiple JSON result files and calculates the overall score.
Integrates with Weave for evaluation tracking and comparison.
"""

import json
import glob
import argparse
import os
import sys

# try:
#     import weave
#     from weave import EvaluationLogger

#     WEAVE_AVAILABLE = True
# except ImportError:
#     WEAVE_AVAILABLE = False
#     print("Warning: Weave not available. Install with: pip install weave")

# Add parent directory to path to import evaluation utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tts_eval.reward_guided_search.collate_final_eval_results import calculate_evaluation_score_direct
from tts_eval.reward_guided_search.qwen_collate_final_evaluation import calculate_qwen_judge_evaluation_score
from tts_eval.reward_guided_search.puzzlevqa_judge_evaluation import calculate_puzzlevqa_judge_evaluation_score
from tts_eval.reward_guided_search.mathvista_helper_functions import calculate_mathvista_judge_evaluation_score, calculate_mathvista_judge_evaluation_score_all_sample_gpt_extraction
from tts_eval.reward_guided_search.mathvision_helper_functions import calculate_mathvision_judge_evaluation_score_all_sample_gpt_extraction
# from utils.evaluation import calculate_evaluation_score
from common.send_telegram_notifications_helper import (
    send_telegram_job_summary,
    send_telegram_error_notification,
)


# def log_evaluation_to_weave(
#     score: float,
#     correct: int,
#     total: int,
#     reward_model_path: str,
#     dataset: str,
#     run_datetime: str,
#     weave_project: str,
#     policy_model_path: str = "",
# ) -> None:
#     """
#     Simple Weave logging for evaluation results comparison.

#     Args:
#         score: Evaluation accuracy score
#         correct: Number of correct answers
#         total: Total number of samples
#         reward_model_path: Reward model path (primary identifier)
#         dataset: Dataset name
#         run_datetime: Run datetime string
#         weave_project: Weave project name
#         policy_model_path: Optional policy model path
#     """
#     if not WEAVE_AVAILABLE:
#         print("❌ Weave not available. Install with: pip install weave")
#         return

#     try:
#         # Initialize Weave
#         print(f"Initializing Weave project: {weave_project}")
#         weave.init(project_name=weave_project)

#         # Use reward_model_path as the primary model identifier
#         # Sanitize model name for Weave (only alphanumeric chars and underscores)
#         raw_model_id = reward_model_path or "no_reward_model"
#         model_id = "".join(c if c.isalnum() else "_" for c in raw_model_id)

#         # Initialize EvaluationLogger with just summary data
#         eval_logger = EvaluationLogger(
#             model=model_id,
#             dataset=dataset,
#         )

#         # Log summary metrics only (much simpler)
#         summary_stats = {
#             "accuracy": score,
#             "correct_count": correct,
#             "total_count": total,
#             "reward_model_path": reward_model_path,
#             "policy_model_path": policy_model_path,
#             "dataset": dataset,
#             "run_datetime": run_datetime,
#         }

#         eval_logger.log_summary(summary_stats)

#         print(
#             f"✓ Weave logging complete! {model_id} - {score:.2%} accuracy ({correct}/{total})"
#         )
#         print(
#             f"  View results at: https://wandb.ai/{weave_project.replace('/', '/workspace/')}"
#         )

#     except Exception as e:
#         print(f"❌ Weave logging failed: {e}")
#         print("Continuing without Weave logging...")


def main():
    try:
        _main()
    except Exception as e:
        # Send error notification on merge failure
        try:
            error_message = f"Failed to merge partition results: {str(e)}"
            send_telegram_error_notification(
                model_path_name="Merge Process",
                error_message=error_message,
                error_traceback="",
                evaluation_run_logs_file=os.getenv("EVAL_RUN_LOG_FILE", ""),
                extra_fields={
                    "stage": "merge_partitions",
                },
            )
        except Exception:
            pass
        # Re-raise the original error
        raise


def _main():
    parser = argparse.ArgumentParser(
        description="Merge partition results from parallel evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing partition result files",
    )
    parser.add_argument(
        "--run_datetime",
        type=str,
        required=True,
        help="Run datetime string used in filenames",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (for scoring logic)",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=4,
        help="Expected number of partitions (default: 4)",
    )
    parser.add_argument(
        "--policy_model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Policy model path for Telegram notification",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="",
        help="Reward model path for Telegram notification",
    )
    parser.add_argument(
        "--weave_project",
        type=str,
        default="aisg-arf/mmr-eval",
        help="Weave project name for logging",
    )
    parser.add_argument(
        "--disable_weave",
        action="store_true",
        help="Disable Weave logging",
    )
    args = parser.parse_args()

    # Find all partition files
    pattern = f"{args.output_dir}/result-p*-*-*-{args.run_datetime}.json"
    partition_files = sorted(glob.glob(pattern))

    print(f"Looking for partition files matching: {pattern}")
    print(f"Found {len(partition_files)} partition files")

    if not partition_files:
        print(f"ERROR: No partition files found matching pattern: {pattern}")
        sys.exit(1)

    if len(partition_files) != args.num_partitions:
        print(
            f"WARNING: Expected {args.num_partitions} partitions but found {len(partition_files)}"
        )
        print("Continuing with available partitions...")

    # Display found partition files
    print("\nPartition files to merge:")
    for i, file in enumerate(partition_files):
        file_size = os.path.getsize(file) / 1024  # Size in KB
        print(f"  {i+1}. {os.path.basename(file)} ({file_size:.1f} KB)")

    # Merge all partitions
    merged_results = []
    total_samples_per_partition = []

    for file in partition_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"WARNING: File {file} does not contain a list, skipping")
                    continue
                merged_results.extend(data)
                total_samples_per_partition.append(len(data))
                print(f"  Loaded {len(data)} samples from {os.path.basename(file)}")
        except Exception as e:
            print(f"ERROR loading {file}: {e}")
            continue

    if not merged_results:
        print("ERROR: No data could be loaded from partition files")
        sys.exit(1)

    # Sort results by question order (if they have indices)
    # This ensures the merged file has questions in the original order
    # (This assumes questions are processed in order within each partition)
    # Use reward_model_path as primary identifier, fallback to model_id or policy_model_path
    primary_model_id = args.reward_model_path
    # Write merged file
    merged_file = os.path.join(
        args.output_dir,
        f"result-merged-0-{len(merged_results)}-{args.run_datetime}.json",
    )
    
    with open(merged_file, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    print(f"\nMerged file written: {merged_file}")
    print(f"Total samples in merged file: {len(merged_results)}")

    # Calculate and display final score
    print("\nCalculating overall evaluation score...")
    if "minicpm" in args.policy_model_path.lower():
        print("No direct eval score calculation for MiniCPM")
        score, correct, total = None, None, None
    else:
        score, correct, total = calculate_evaluation_score_direct(merged_file)
    # score, correct, total = calculate_evaluation_score_direct(merged_file)
    
    # Run dataset-specific judge-based evaluation on merged results
    print(f"\nRunning judge-based evaluation for dataset: {args.dataset}...")
    judge_score = None
    judge_score_percent = "N/A"
    num_correct_samples_after_llm_judgement = None
    num_samples_after_llm_judgement = None
    
    try:
        # Determine which judge evaluation to use based on dataset type
        if "mathvista" in args.dataset.lower():
            print("Using MathVista-specific LLM judge evaluation...")
            if "minicpm" in args.policy_model_path.lower():
                judge_results = calculate_mathvista_judge_evaluation_score_all_sample_gpt_extraction(merged_file)
            else:
                judge_results = calculate_mathvista_judge_evaluation_score(merged_file)
            judge_type = "MathVista"
        elif "puzzle" in args.dataset.lower():
            print("Using PuzzleVQA-specific LLM judge evaluation...")
            if "minicpm" in args.policy_model_path.lower():
                judge_results = calculate_mathvision_judge_evaluation_score_all_sample_gpt_extraction(merged_file)
            else:
                judge_results = calculate_puzzlevqa_judge_evaluation_score(merged_file)
            judge_type = "PuzzleVQA"
        elif "mmmu" in args.dataset.lower():
            print("Using MMMU-specific LLM judge evaluation...")
            if "minicpm" in args.policy_model_path.lower():
                judge_results = calculate_mathvision_judge_evaluation_score_all_sample_gpt_extraction(merged_file)
            else:
                judge_results = calculate_qwen_judge_evaluation_score(merged_file)
            judge_type = "MMMU"
        elif "mathvision" in args.dataset.lower():
            if "minicpm" in args.policy_model_path.lower():
                print("Using MathVision-specific LLM judge evaluation...")
                judge_results = calculate_mathvision_judge_evaluation_score_all_sample_gpt_extraction(merged_file)
            else:
                # judge_results = calculate_mathvision_judge_evaluation_score(merged_file)
                print("MathVision judge evaluation not implemented for non-MiniCPM models, use Tej extractor, just like for Q7B")
            judge_type = "MathVision"
        
        if judge_results is not None:
            judge_score = judge_results["overall_accuracy"]
            judge_score_percent = f"{judge_score:.2%}"
            print(f"{judge_type} judge evaluation completed: {judge_score_percent}")
            num_correct_samples_after_llm_judgement = judge_results[
                "num_correct_samples_after_llm_judgement"
            ]
            num_samples_after_llm_judgement = judge_results[
                "num_samples_after_llm_judgement"
            ]
            
            # Report additional PuzzleVQA-specific stats if available
            if judge_type == "PuzzleVQA" and "judge_improved_count" in judge_results:
                print(
                    f"PuzzleVQA judge statistics:\n"
                    f"  Initial correct: {judge_results.get('initial_correct_count', 'N/A')}\n"
                    f"  Judge processed: {judge_results.get('judge_processed_count', 'N/A')} failed extractions\n"
                    f"  Judge improved: {judge_results.get('judge_improved_count', 'N/A')} samples\n"
                    f"  Final correct: {num_correct_samples_after_llm_judgement}/{num_samples_after_llm_judgement}"
                )
            else:
                print(
                    f"Number of correct samples after LLM answer extraction breakdown: \n Number of correct samples: {num_correct_samples_after_llm_judgement}\n Number of samples after LLM answer extraction: {num_samples_after_llm_judgement}"
                )
        else:
            print(f"{judge_type} judge evaluation failed - no results returned")
    except Exception as e:
        print(f"Judge evaluation error: {e}")
        print("Continuing with basic evaluation only...")

    # Log to Weave (after calculating score)
    # if not args.disable_weave:
    #     if WEAVE_AVAILABLE:
    #         print("\n" + "=" * 60)
    #         print("WEAVE LOGGING")
    #         print("=" * 60)
    #         log_evaluation_to_weave(
    #             score=score if score is not None else 0.0,
    #             correct=correct if correct is not None else 0,
    #             total=total if total is not None else 0,
    #             reward_model_path=args.reward_model_path,
    #             dataset=args.dataset,
    #             run_datetime=args.run_datetime,
    #             weave_project=args.weave_project,
    #             policy_model_path=args.policy_model_path,
    #         )
    #     else:
    #         print("Weave logging skipped - Weave not available")
    # else:
    #     print("Weave logging disabled")

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL MERGED EVALUATION RESULTS")
    print("=" * 60)
    print(f"Reward Model: {args.reward_model_path}")
    print(f"Policy Model: {args.policy_model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Run DateTime: {args.run_datetime}")
    print(f"Partitions merged: {len(partition_files)}")
    print(f"Samples per partition: {total_samples_per_partition}")
    print(f"Total samples: {total}")
    
    if score is not None:
        print(f"Correct answers: {correct}")
        print(f"Basic Accuracy: {score:.4f} ({score*100:.2f}%)")
    else:
        print("Basic Score calculation failed")
    
    print(f"Judge Accuracy: {judge_score_percent}")
    if judge_score is not None:
        print(f"Judge Score: {judge_score:.4f}")
    
    print(f"Merged results file: {merged_file}")
    if not args.disable_weave:
        print(f"Weave project: {args.weave_project}")
    print("=" * 60)

    # Write a summary file for easy reference
    summary_file = os.path.join(
        args.output_dir,
        f"summary-merged-{args.run_datetime}.txt",
    )
    
    with open(summary_file, "w") as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Reward Model: {args.reward_model_path}\n")
        f.write(f"Policy Model: {args.policy_model_path}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Run DateTime: {args.run_datetime}\n")
        f.write(f"Partitions Merged: {len(partition_files)}\n")
        f.write(f"Total Samples: {total}\n")
        if score is not None:
            f.write(f"Correct: {correct}\n")
            f.write(f"Basic Accuracy: {score:.4f} ({score*100:.2f}%)\n")
        else:
            f.write("Basic Score: N/A (calculation failed)\n")
        
        f.write(f"Judge Accuracy: {judge_score_percent}\n")
        if judge_score is not None:
            f.write(f"Judge Score: {judge_score:.4f}\n")
        if num_correct_samples_after_llm_judgement is not None:
            f.write(
                f"Judge Correct Samples: {num_correct_samples_after_llm_judgement}\n"
            )
        if num_samples_after_llm_judgement is not None:
            f.write(f"Judge Total Samples: {num_samples_after_llm_judgement}\n")
        f.write(f"Merged File: {merged_file}\n")
        if not args.disable_weave:
            f.write(f"Weave Project: {args.weave_project}\n")
        f.write("\nPartition Files:\n")
        for file in partition_files:
            f.write(f"  - {os.path.basename(file)}\n")
    
    print(f"\nSummary written to: {summary_file}")
    
    # Send Telegram notification for merged results
    try:
        eval_score_percent = f"{score*100:.2f}%" if score is not None else "N/A"
        judge_breakdown = (
            f" [{num_correct_samples_after_llm_judgement}/{num_samples_after_llm_judgement}]"
            if num_correct_samples_after_llm_judgement is not None
            and num_samples_after_llm_judgement is not None
            else ""
        )
        # Create dataset-appropriate summary message
        if args.dataset.startswith("puzzle") or args.dataset == "puzzleVQA_1K_subset":
            extraction_description = "LLM Answer Extraction tries to extract answer from reasoning text if regex patterns fail."
        else:
            extraction_description = "LLM Answer Extraction tries to extract answer from the text if not given a boxed answer.\nAnd currently assigns a random answer to null answer samples"
        
        eval_summary = f"""{args.dataset}: 
        Basic: {correct}/{total} = {eval_score_percent} 
        With LLM Answer Extraction: {judge_breakdown} = {judge_score_percent}
        
        Basic Score includes null answer samples which are considered as incorrect; 
        {extraction_description}"""

        print(eval_summary)
        
        # send_telegram_job_summary(
        #     model_path_name=args.policy_model_path,
        #     evaluation_results_json_file=merged_file,
        #     evaluation_run_logs_file=os.getenv("EVAL_RUN_LOG_FILE", ""),
        #     extra_fields={
        #         # "model_id": primary_model_id,
        #         "policy_model_path": args.policy_model_path,
        #         "reward_model_path": args.reward_model_path,
        #         "dataset": args.dataset,
        #         "total_samples": total,
        #         "correct_answers": correct,
        #         "basic_accuracy": eval_score_percent,
        #         "judge_accuracy": judge_score_percent,
        #         "judge_correct_samples": num_correct_samples_after_llm_judgement
        #         if num_correct_samples_after_llm_judgement is not None
        #         else "N/A",
        #         "judge_total_samples": num_samples_after_llm_judgement
        #         if num_samples_after_llm_judgement is not None
        #         else "N/A",
        #         "partitions_merged": len(partition_files),
        #         "run_datetime": args.run_datetime,
        #         # "weave_project": args.weave_project
        #         # if not args.disable_weave
        #         # else "disabled",
        #     },
        #     separator="\t",
        #     include_header=True,
        #     send_files=True,
        #     message_prefix=f"✅[Eval Success]\n{eval_summary}",
        # )
        print("\nTelegram notification sent for merged results")
    except Exception as e:
        print(f"\nTelegram notification error: {e}")
        # Don't fail the merge if Telegram notification fails
        pass


if __name__ == "__main__":
    main()