#!/bin/bash

# Launcher script to submit multiple PBS jobs for BoN evaluation
# Each checkpoint gets its own PBS job submitted to the queue

# CHECKPOINT_BASE_PATH="/data/projects/71001002/ob1/prm-training-code/training_outputs"

POLICY_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct" # Options: "google/gemma-3-12/27b-it", "Qwen/Qwen2.5-VL-3/7/32/72B-Instruct"
# POLICY_MODEL_PATH="google/gemma-3-27b-it"

# Dataset configuration
DATASETS=(
    "AlgoPuzzleVQA_900_subset"  # AlgoPuzzleVQA 900 subset: 50 samples from each of 18 puzzle types (900 total)
    "puzzleVQA_1K_subset"  # PuzzleVQA 1K subset: 50 samples from each of 20 puzzle types (1000 total)
    "mathvista_testmini"
    "MMMU_DEV_VAL" # Original MMMU dataset from TSV file
    "mathvision_test"
)

# Development mode flag (set to "true" to enable development mode)
DEVELOPMENT_MODE="true"  # options: "true" "false"

# PRM Checkpoints
CHECKPOINT_NAMES=(
    "/home/users/nus/ob1/scratch/.cache/huggingface/hub/models--ob11--Qwen-VL-PRM-3B/snapshots/99d3938012db52cb6a29506b9b553eb3af491a7e"
    # "Q3B_mc0_sr_mc0_balanced_bs2_gs4_lr1e-5_VF_0827_1452"
    # "Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452"
    # "Q3B_sr_mc0_ablation_bs2_gs4_lr1e-5_VF_0828_211639"
    # "Q7B_mc0_sr_mc0_balanced_bs2_gs4_lr1e-5_VF_0826_2309"
    # "Q7B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0826_2309"
    # "Q7B_mc0_sr_mc0_ablation_bs2_gs4_lr1e-5_VF_0826_2309"
)

# Parallel execution flag (set to "true" to use 8 GPUs in parallel)
USE_PARALLEL="true"  # Options: "true" "false"

# Extract model prefix from policy model path for job naming
if [[ $POLICY_MODEL_PATH =~ [Qq]wen.*32B ]]; then
    model_prefix="Q32B"
elif [[ $POLICY_MODEL_PATH =~ [Qq]wen.*7B ]]; then
    model_prefix="Q7B"
elif [[ $POLICY_MODEL_PATH =~ [Qq]wen.*3B ]]; then
    model_prefix="Q3B"
elif [[ $POLICY_MODEL_PATH =~ [Gg]emma.*27b ]]; then
    model_prefix="G27B"
elif [[ $POLICY_MODEL_PATH =~ [Gg]emma.*12b ]]; then
    model_prefix="G12B"
elif [[ $POLICY_MODEL_PATH =~ [Mm]iniCPM-V-2_6 ]]; then
    model_prefix="M26"
else
    model_prefix="UNKNOWN"  # fallback
fi

base_job_name_prefix="NG16_${model_prefix}"

# Check if parallel mode is requested via environment variable
if [ -n "$PARALLEL_MODE" ]; then
    USE_PARALLEL="$PARALLEL_MODE"
fi

# Counter for job submissions
job_count=0

echo "Starting PBS job sweep for BoN evaluation at $(date)"
echo "Policy model: ${POLICY_MODEL_PATH} (${model_prefix})"
echo "Number of checkpoints: ${#CHECKPOINT_NAMES[@]}"
echo "Datasets: ${DATASETS[*]}"
echo "Development mode: ${DEVELOPMENT_MODE}"
echo "Parallel mode: ${USE_PARALLEL}"
echo "Total jobs to submit: $((${#CHECKPOINT_NAMES[@]} * ${#DATASETS[@]}))"
if [ "$USE_PARALLEL" = "true" ]; then
    echo "Each job will use 8 GPUs (4 parallel partitions with 2 GPUs each)"
else
    echo "Each job will use 2 GPUs (sequential processing)"
fi
echo ""

# Loop through all checkpoint names and datasets to create job per combination
for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
    # Construct full checkpoint path
    # checkpoint="${CHECKPOINT_BASE_PATH}/${checkpoint_name}"
    checkpoint="/home/users/nus/ob1/scratch/.cache/huggingface/hub/models--ob11--Qwen-VL-PRM-3B/snapshots/99d3938012db52cb6a29506b9b553eb3af491a7e"
    
    # Create short name for job by extracting key segments
    # Extract important components: custom_tok, ablation_v3, Q7b_UT, vit_frozen/trained
    checkpoint_short=""
    if [[ $checkpoint_name == *"custom_tok"* ]]; then
        checkpoint_short+="custom_tok_"
    elif [[ $checkpoint_name == *"normal_tok"* ]]; then
        checkpoint_short+="normal_tok_"
    fi
    
    if [[ $checkpoint_name == *"ablation"* ]]; then
        checkpoint_short+="ablation_"
        if [[ $checkpoint_name == *"v3"* ]]; then
            checkpoint_short+="v3_"
        elif [[ $checkpoint_name == *"v1"* ]]; then
            checkpoint_short+="v1_"
        fi
    fi
    
    if [[ $checkpoint_name == *"7B-Instruct"* ]]; then
        checkpoint_short+="Q7b_"
        if [[ $checkpoint_name == *"updated-tokens"* ]]; then
            checkpoint_short+="UT_"
        fi
    elif [[ $checkpoint_name == *"32B-Instruct"* ]]; then
        checkpoint_short+="Q32b_"
    fi
    
    if [[ $checkpoint_name == *"vit_frozen"* ]]; then
        checkpoint_short+="vit_frozen"
    elif [[ $checkpoint_name == *"vit_trained"* ]]; then
        checkpoint_short+="vit_trained"
    fi
    
    # Loop through each dataset to create separate jobs
    for dataset in "${DATASETS[@]}"; do
        # Create dynamic dataset prefix for this specific dataset
        dataset_prefix="${dataset:0:4}"
        dynamic_job_name="${base_job_name_prefix}_${dataset_prefix}"
        
        # Create unique job name with dataset
        datetime=$(date +"%Y%m%d-%H%M%S")
        job_name="${dynamic_job_name}-${checkpoint_short}-${dataset}-${datetime}"
        
        # Create single log file path for both PBS output and Python logger
        # Use absolute path for PBS and relative path will be handled by the script
        log_file="reward_guided_search/logs/${job_name}.log"
        abs_log_file="${PWD}/${log_file}"
        
        # Ensure log directory exists
        mkdir -p reward_guided_search/logs
        
        echo "Submitting job $((job_count+1)): ${job_name}"
        echo "  Checkpoint: ${checkpoint}"
        echo "  Dataset: ${dataset}"
        echo "  Development mode: ${DEVELOPMENT_MODE}"
        echo "  Parallel mode: ${USE_PARALLEL}"
        echo "  Log file: ${log_file}"
        
        # Submit job with checkpoint path, policy model path, single dataset, development mode, and log file as environment variables
        # PBS will write stdout to this file, and Python logger will append to it
        # Add -r y to enable automatic requeue on exit code 99 (CUDA failure)
        if [ "$USE_PARALLEL" = "true" ]; then
            # Use parallel PBS script for 8-GPU execution
            job_id=$(qsub -r y -v CHECKPOINT_PATH="${checkpoint}",POLICY_MODEL_PATH="${POLICY_MODEL_PATH}",POLICY_MODEL_NAME="policy_${model_prefix}",EVAL_RUN_LOG_FILE="${abs_log_file}",DATASET="${dataset}",DEVELOPMENT_MODE="${DEVELOPMENT_MODE}" \
                    -N "${job_name}" -o "${abs_log_file}" vllm_run_bon_evaluation_parallel.pbs)
        else
            # Use standard PBS script for 2-GPU execution
            job_id=$(qsub -r y -v CHECKPOINT_PATH="${checkpoint}",POLICY_MODEL_PATH="${POLICY_MODEL_PATH}",POLICY_MODEL_NAME="policy_${model_prefix}",EVAL_RUN_LOG_FILE="${abs_log_file}",DATASET="${dataset}",DEVELOPMENT_MODE="${DEVELOPMENT_MODE}" \
                    -N "${job_name}" -o "${abs_log_file}" vllm_run_bon_evaluation.pbs)
        fi
        
        echo "  Job ID: ${job_id}"
        echo "  Note: Job will automatically requeue if CUDA is unavailable (exit code 99)"
        echo ""
        
        ((job_count++))
        
        # Add delay between submissions to avoid overwhelming the scheduler
        sleep 2
    done
done

echo "Submitted ${job_count} jobs total"
echo "Use 'qstat -u $USER' to check job status"
echo ""
echo "Log files are in: reward_guided_search/logs/"
echo "Results will be in: reward_guided_search/outputs/policy_${model_prefix}/"
if [ "$USE_PARALLEL" = "true" ]; then
    echo "Note: Each parallel job will create 4 partition files and 1 merged result file"
fi