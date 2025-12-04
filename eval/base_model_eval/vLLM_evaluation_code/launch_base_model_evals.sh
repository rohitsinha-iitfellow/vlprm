#!/bin/bash

# Launcher script to submit multiple PBS jobs for base model evaluation
# Each model-dataset combination gets its own PBS job submitted to the queue

# Model configuration - 3 models to evaluate
MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct"
    # "google/gemma-3-27b-it"
    # "google/gemma-3-12b-it"
    # "OpenGVLab/InternVL2_5-8B"
    # "openbmb/MiniCPM-V-2_6"
)

# Dataset configuration - 4 datasets to evaluate on
DATASETS=(
    "mathvista_testmini"    
    # "mathvision_test"
)

# Development mode flag (set to "true" to enable development mode with fewer samples)
DEVELOPMENT_MODE="false"  # options: "true" "false"

# Counter for job submissions
job_count=0

echo "Starting PBS job sweep for base model evaluation at $(date)"
echo "Number of models: ${#MODELS[@]}"
echo "Number of datasets: ${#DATASETS[@]}"
echo "Development mode: ${DEVELOPMENT_MODE}"
echo "Total jobs to submit: $((${#MODELS[@]} * ${#DATASETS[@]}))"
echo ""

# Loop through all models and datasets to create job per combination
for model_path in "${MODELS[@]}"; do
    # Extract model prefix from model path for job naming
    if [[ $model_path =~ [Qq]wen.*32B ]]; then
        model_prefix="Q32B"
    elif [[ $model_path =~ [Qq]wen.*7B ]]; then
        model_prefix="Q7B"
    elif [[ $model_path =~ [Qq]wen.*3B ]]; then
        model_prefix="Q3B"
    elif [[ $model_path =~ [Gg]emma.*27b ]]; then
        model_prefix="G27B"
    elif [[ $model_path =~ [Gg]emma.*12b ]]; then
        model_prefix="G12B"
    elif [[ $model_path =~ [Ii]nternVL2_5-8B ]]; then
        model_prefix="I8B"
    elif [[ $model_path =~ [Mm]iniCPM-V-2_6 ]]; then
        model_prefix="M26"
    else
        model_prefix="UNKNOWN"  # fallback
    fi
    
    # Create base job name with model prefix
    base_job_name_prefix="BASE_${model_prefix}"
    
    # Loop through each dataset to create separate jobs
    for dataset in "${DATASETS[@]}"; do
        # Create dynamic dataset prefix for this specific dataset
        dataset_prefix="${dataset:0:4}"
        
        # Create unique job name with model and dataset
        datetime=$(date +"%Y%m%d-%H%M%S")
        job_name="${base_job_name_prefix}_${dataset_prefix}-${dataset}-${datetime}"
        
        # Create log file path
        log_dir="logs"
        log_file="${log_dir}/${job_name}.log"
        abs_log_file="${PWD}/${log_file}"
        
        # Ensure log directory exists
        mkdir -p "$log_dir"
        
        echo "Submitting job $((job_count+1)): ${job_name}"
        echo "  Model: ${model_path} (${model_prefix})"
        echo "  Dataset: ${dataset}"
        echo "  Development mode: ${DEVELOPMENT_MODE}"
        echo "  Log file: ${log_file}"
        
        # Submit job with model path, dataset, development mode, and log file as environment variables
        # Add -r y to enable automatic requeue on exit code 99 (CUDA failure)
        job_id=$(qsub -r y \
            -v POLICY_MODEL_PATH="${model_path}",DATASET="${dataset}",DEVELOPMENT_MODE="${DEVELOPMENT_MODE}",EVAL_RUN_LOG_FILE="${abs_log_file}" \
            -N "${job_name}" \
            -o "${abs_log_file}" \
            run_base_model_eval.pbs)
        
        echo "  Job ID: ${job_id}"
        echo "  Note: Job will automatically requeue if CUDA is unavailable (exit code 99)"
        echo ""
        
        ((job_count++))
        
        # Add delay between submissions to avoid overwhelming the scheduler
        sleep 2
    done
done

echo "==========================================="
echo "Submitted ${job_count} jobs total"
echo "Use 'qstat -u $USER' to check job status"
echo ""
echo "Log files are in: ${log_dir}/"
echo "Results will be in: outputs/{model_prefix}/{dataset}/"
echo "==========================================="