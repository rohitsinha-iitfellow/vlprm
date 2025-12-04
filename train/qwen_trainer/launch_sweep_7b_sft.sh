#!/bin/bash

# Full sweep launch script for multiple model/dataset combinations
# Run the test script first (launch_test_7b_sft.sh) before using this
# Run this script from the qwen_training directory: bash scripts/launch_sweep_7b_sft.sh

# Arrays of models to sweep
models=(
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

# Arrays of datasets to sweep (from qwenvl/data/__init__.py)
datasets=(
    "VL-PRM300K-V1-train"
)

# Vision tuning options to sweep
tune_mm_vision_options=(
    # "False"  # Default: Vision tower frozen
    "True"   # Vision tower trainable
)

# Image size options to sweep
image_sizes=(
    "smart-resize"  # Use image_qwen_smart_resize key
    # "qwen-size"     # Use image key
)

# Base job name
base_job_name="train-VL-PRM"

# Counter for job submissions
job_count=0

echo "Starting PBS job sweep at $(date)"
echo "Models: ${models[@]}"
echo "Datasets: ${datasets[@]}"
echo "Vision tuning options: ${tune_mm_vision_options[@]}"
echo "Image sizes: ${image_sizes[@]}"
echo "Total combinations: $((${#models[@]} * ${#datasets[@]} * ${#tune_mm_vision_options[@]} * ${#image_sizes[@]}))"
echo ""

# Create logs directory if it doesn't exist
mkdir -p scripts/pbs_queue_logs

# Loop through all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for tune_mm_vision in "${tune_mm_vision_options[@]}"; do
            for image_size in "${image_sizes[@]}"; do
                # Set image key short form
                case "${image_size}" in
                    smart-resize)
                        image_size_short="sr"
                        ;;
                    qwen-size)
                        image_size_short="qs"
                        ;;
                esac
                
                # Create short names for job naming
                model_short=$(basename "${model}" | sed 's/Qwen2.5-VL-7B-Instruct/Q7B/g' | sed 's/updated-tokens/UT/g')

                # Build an informative dataset_short, e.g., "no-perception-v1-p100", "full-dataset-v1-p100"
                ds_work="${dataset}"
                # Strip known prefixes if present
                ds_work="${ds_work#visualprm_data_}"
                # Separate sampling percent (default 100 if absent)
                if [[ "${ds_work}" == *%* ]]; then
                    ds_pct="${ds_work##*%}"
                    ds_core="${ds_work%%%*}"
                else
                    ds_pct="100"
                    ds_core="${ds_work}"
                fi
                # Split name and version (version is the suffix after last '_', e.g., v1)
                ds_ver="${ds_core##*_}"
                ds_name_raw="${ds_core%_*}"
                # Keep underscores for consistency with pattern matching
                ds_name="${ds_name_raw}"
                # Friendly aliases
                case "${ds_name}" in
                    full_non_balanced)
                        ds_name="full_dataset"
                        ;;
                esac
                dataset_short="${ds_name}_${ds_ver}_p${ds_pct}"
                
                # Add vision tuning suffix
                if [ "${tune_mm_vision}" = "True" ]; then
                    vision_suffix="vit_trained"
                else
                    vision_suffix="vit_frozen"
                fi
                
                # Create unique job name
                datetime=$(date +"%Y%m%d_%H%M%S")
                job_name="${base_job_name}_${image_size_short}_${model_short}_${dataset_short}_${vision_suffix}_${datetime}"
                output_file="scripts/pbs_queue_logs/${job_name}.out"
                
                echo "Submitting job $((job_count+1)): ${job_name}"
                echo "  Image size: ${image_size}"
                echo "  Model: ${model}"
                echo "  Dataset: ${dataset}"
                echo "  Vision tuning: ${tune_mm_vision}"
                
                # Submit job with environment variables (submitting from parent directory)
                # Add -r y to enable automatic requeue on exit code 99 (CUDA failure)
                cd ..
                job_id=$(qsub -r y -v MODEL_PATH="${model}",DATASET_NAME="${dataset}",TUNE_MM_VISION="${tune_mm_vision}",IMAGE_SIZE="${image_size}",PBS_OUTPUT_FILE="${output_file}" \
                        -N "${job_name}" -o "qwen_training/${output_file}" qwen_training/scripts/sft_7b_parameterized.pbs)
                cd qwen_training
                
                echo "  Job ID: ${job_id}"
                echo "  Note: Job will automatically requeue if CUDA is unavailable (exit code 99)"
                echo ""
                
                ((job_count++))
                
                # Add delay between submissions to avoid overwhelming the scheduler
                sleep 2
            done
        done
    done
done

echo "Submitted ${job_count} jobs total"
echo "Use 'qstat -u $USER' to check job status"
echo ""
echo "Log files are in: qwen_training/scripts/pbs_queue_logs/"
echo "Output models will be in: ./outputs/"