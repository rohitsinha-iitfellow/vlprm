#!/bin/bash

# Arrays of parameter values to sweep
micro_batch_sizes=(2)
gradient_accumulation_steps=(4)
learning_rates=(1e-5 2e-5)
max_grad_norms=(1)
tune_vision_values=(true)
datasets=(
"ob11/VL-PRM300K-V1-train"
)


# Base job name
base_job_name="train-VL-PRM"

# Counter for job submissions
job_count=0

echo "Starting PBS job sweep at $(date)"
echo "Micro batch sizes: ${micro_batch_sizes[@]}"
echo "Gradient accumulation steps: ${gradient_accumulation_steps[@]}"
echo "Learning rates: ${learning_rates[@]}"
echo "Max grad norms: ${max_grad_norms[@]}"
echo "Tune vision values: ${tune_vision_values[@]}"
echo "Datasets: ${datasets[@]}"

# Create logs directory if it doesn't exist
mkdir -p pbs_queue_logs

# Usage: ./launch_sft_qwen_sweep_clean.sh
# Loop through all combinations
for dataset in "${datasets[@]}"; do
    for mbs in "${micro_batch_sizes[@]}"; do
        for gas in "${gradient_accumulation_steps[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for max_grad_norm in "${max_grad_norms[@]}"; do
                    for tune_vision in "${tune_vision_values[@]}"; do
                    datetime=$(date +"%m%d_%H%M%S")
                    vision_suffix=$(if [ "$tune_vision" = "true" ]; then echo "VT"; else echo "VF"; fi)
                    dataset_suffix=$(echo "$dataset" | rev | cut -d'/' -f1 | rev | sed -e 's/VisualPRM300Kv0-balanced.*/mc0_balanced/g' -e 's/VisualPRM300Kv0-ablation.*/mc0_ablation/g' -e 's/VisualPRM300Kv0-full-dataset.*/mc0_full/g')
                    job_name="${base_job_name}_sr_${dataset_suffix}_bs${mbs}_gs${gas}_lr${lr}_${vision_suffix}_${datetime}"
                    output_file="pbs_queue_logs/${job_name}.out"
                    
                    echo "Submitting job ${job_count}: ${job_name} (dataset=${dataset}, mbs=${mbs}, gas=${gas}, lr=${lr}, max_grad_norm=${max_grad_norm}, tune_vision=${tune_vision})"
                    
                    # Submit job with command line options and environment variables
                    job_id=$(qsub -v MICRO_BATCH_SIZE="${mbs}",GRADIENT_ACCUMULATION_STEPS="${gas}",LEARNING_RATE="${lr}",MAX_GRAD_NORM="${max_grad_norm}",TUNE_VISION="${tune_vision}",DATASET_NAME="${dataset}",PBS_OUTPUT_FILE="${output_file}" \
                            -N "${job_name}" -o "${output_file}" train/sft_qwen_parameterized.pbs)
                    
                    echo "  Job ID: ${job_id}"
                    
                    ((job_count++))
                    
                    # Optional: Add delay between submissions
                    sleep 1
                    done
                done
            done
        done
    done
done

echo "Submitted ${job_count} jobs total"
echo "Use 'qstat -u $USER' to check job status" 