# not yet tested and verified

# #!/bin/bash

# # 12-Node Launcher script for mathvision_test dataset
# # Splits the dataset across 12 nodes, each running 4 parallel partitions

# CHECKPOINT_BASE_PATH="/scratch_aisg/SPEC-SF-AISG/ob1/prm-training-code/training_outputs"

# # Policy model configuration
# POLICY_MODEL_PATH="google/gemma-3-27b-it"

# # Dataset configuration - specifically for mathvision_test
# DATASET="mathvision_test"
# TOTAL_SAMPLES=3040
# TOTAL_NODES=12

# # Development mode flag
# DEVELOPMENT_MODE="false"  # options: "true" "false"

# # PRM Checkpoints (using the ones from current vllm_launch_bon_evaluation.sh)
# CHECKPOINT_NAMES=(
#     # "Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452"
#     "Q7B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0826_2309"
# )

# # Extract model prefix from policy model path for job naming
# if [[ $POLICY_MODEL_PATH =~ [Qq]wen.*32B ]]; then
#     model_prefix="Q32B"
# elif [[ $POLICY_MODEL_PATH =~ [Qq]wen.*7B ]]; then
#     model_prefix="Q7B"
# elif [[ $POLICY_MODEL_PATH =~ [Gg]emma.*27b ]]; then
#     model_prefix="G27B"
# elif [[ $POLICY_MODEL_PATH =~ [Gg]emma.*12b ]]; then
#     model_prefix="G12B"
# elif [[ $POLICY_MODEL_PATH =~ [Mm]iniCPM-V-2_6 ]]; then
#     model_prefix="M26"
# else
#     model_prefix="UNKNOWN"
# fi

# base_job_name_prefix="NG16_${model_prefix}_12N"

# # Counter for job submissions
# job_count=0

# echo "=========================================="
# echo "Starting 12-Node PBS job sweep for mathvision_test"
# echo "=========================================="
# echo "Policy model: ${POLICY_MODEL_PATH} (${model_prefix})"
# echo "Dataset: ${DATASET} (${TOTAL_SAMPLES} samples)"
# echo "Number of checkpoints: ${#CHECKPOINT_NAMES[@]}"
# echo "Number of nodes: ${TOTAL_NODES}"
# echo "Development mode: ${DEVELOPMENT_MODE}"
# echo ""
# echo "Data distribution:"

# # Calculate samples per node
# SAMPLES_PER_NODE=$((TOTAL_SAMPLES / TOTAL_NODES))
# REMAINDER=$((TOTAL_SAMPLES % TOTAL_NODES))

# echo "  Base samples per node: ${SAMPLES_PER_NODE}"
# if [ $REMAINDER -gt 0 ]; then
#     echo "  Last node gets additional ${REMAINDER} samples"
# fi
# echo ""
# echo "Total jobs to submit: $((${#CHECKPOINT_NAMES[@]} * ${TOTAL_NODES}))"
# echo "Each job will use 8 GPUs (4 parallel partitions with 2 GPUs each)"
# echo ""

# # Loop through all checkpoint names
# for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
#     # Construct full checkpoint path
#     checkpoint="${CHECKPOINT_BASE_PATH}/${checkpoint_name}"

#     # Create short name for job
#     checkpoint_short=""
#     if [[ $checkpoint_name == *"custom_tok"* ]]; then
#         checkpoint_short+="custom_tok_"
#     elif [[ $checkpoint_name == *"normal_tok"* ]]; then
#         checkpoint_short+="normal_tok_"
#     fi

#     if [[ $checkpoint_name == *"ablation"* ]]; then
#         checkpoint_short+="ablation_"
#     elif [[ $checkpoint_name == *"balanced"* ]]; then
#         checkpoint_short+="balanced_"
#     elif [[ $checkpoint_name == *"full"* ]]; then
#         checkpoint_short+="full_"
#     fi

#     if [[ $checkpoint_name == *"Q3B"* ]]; then
#         checkpoint_short+="Q3B"
#     elif [[ $checkpoint_name == *"Q7B"* ]]; then
#         checkpoint_short+="Q7B"
#     fi

#     # Loop through 12 nodes
#     for node_id in {0..11}; do
#         # Calculate data range for this node
#         NODE_DATA_BEGIN=$((node_id * SAMPLES_PER_NODE))

#         if [ $node_id -eq 11 ]; then
#             # Last node gets all remaining samples
#             NODE_DATA_END=$TOTAL_SAMPLES
#         else
#             NODE_DATA_END=$(((node_id + 1) * SAMPLES_PER_NODE))
#         fi

#         NODE_SAMPLES=$((NODE_DATA_END - NODE_DATA_BEGIN))

#         # Create unique job name with node ID
#         datetime=$(date +"%Y%m%d-%H%M%S")
#         job_name="${base_job_name_prefix}-${checkpoint_short}-node${node_id}-${datetime}"

#         # Create log file path
#         log_file="reward_guided_search/logs/${job_name}.log"
#         abs_log_file="${PWD}/${log_file}"

#         # Ensure log directory exists
#         mkdir -p reward_guided_search/logs

#         echo "Submitting job $((job_count+1)): ${job_name}"
#         echo "  Node ID: ${node_id}"
#         echo "  Checkpoint: ${checkpoint}"
#         echo "  Data range: [${NODE_DATA_BEGIN}, ${NODE_DATA_END}) (${NODE_SAMPLES} samples)"
#         echo "  Log file: ${log_file}"

#         # Submit job with node-specific data range
#         job_id=$(qsub -r y \
#             -v CHECKPOINT_PATH="${checkpoint}",POLICY_MODEL_PATH="${POLICY_MODEL_PATH}",POLICY_MODEL_NAME="policy_${model_prefix}",DATASET="${DATASET}",DEVELOPMENT_MODE="${DEVELOPMENT_MODE}",NODE_DATA_BEGIN="${NODE_DATA_BEGIN}",NODE_DATA_END="${NODE_DATA_END}",NODE_ID="${node_id}",TOTAL_NODES="${TOTAL_NODES}",EVAL_RUN_LOG_FILE="${abs_log_file}" \
#             -N "${job_name}" \
#             -o "${abs_log_file}" \
#             vllm_run_bon_evaluation_parallel_node.pbs)

#         echo "  Job ID: ${job_id}"
#         echo ""

#         ((job_count++))

#         # Add delay between submissions
#         sleep 2
#     done
# done

# echo "=========================================="
# echo "Submitted ${job_count} jobs total"
# echo "Use 'qstat -u $USER' to check job status"
# echo ""
# echo "Log files are in: reward_guided_search/logs/"
# echo "Results will be in: reward_guided_search/outputs/policy_${model_prefix}/"
# echo ""
# echo "Each node processes:"
# for node_id in {0..11}; do
#     NODE_DATA_BEGIN=$((node_id * SAMPLES_PER_NODE))
#     if [ $node_id -eq 11 ]; then
#         NODE_DATA_END=$TOTAL_SAMPLES
#     else
#         NODE_DATA_END=$(((node_id + 1) * SAMPLES_PER_NODE))
#     fi
#     NODE_SAMPLES=$((NODE_DATA_END - NODE_DATA_BEGIN))
#     echo "  Node ${node_id}: samples [${NODE_DATA_BEGIN}, ${NODE_DATA_END}) = ${NODE_SAMPLES} samples"
# done
# echo ""
# echo "Note: Results will need to be merged across all 12 nodes after completion"