#!/bin/bash

# Parallel launcher script for MMMU inference using 8 GPUs
# Each GPU runs an independent process on a dataset partition

# Configuration
MODEL_PATH=${MODEL_PATH:-""}  # Set via environment or modify here
DATASET=${DATASET:-"MMMU_DEV_VAL"}  # Options: MMMU_DEV_VAL, MMMU_TEST
DATA_DIR=${DATA_DIR:-""}  # Path to data directory
OUTPUT_DIR=${OUTPUT_DIR:-"outputs"}
USE_COT=${USE_COT:-false}
COT_PROMPT=${COT_PROMPT:-""}
NUM_GPUS=8

# Validate required parameters
if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH not set. Please provide the model path."
    echo "Usage: MODEL_PATH=/path/to/model ./launch_parallel_inference.sh"
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR not set. Please provide the data directory path."
    echo "Usage: DATA_DIR=/path/to/data MODEL_PATH=/path/to/model ./launch_parallel_inference.sh"
    exit 1
fi

# Determine dataset size
case "$DATASET" in
    "MMMU_DEV_VAL")
        TOTAL_SAMPLES=900
        ;;
    "MMMU_DEV")
        TOTAL_SAMPLES=150
        ;;
    "MMMU_TEST")
        TOTAL_SAMPLES=10500
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Using default size of 900"
        TOTAL_SAMPLES=900
        ;;
esac

# Development mode for testing
DEVELOPMENT_MODE=${DEVELOPMENT_MODE:-false}
if [ "$DEVELOPMENT_MODE" = "true" ]; then
    TOTAL_SAMPLES=16  # 2 samples per GPU for testing
    echo "DEVELOPMENT MODE: Using only $TOTAL_SAMPLES samples"
fi

# Calculate partition sizes
CHUNK_SIZE=$((TOTAL_SAMPLES / NUM_GPUS))
REMAINDER=$((TOTAL_SAMPLES % NUM_GPUS))

# Create output directory and timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET}_${TIMESTAMP}"
mkdir -p "$RUN_OUTPUT_DIR"
mkdir -p "$RUN_OUTPUT_DIR/logs"

# Base output filename
OUTPUT_FILE="${RUN_OUTPUT_DIR}/results.jsonl"

echo "==========================================="
echo "MMMU Parallel Inference Configuration"
echo "==========================================="
echo "Model Path: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Data Directory: $DATA_DIR"
echo "Total Samples: $TOTAL_SAMPLES"
echo "Number of GPUs: $NUM_GPUS"
echo "Samples per GPU: $CHUNK_SIZE (last GPU handles +$REMAINDER)"
echo "Output Directory: $RUN_OUTPUT_DIR"
echo "Use CoT: $USE_COT"
echo "Timestamp: $TIMESTAMP"
echo "==========================================="
echo ""

# Array to store process IDs
declare -a pids

# Launch parallel processes
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # Calculate data range for this partition
    START=$((gpu_id * CHUNK_SIZE))
    if [ $gpu_id -eq $((NUM_GPUS - 1)) ]; then
        # Last partition gets remaining samples
        END=$TOTAL_SAMPLES
    else
        END=$(((gpu_id + 1) * CHUNK_SIZE))
    fi
    
    # Create partition log file
    LOG_FILE="${RUN_OUTPUT_DIR}/logs/partition_${gpu_id}.log"
    
    echo "Launching GPU $gpu_id:"
    echo "  Data range: [$START, $END)"
    echo "  Samples: $((END - START))"
    echo "  Log: $LOG_FILE"
    
    # Build command
    CMD="python vllm_run_mmmu.py infer \
        --model-path \"$MODEL_PATH\" \
        --dataset \"$DATASET\" \
        --data-dir \"$DATA_DIR\" \
        --output-file \"$OUTPUT_FILE\" \
        --data-begin $START \
        --data-end $END \
        --partition-id $gpu_id \
        --gpu-id $gpu_id"
    
    # Add CoT options if enabled
    if [ "$USE_COT" = "true" ]; then
        CMD="$CMD --use-cot"
        if [ -n "$COT_PROMPT" ]; then
            CMD="$CMD --cot-prompt \"$COT_PROMPT\""
        fi
    fi
    
    # Launch process in background
    eval $CMD > "$LOG_FILE" 2>&1 &
    pids[$gpu_id]=$!
    
    echo "  PID: ${pids[$gpu_id]}"
    echo ""
    
    # Small delay to avoid initialization race conditions
    sleep 2
done

echo "All processes launched. Monitoring progress..."
echo ""

# Monitor progress
WAIT_INTERVAL=30
ELAPSED=0

while true; do
    ALL_DONE=true
    echo "Status check at $(date) (elapsed: ${ELAPSED}s):"
    
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        if kill -0 ${pids[$gpu_id]} 2>/dev/null; then
            echo "  GPU $gpu_id (PID ${pids[$gpu_id]}): RUNNING"
            ALL_DONE=false
        else
            wait ${pids[$gpu_id]}
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                echo "  GPU $gpu_id (PID ${pids[$gpu_id]}): COMPLETED"
            else
                echo "  GPU $gpu_id (PID ${pids[$gpu_id]}): FAILED (exit code: $EXIT_CODE)"
            fi
        fi
    done
    
    if [ "$ALL_DONE" = true ]; then
        break
    fi
    
    echo ""
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

echo ""
echo "All processes have finished. Checking results..."

# Check if all partitions succeeded
SUCCESS=true
FAILED_PARTITIONS=""

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    wait ${pids[$gpu_id]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: GPU $gpu_id failed with exit code $EXIT_CODE"
        echo "Check log: ${RUN_OUTPUT_DIR}/logs/partition_${gpu_id}.log"
        SUCCESS=false
        FAILED_PARTITIONS="$FAILED_PARTITIONS $gpu_id"
    fi
done

if [ "$SUCCESS" = false ]; then
    echo "ERROR: One or more partitions failed. Failed partitions:$FAILED_PARTITIONS"
    echo "Attempting to merge available results anyway..."
fi

echo ""
echo "==========================================="
echo "Merging partition results..."
echo "==========================================="

# Merge results
MERGE_CMD="python merge_partition_results.py \
    --output-dir \"$RUN_OUTPUT_DIR\" \
    --dataset \"$DATASET\" \
    --num-partitions $NUM_GPUS \
    --timestamp \"$TIMESTAMP\""

eval $MERGE_CMD

MERGE_EXIT_CODE=$?

if [ $MERGE_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Merge script failed with exit code $MERGE_EXIT_CODE"
    exit $MERGE_EXIT_CODE
fi

echo ""
echo "==========================================="
echo "Parallel inference completed at $(date)"
echo "==========================================="
echo "Results directory: $RUN_OUTPUT_DIR"
echo "Merged results: ${RUN_OUTPUT_DIR}/results_merged.jsonl"
echo ""

# Display summary statistics if available
if [ -f "${RUN_OUTPUT_DIR}/results_merged.jsonl" ]; then
    TOTAL_LINES=$(wc -l < "${RUN_OUTPUT_DIR}/results_merged.jsonl")
    echo "Total samples processed: $TOTAL_LINES"
fi

exit 0