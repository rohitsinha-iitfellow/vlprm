#!/bin/bash

# Example script for running parallel MMMU inference
# This demonstrates how to use the parallel launcher

# ============================================
# CONFIGURATION - Modify these as needed
# ============================================

# Model path (REQUIRED - modify this to your model path)
# export MODEL_PATH="google/gemma-3-27b-it"
export MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# Data directory (REQUIRED - path to MMMU data)
export DATA_DIR="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/qwen-evaluation/mmmu/data"  # UPDATE THIS!

# Dataset to evaluate
export DATASET="MMMU_DEV_VAL"  # Options: MMMU_DEV_VAL, MMMU_DEV, MMMU_TEST

# Output directory
export OUTPUT_DIR="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/qwen-evaluation/mmmu/results/parallel_inference"

# Chain-of-Thought settings
export USE_COT=true  # Set to true to enable CoT
# export COT_PROMPT=""  # Optional custom CoT prompt

# Development mode (for testing with small subset)
export DEVELOPMENT_MODE=false  # Set to true for testing with 16 samples

# ============================================
# VALIDATION
# ============================================

# Check if model path is set correctly
if [ "$MODEL_PATH" = "/path/to/your/Qwen2.5-VL-7B-Instruct" ]; then
    echo "ERROR: Please update MODEL_PATH in this script to point to your model"
    echo "Edit this file and set MODEL_PATH to the correct path"
    exit 1
fi

if [ "$DATA_DIR" = "/path/to/MMMU/data" ]; then
    echo "ERROR: Please update DATA_DIR in this script to point to your MMMU data"
    echo "Edit this file and set DATA_DIR to the correct path"
    exit 1
fi

# Check if paths exist
# if [ ! -d "$MODEL_PATH" ]; then
#     echo "ERROR: Model path does not exist: $MODEL_PATH"
#     exit 1
# fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Check GPU availability
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ $NUM_GPUS -lt 8 ]; then
    echo "WARNING: Found only $NUM_GPUS GPUs, but script expects 8"
    echo "The script will still attempt to run but may fail"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================
# RUN PARALLEL INFERENCE
# ============================================

echo "==========================================="
echo "Starting Parallel MMMU Inference"
echo "==========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Data Dir: $DATA_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Use CoT: $USE_COT"
echo "Development Mode: $DEVELOPMENT_MODE"
echo "Number of GPUs: 8"
echo "==========================================="
echo ""

# Activate virtual environment if it exists

if [ -f "../.venv/bin/activate" ]; then
    echo "Activating virtual environment from qwen-evaluation directory..."
    source ../.venv/bin/activate
fi

# Set Python path if needed
export PYTHONPATH="${PYTHONPATH}:/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval"

# Run the parallel launcher
./launch_parallel_inference.sh

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "Parallel inference completed successfully!"
    echo "==========================================="
    echo "Results are in: $OUTPUT_DIR"
    echo ""
    echo "To run evaluation on the results:"
    echo "  python run_mmmu.py eval \\"
    echo "    --data-dir \"$DATA_DIR\" \\"
    echo "    --input-file \"$OUTPUT_DIR/*/results_merged.jsonl\" \\"
    echo "    --output-file \"$OUTPUT_DIR/evaluation_results.csv\" \\"
    echo "    --dataset \"$DATASET\""
else
    echo ""
    echo "ERROR: Parallel inference failed with exit code $exit_code"
    echo "Check the logs in: $OUTPUT_DIR/*/logs/"
fi

exit $exit_code