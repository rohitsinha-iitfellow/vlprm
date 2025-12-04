#!/bin/bash

# Launcher script to submit PBS jobs for PRM evaluation
# Similar to launch_bon_evaluation.sh but for PRM evaluation

# Model configuration
MODEL_PATH="OpenGVLab/VisualPRM-8B"
# MODEL_PATH="OpenGVLab/VisualPRM-8B-v1_1"

# Data configuration  
BASE_DATA_DIR="../../evaluation/reward_guided_search/VisualPRM_relabelling"
# INPUT_JSON_DATA_PATH="test_input_data"
INPUT_JSON_DATA_PATH="input_data"

# Run settings to evaluate
RUN_SETTINGS=(
    "step_agg"
    "non_greedy"
)

DATASETS=(
    # "AlgoPuzzleVQA_900_subset"  # AlgoPuzzleVQA 900 subset: 50 samples from each of 18 puzzle types (900 total)
    # "puzzleVQA_1K_subset"  # PuzzleVQA 1K subset: 50 samples from each of 20 puzzle types (1000 total)
    # "mathvista_testmini"
    "MMMU_DEV_VAL" # Original MMMU dataset from TSV file
    # "mathvision_test"         
)
# Base job name
base_job_name="JS"

# Counter for job submissions
job_count=0

echo "Starting PBS job sweep for PRM evaluation at $(date)"
echo "Model: ${MODEL_PATH}"
echo "Data path: ${BASE_DATA_DIR}/${INPUT_JSON_DATA_PATH}"
echo "Run settings: ${RUN_SETTINGS[*]}"
echo "Filtering for datasets: ${DATASETS[*]}"
echo ""

# Discover all policy model directories
POLICY_MODELS=($(find "${BASE_DATA_DIR}/${INPUT_JSON_DATA_PATH}" -maxdepth 1 -type d -name "*policy*" | sort))

if [ ${#POLICY_MODELS[@]} -eq 0 ]; then
    echo "ERROR: No policy model directories found in ${BASE_DATA_DIR}/${INPUT_JSON_DATA_PATH}"
    exit 1
fi

echo "Found ${#POLICY_MODELS[@]} policy model(s):"
for policy in "${POLICY_MODELS[@]}"; do
    echo "  $(basename "$policy")"
done
echo ""

total_jobs=0

# Loop through each policy model
for POLICY_DIR in "${POLICY_MODELS[@]}"; do
    POLICY_MODEL=$(basename "$POLICY_DIR")
    echo "Processing policy model: $POLICY_MODEL"
    
    # Find all reward model directories within this policy
    REWARD_MODELS=($(find "$POLICY_DIR" -maxdepth 1 -type d -name "*prm*" | sort))
    
    if [ ${#REWARD_MODELS[@]} -eq 0 ]; then
        echo "  WARNING: No reward model directories found in $POLICY_DIR"
        continue
    fi
    
    echo "  Found ${#REWARD_MODELS[@]} reward model(s):"
    for reward in "${REWARD_MODELS[@]}"; do
        echo "    $(basename "$reward")"
    done
    
    # Loop through each reward model
    for REWARD_DIR in "${REWARD_MODELS[@]}"; do
        REWARD_MODEL=$(basename "$REWARD_DIR")
        echo "  Processing reward model: $REWARD_MODEL"
        
        # Find all JSON files in this reward model directory
        ALL_JSON_FILES=($(find "$REWARD_DIR" -maxdepth 1 -name "*.json" | sort))
        
        if [ ${#ALL_JSON_FILES[@]} -eq 0 ]; then
            echo "    WARNING: No JSON files found in $REWARD_DIR"
            continue
        fi
        
        # Filter JSON files based on DATASETS patterns (case-insensitive)
        JSON_FILES=()
        for json_file in "${ALL_JSON_FILES[@]}"; do
            json_basename=$(basename "$json_file" .json)
            for dataset in "${DATASETS[@]}"; do
                if [[ "${json_basename,,}" == *"${dataset,,}"* ]]; then
                    JSON_FILES+=("$json_file")
                    break
                fi
            done
        done
        
        if [ ${#JSON_FILES[@]} -eq 0 ]; then
            echo "    WARNING: No JSON files matching dataset patterns found in $REWARD_DIR"
            echo "    Available files: $(printf '%s ' "${ALL_JSON_FILES[@]##*/}")"
            continue
        fi
        
        echo "    Found ${#JSON_FILES[@]} JSON file(s) matching dataset patterns (out of ${#ALL_JSON_FILES[@]} total)"
        
        # Loop through each JSON file and each run setting
        for JSON_FILE in "${JSON_FILES[@]}"; do
            JSON_BASENAME=$(basename "$JSON_FILE" .json)
            echo "    Processing JSON: $JSON_BASENAME"
            
            for SETTING in "${RUN_SETTINGS[@]}"; do
                # Create unique job name with hierarchy info
                datetime=$(date +"%Y%m%d-%H%M%S")
                job_name="${base_job_name}-${POLICY_MODEL}-${REWARD_MODEL}-${SETTING}-${datetime}"
                
                # Create hierarchical log file path
                log_file="logs/${POLICY_MODEL}/${REWARD_MODEL}/${JSON_BASENAME}-${SETTING}-${datetime}.log"
                abs_log_file="${PWD}/${log_file}"
                
                # Ensure log directory exists
                mkdir -p "logs/${POLICY_MODEL}/${REWARD_MODEL}"
                
                # Calculate relative path from BASE_DATA_DIR to JSON file
                RELATIVE_JSON_PATH="${JSON_FILE#${BASE_DATA_DIR}/}"
                
                echo "      Submitting job $((job_count+1)): ${job_name}"
                echo "        JSON: $JSON_BASENAME"
                echo "        Setting: ${SETTING}"
                echo "        Log: ${log_file}"
                
                # Submit job with all necessary environment variables
                job_id=$(qsub -v MODEL_PATH="${MODEL_PATH}",BASE_DATA_DIR="${BASE_DATA_DIR}",JSON_FILE_PATH="${RELATIVE_JSON_PATH}",RUN_SETTING="${SETTING}",POLICY_MODEL="${POLICY_MODEL}",REWARD_MODEL="${REWARD_MODEL}",JSON_BASENAME="${JSON_BASENAME}",EVAL_RUN_LOG_FILE="${abs_log_file}",DATETIME="${datetime}" \
                        -N "${job_name}" -o "${abs_log_file}" run_prm_eval.pbs)
                
                echo "        Job ID: ${job_id}"
                echo ""
                
                ((job_count++))
                ((total_jobs++))
                
                # Add delay between submissions to avoid overwhelming the scheduler
                sleep 1
            done
        done
    done
    echo ""
done

echo "=========================================="
echo "Job submission summary:"
echo "  Policy models processed: ${#POLICY_MODELS[@]}"
echo "  Total jobs submitted: $total_jobs"
echo "  Run settings per file: ${#RUN_SETTINGS[@]}"
echo "=========================================="

echo "Submitted ${job_count} jobs total"
echo "Use 'qstat -u $USER' to check job status"
echo ""
echo "Log files are in: logs/"
echo "Results will be in: outputs/"
