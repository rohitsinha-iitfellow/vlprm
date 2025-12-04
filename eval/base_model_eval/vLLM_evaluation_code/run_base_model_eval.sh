export PYTHONPATH="${PYTHONPATH}:/data/projects/71001002/ob1/vlprm/"
# export PYTHONPATH="${PYTHONPATH}:<absolute_path_to_parent_dir>"

source /data/projects/71001002/ob1/vlprm/eval/.venv/bin/activate
echo "Python path after activation: $(which python)"
echo "Python version: $(python --version)"


dataset_list=(
    "mathvista_testmini"
    # "mathvision_test"
)

# POLICY_MODEL_PATH="OpenGVLab/InternVL2_5-8B"
# POLICY_MODEL_PATH="openbmb/MiniCPM-V-2_6"
POLICY_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

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
elif [[ $POLICY_MODEL_PATH =~ [Ii]nternVL2_5-8B ]]; then
    model_prefix="I8B"
elif [[ $POLICY_MODEL_PATH =~ [Mm]iniCPM-V-2_6 ]]; then
    model_prefix="M26"
else
    model_prefix="UNKNOWN"  # fallback
fi

base_job_name_prefix="vLLM_${model_prefix}"
CUDA_VISIBLE_DEVICES="0"

if [ "$POLICY_MODEL_PATH" == "openbmb/MiniCPM-V-2_6" ]; then
    # RUN_FILE="run_inference_and_judge_minicpm_vlmevalkit.py"
    RUN_FILE="run_inference_and_judge.py"
else
    RUN_FILE="run_inference_and_judge.py"
fi

# Performing Search and Generating Responses
for dataset in "${dataset_list[@]}"; do
    echo "Dataset: $dataset"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $RUN_FILE \
        --policy_model_path $POLICY_MODEL_PATH \
        --data $dataset \
        --output_dir ./outputs/$model_prefix/$dataset-results \
        --development_mode
done


# Evaluating Responses

# python collate_final_eval_results.py path_to_json_result_file.json