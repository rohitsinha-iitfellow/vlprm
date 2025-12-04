# Example Usage
# DATA_DIR="/home/ubuntu/porialab-us-midwest-1/Tej/mmr-eval/traces_data_backup"

source /scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/qwen-evaluation/.venv/bin/activate

datetime=$(date +"%Y%m%d-%H%M%S")

OUTPUT_DATA_DIR="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation/judge_step_eval/outputs"
BASE_DATA_DIR="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation/reward_guided_search/VisualPRM_relabelling"
INPUT_JSON_DATA_PATH="input_data/g27b_policy/q3b_prm/Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452_mathvision_result-parallel-merged-0-3040-20250922_124448.json"

MODEL_PATH="OpenGVLab/VisualPRM-8B"
# MODEL_PATH="OpenGVLab/VisualPRM-8B-v1_1"

if [[ $MODEL_PATH =~ [V]isualPRM-8B ]]; then
    model_prefix="V8B"
elif [[ $MODEL_PATH =~ [V]isualPRM-8B-v1_1 ]]; then
    model_prefix="V8B-v1_1"
else
    model_prefix="UNKNOWN"  # fallback
fi

RUN_SETTING="step_agg"
# RUN_SETTING="non_greedy"

base_job_name_prefix="PRM_${model_prefix}"

CUDA_VISIBLE_DEVICES=0 python prm_tts_eval.py \
    --model-path $MODEL_PATH \
    --data-path ${BASE_DATA_DIR}/${INPUT_JSON_DATA_PATH} \
    --output-path ${OUTPUT_DATA_DIR}/single_runs/prm_${model_prefix}/mathvision_${RUN_SETTING}-${datetime}.json \
    --tts-type ${RUN_SETTING} \
    2>&1 | tee logs/inference_logs/single_runs/prm_${model_prefix}/mathvision_${RUN_SETTING}-${datetime}

# Change to step_agg for step_aggregation
#     --tts-type step_agg \
