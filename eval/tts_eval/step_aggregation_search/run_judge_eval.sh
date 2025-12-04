# Example Usage
# DATA_DIR="/home/ubuntu/porialab-us-midwest-1/Tej/mmr-eval/traces_data_backup"

# q7B mathvista
# CUDA_VISIBLE_DEVICES=0,1 python llm_as_judge_eval.py \
#     --model-path Qwen/Qwen2.5-VL-32B-Instruct \
#     --tensor-parallel-size 2 \
#     --data-path ${DATA_DIR}/q7b_policy_step/q3b_prm/Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452_mathvista_testmini_result-merged-0-1000-20250905_181632.json \
#     --output-path ${DATA_DIR}/q7b_policy_q32_judge/q3b_prm/Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452_mathvista_testmini_result-merged-0-1000-20250905_181632.json \
#     2>&1 | tee logs/q32b_as_judge/Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452_mathvista_testmini_result-merged-0-1000-20250905_181632.log

source /scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/qwen-evaluation/.venv/bin/activate

OUTPUT_DATA_DIR="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation/judge_step_eval/outputs"
BASE_DATA_DIR="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation/reward_guided_search/"
OUTPUT_JSON_DATA_PATH=""


CUDA_VISIBLE_DEVICES=0,1 python llm_as_judge_eval.py \
    --model-path Qwen/Qwen2.5-VL-32B-Instruct \
    --tensor-parallel-size 2 \
    --data-path ${BASE_DATA_DIR}/${OUTPUT_JSON_DATA_PATH} \
    --output-path ${OUTPUT_DATA_DIR}/q7b_policy_q32_judge/q3b_prm/Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452_mathvista_testmini_result-merged-0-1000-20250905_181632.json \
    2>&1 | tee logs/q32b_as_judge/Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452_mathvista_testmini_result-merged-0-1000-20250905_181632.log