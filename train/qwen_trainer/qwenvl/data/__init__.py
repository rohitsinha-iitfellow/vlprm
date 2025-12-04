import re

LATEST_SNAPSHOT_ID = "276cb95c6d69b655ed694aff6111766528e034bd"
SNAPSHOT_BASE_PATH = f"/scratch_aisg/SPEC-SF-AISG/cache/huggingface/hub/datasets--ob11--visualPRM_v4/snapshots/{LATEST_SNAPSHOT_ID}"

MC0_VISUAL_PRM_DATA_CUSTOM_TOK_ERRORS_BALANCED_v4 = {
    "annotation_path": f"{SNAPSHOT_BASE_PATH}/custom_token/mc0_custom_tok_balanced_errors_v4.jsonl",
    "data_path": "",  # Empty since image paths in annotations are absolute
}

MC0_VISUAL_PRM_DATA_CUSTOM_TOK_FULL_NON_BALANCED_v2 = {
    "annotation_path": f"{SNAPSHOT_BASE_PATH}/custom_token/mc0_custom_token_full_dataset_non_balanced_v2.jsonl",
    "data_path": "",
}

MC0_VISUAL_PRM_DATA_CUSTOM_TOK_ABLATION_ONLY_REASONING_ERRORS_v4 = {
    "annotation_path": f"{SNAPSHOT_BASE_PATH}/custom_token/mc0_custom_tok_ablation_no_perception_errors_v4.jsonl",
    "data_path": "",
}

MC0_VISUAL_PRM_DATA_NORMAL_TOK_ERRORS_BALANCED_v2 = {
    "annotation_path": f"{SNAPSHOT_BASE_PATH}/normal_token/mc0_normal_tok_balanced_all_reasoning_errors_with_perception_errors_v2_130K.jsonl",
    "data_path": "",  # Empty since image paths in annotations are absolute
}

MC0_VISUAL_PRM_DATA_NORMAL_TOK_FULL_NON_BALANCED_v2 = {
    "annotation_path": f"{SNAPSHOT_BASE_PATH}/normal_token/mc0_normal_tok_full_dataset_non_balanced_all_perception_and_reasoning_errors_v2_300K.jsonl",
    "data_path": "",
}

MC0_VISUAL_PRM_DATA_NORMAL_TOK_ABLATION_ONLY_REASONING_ERRORS_v2 = {
    "annotation_path": f"{SNAPSHOT_BASE_PATH}/normal_token/mc0_normal_tok_ablation_all_reasoning_errors_no_perception_errors_v2_100K.jsonl",
    "data_path": "",
}

data_dict = {
    "mc0_visualprm_data_custom_tok_errors_balanced_v4": MC0_VISUAL_PRM_DATA_CUSTOM_TOK_ERRORS_BALANCED_v4,
    "mc0_visualprm_data_custom_tok_full_non_balanced_v2": MC0_VISUAL_PRM_DATA_CUSTOM_TOK_FULL_NON_BALANCED_v2,
    "mc0_visualprm_data_custom_tok_ablation_only_reasoning_errors_v4": MC0_VISUAL_PRM_DATA_CUSTOM_TOK_ABLATION_ONLY_REASONING_ERRORS_v4,
    "mc0_visualprm_data_normal_tok_errors_balanced_v2": MC0_VISUAL_PRM_DATA_NORMAL_TOK_ERRORS_BALANCED_v2,
    "mc0_visualprm_data_normal_tok_full_non_balanced_v2": MC0_VISUAL_PRM_DATA_NORMAL_TOK_FULL_NON_BALANCED_v2,
    "mc0_visualprm_data_normal_tok_ablation_only_reasoning_errors_v2": MC0_VISUAL_PRM_DATA_NORMAL_TOK_ABLATION_ONLY_REASONING_ERRORS_v2,
}

# TODO: remember to change dataset basepath in qwen_training/qwenvl/data/data_qwen.py

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["mc0_visualprm_data_custom_tok_errors_balanced_v3%100", "mc0_visualprm_data_custom_tok_full_non_balanced_v1%100", "mc0_visualprm_data_normal_tok_errors_balanced_v2%100", "mc0_visualprm_data_normal_tok_full_non_balanced_v2%100", "mc0_visualprm_data_normal_tok_ablation_only_reasoning_errors_v2%100", "mc0_visualprm_data_custom_tok_ablation_only_reasoning_errors_v3%100"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
