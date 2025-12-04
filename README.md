<div align="center">
  <h1>Training Vision-Language Process Reward Models (VL-PRMs) for Test-Time Scaling in Multimodal Reasoning</h1>
  <p>Pairing VL-PRMs trained with abstract reasoning problems results in strong generalization and reasoning performance improvements when used with strong vision-language models in test-time scaling settings.
 </p>
</div>
<br>

<!-- ![](visuals/) -->

****************************************************************

**Updates:**

* 2025-09: Released VL-PRMs: [Arxiv](https://arxiv.org/pdf/2509.23250)

****************************************************************

This repository provides an overview of all resources for the paper ["Training Vision-Language Process Reward Models for Test-Time Scaling in Multimodal Reasoning: Key Insights and Lessons Learned"](https://arxiv.org/pdf/2509.23250).

- [Artifacts](#artifacts)
- [Structure](#structure)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Known Issues](#known-issues)

### Artifacts

- **Paper**: https://arxiv.org/pdf/2509.23250
- **VL-PRM300K Dataset**: 
    - [Raw Version](https://huggingface.co/datasets/ob11/VL-PRM300K)
    - [Training Version](https://huggingface.co/datasets/ob11/VL-PRM300K-V1-train) 
- **Model**:
    - [VL-PRM-3B](https://huggingface.co/ob11/Qwen-VL-PRM-3B)
    - [VL-PRM-7B](https://huggingface.co/ob11/Qwen-VL-PRM-7B)
- **Weight and Biases Training Logs**:
    - [VL-PRM-3B](https://wandb.ai/aisg-arf/multimodal-reasoning/runs/pnsncs80)
    - [VL-PRM-7B](https://wandb.ai/aisg-arf/multimodal-reasoning/runs/pj4oc0qh)
- **Evaluation Files**: 
    - All our evaluation result files are at: https://huggingface.co/datasets/ob11/VL-PRM-Evaluation-Results

### Structure

- `rollout_data_generation/`: Synthetic data creation code
- `train/`: Training scripts
- `eval/`: Evaluation scripts

### Training

To train VL-PRM-3B/VL-PRM-7B, you can use the `train/huggingface_trainer/train/sft_qwen.sh` script to launch a training job, or you can launch a sweep via `train/huggingface_trainer/launch_sft_qwen_sweep_clean.sh` if you are on a PBS cluster.

Alternatively, you can use the `train/qwen_trainer/sft_7b.sh` script, which works for VL-PRM-3B/VL-PRM-7B, or you can launch a sweep via `train/qwen_trainer/launch_sweep_7b_sft.sh` if you are on a PBS cluster. This trainer is adapted from the official Qwen2.5-VL repository [link](https://github.com/QwenLM/Qwen2.5-VL/tree/main/train). You need to set the image path in `train/qwen_trainer/qwenvl/data/data_qwen.py` or `train/qwen_trainer/qwenvl/data/data_qwen_packed.py`. Set your HF_TOKEN in the `.env.pbs` file.

The first method uses the HuggingFace Trainer API out of box, the second method uses HuggingFace Trainer with some input processing recommended by the official Qwen2.5-VL repository [link](https://github.com/QwenLM/Qwen2.5-VL/tree/main/train) to speed up training. We experience similar results with both methods.

To train VL-PRM-3B/VL-PRM-7B, we recommend 8 H100/H200 GPUs i.e. 1 node with 8 GPUs. We used 8 H200 GPUs for most of our jobs.

Quick start:
```
git clone https://github.com/theogbrand/vlprm.git
cd vlprm/train/huggingface_trainer
uv pip install -r requirements.txt
bash train/sft_qwen.sh
```
*Note: If you encounter an out-of-memory (OOM) issues, consider reducing the batch size and gradient accumulation steps.*

### Evaluation

We cloned [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) at commit `db0c9ae2c7c2150b9b730b88326ebfb0bfb91356` and modified it accordingly based on the base policy model and evaluation task selected. 

We recommend using vLLM version 0.10.1.1 with V1 engine, transformers version 4.55.2 and flash-attn version 2.8.0.post2 for inference. We experience varying results across models when different versions of these key packages are used, and found fixing these versions resulted in the best overall performance. 

You can launch the evaluation using the corresponding launch scripts in each evaluation directory. For example, `eval/tts_eval/reward_guided_search/vllm_launch_bon_evaluation.sh`. 

Evaluation result files can be found at: https://huggingface.co/datasets/ob11/VL-PRM-Evaluation-Results/

### Data

To recreate VL-PRM-300K follow the steps in `rollout_data_generation/README.md`. 

### Known Issues

- Reproducing the results in the paper requires using the exact same seed for the synthetic data generation and training the VL-PRMs.

### Citation

```bibtex
@misc{ong2025vlprms,
      title={Training Vision-Language Process Reward Models for Test-Time Scaling in Multimodal Reasoning: Key Insights and Lessons Learned}, 
      author={Brandon Ong, Tej Deep Pala, Vernon Toh, William Chandra Tjhi, and Soujanya Poria},
      year={2025},
      eprint={2509.23250},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/pdf/2509.23250}, 
}
```
