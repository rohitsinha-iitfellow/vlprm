source .venv/bin/activate
source .env.pbs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-VL-7B-Instruct"
dataset_name="ob11/VL-PRM300K-V1-train"
epochs=2
micro_batch_size=2 # -> batch_size will be 64 if 8 gpus, per device batch size in single node; max this without OOM
gradient_accumulation_steps=32 # gradually increase first, requires more GPU memory but less than increasing micro_batch_size
lr=1e-5
max_steps=-1 # only used for streaming, set later
min_lr=1e-6
weight_decay=1e-4
gpu_count=$(nvidia-smi -L | wc -l) # -> not used now
push_to_hub=false # -> not used now

accelerate launch --config_file=train/configs/deepspeed_zero2.yaml \
    train/sft_qwen.py \
    --dataset_name ${dataset_name} \
    --model_name_or_path ${base_model} \
    --per_device_train_batch_size ${micro_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_dir training_outputs/sft-$(echo ${dataset_name} | rev | cut -d'-' -f1-5 | rev)-${base_model}-${uid} \
    --bf16 True \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --num_train_epochs ${epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \