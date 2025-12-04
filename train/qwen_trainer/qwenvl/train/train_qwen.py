# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import traceback

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class
from tg_notifications.training_notifications import (
    send_training_completion_notification,
    send_training_error_notification,
)

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank
    
    # Early setup for potential error notifications
    training_log_path = ""
    pbs_output_file = os.getenv("PBS_OUTPUT_FILE", "")
    pbs_workdir = os.getenv("PBS_O_WORKDIR", "")
    
    if pbs_output_file and pbs_workdir:
        training_log_path = os.path.join(pbs_workdir, "qwen_training", pbs_output_file)
    elif pbs_output_file:
        training_log_path = pbs_output_file
    
    try:
        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        local_rank = training_args.local_rank
        os.makedirs(training_args.output_dir, exist_ok=True)

        if "qwen2.5" in model_args.model_name_or_path.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            data_args.image_processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
            ).image_processor
            data_args.model_type = "qwen2.5vl"
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
                model_args.model_name_or_path,
            )
            data_args.model_type = "qwen2vl"

        if data_args.data_flatten:
            replace_qwen2_vl_attention_class()
        model.config.use_cache = False

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
        
        if data_args.data_packing:
            data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
        else:
            data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
            trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )

        # Get training configuration for notifications
        run_name = training_args.run_name if hasattr(training_args, 'run_name') else "unnamed_run"
        
        # Additional training info for notifications
        extra_info = {
            "num_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        }
        
        # Training execution
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        
        # Save trainer state and model
        trainer.save_state()
        data_args.image_processor.save_pretrained(training_args.output_dir)
        
        model.config.use_cache = True
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        
        # Send success notification (only from rank 0)
        if torch.distributed.get_rank() == 0:
            print("Training completed successfully. Sending notification...")
            send_training_completion_notification(
                model_path_name=model_args.model_name_or_path,
                output_dir=training_args.output_dir,
                dataset_use=data_args.dataset_use,
                tune_mm_vision=model_args.tune_mm_vision,
                run_name=run_name,
                training_log_path=training_log_path,
                extra_info=extra_info,
            )
            print("Training completion notification sent successfully.")
        
    except Exception as e:
        # Log the error
        error_msg = f"Training failed with error: {str(e)}"
        print(error_msg)
        print(f"Full traceback: {traceback.format_exc()}")
        
        # Send error notification with available info (only from rank 0 if distributed is initialized)
        try:
            # Check if we're in distributed mode and only send from rank 0
            should_send_notification = True
            if 'torch' in locals() and hasattr(torch, 'distributed'):
                if torch.distributed.is_initialized():
                    should_send_notification = (torch.distributed.get_rank() == 0)
            
            if should_send_notification:
                # Use available variables with fallbacks
                model_path = model_args.model_name_or_path if 'model_args' in locals() else "Unknown"
                output_dir = training_args.output_dir if 'training_args' in locals() else "Unknown"
                dataset = data_args.dataset_use if 'data_args' in locals() else "Unknown"
                vision_tuning = model_args.tune_mm_vision if 'model_args' in locals() else False
                run_name_val = run_name if 'run_name' in locals() else "unnamed_run"
                extra_info_val = extra_info if 'extra_info' in locals() else {}
                
                send_training_error_notification(
                    model_path_name=model_path,
                    output_dir=output_dir,
                    dataset_use=dataset,
                    tune_mm_vision=vision_tuning,
                    run_name=run_name_val,
                    error=e,
                    training_log_path=training_log_path,
                    extra_info=extra_info_val,
                )
                print("Error notification sent successfully.")
        except Exception as notif_error:
            print(f"Failed to send error notification: {notif_error}")
        
        # Re-raise the original exception to maintain proper exit code
        raise


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
