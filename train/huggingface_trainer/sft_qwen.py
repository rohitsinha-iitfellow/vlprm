import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
)
import dotenv
dotenv.load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_vision_component(model):
    """Get the vision component based on model type using isinstance checks"""

    # Qwen models use model.visual (both Qwen2-VL and Qwen2.5-VL)
    if isinstance(
        model, (Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)
    ):
        if hasattr(model, "visual"):
            return model.visual
        else:
            raise ValueError(
                f"Qwen model {model.__class__.__name__} does not have 'visual' attribute"
            )

    else:
        model_class_name = model.__class__.__name__

        raise ValueError(
            f"Unknown vision component for model type: {model_class_name}. No 'visual' or 'vision_tower' found."
        )


def set_model_vision_freezing(model, tune_vision):
    """Freeze/unfreeze vision encoder parameters - works for both Qwen models"""
    vision_component = get_vision_component(model)

    if tune_vision:
        for n, p in vision_component.named_parameters():
            p.requires_grad = True
    else:
        for n, p in vision_component.named_parameters():
            p.requires_grad = False

def log_vision_parameters(model):
    """Log vision encoder parameter counts for debugging"""
    vision_component = get_vision_component(model)

    total_vision_params = sum(p.numel() for p in vision_component.parameters())
    trainable_vision_params = sum(
        p.numel() for p in vision_component.parameters() if p.requires_grad
    )
    logging.info(f"Vision encoder - Total: {total_vision_params:,}, Trainable: {trainable_vision_params:,}")

def log_detailed_parameter_status(model):
    """Log detailed parameter status showing exactly which layers are frozen vs trainable"""
    vision_component = get_vision_component(model)

    logging.info("=" * 80)
    logging.info("DETAILED PARAMETER STATUS")
    logging.info("=" * 80)

    # Vision encoder detailed status - show actual component name
    if isinstance(
        model, (Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)
    ):
        vision_attr_name = "visual"
    else:
        # Fallback - determine by what attribute exists
        vision_attr_name = "visual" if hasattr(model, "visual") else "vision_tower"
    logging.info(f"\nVISION ENCODER (model.{vision_attr_name}) PARAMETERS:")
    vision_trainable = []
    vision_frozen = []

    for name, param in vision_component.named_parameters():
        param_count = param.numel()
        if param.requires_grad:
            vision_trainable.append((name, param_count))
        else:
            vision_frozen.append((name, param_count))
    
    logging.info(f"\nTRAINABLE Vision Parameters ({len(vision_trainable)} layers):")
    total_trainable = 0
    for name, count in vision_trainable:
        logging.info(f"  {name}: {count:,} params")
        total_trainable += count
    logging.info(f"  → Total trainable vision params: {total_trainable:,}")
    
    logging.info(f"\nFROZEN Vision Parameters ({len(vision_frozen)} layers):")
    total_frozen = 0
    for name, count in vision_frozen:
        logging.info(f"  {name}: {count:,} params")
        total_frozen += count
    logging.info(f"  → Total frozen vision params: {total_frozen:,}")
    
    # Overall model status
    logging.info("\nOVERALL MODEL PARAMETER STATUS:")
    total_model_params = sum(p.numel() for p in model.parameters())
    total_model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_model_frozen = total_model_params - total_model_trainable
    
    logging.info(f"  Total model parameters: {total_model_params:,}")
    logging.info(f"  Trainable parameters: {total_model_trainable:,} ({100*total_model_trainable/total_model_params:.1f}%)")
    logging.info(f"  Frozen parameters: {total_model_frozen:,} ({100*total_model_frozen/total_model_params:.1f}%)")
    
    # Memory savings estimate
    if total_model_frozen > 0:
        memory_savings_pct = 100 * total_model_frozen / total_model_params
        logging.info(f"  Estimated memory savings from freezing: ~{memory_savings_pct:.1f}%")
    
    logging.info("=" * 80)

def verify_parameter_freezing(model, expected_tune_vision=None):
    """Quick verification function to check if parameter freezing is working as expected"""
    vision_component = get_vision_component(model)

    vision_trainable_count = sum(
        1 for p in vision_component.parameters() if p.requires_grad
    )
    vision_total_count = sum(1 for p in vision_component.parameters())
    
    if expected_tune_vision is not None:
        if expected_tune_vision and vision_trainable_count == 0:
            logging.warning("WARNING: Expected tune_vision=True but no vision parameters are trainable!")
            return False
        elif not expected_tune_vision and vision_trainable_count > 0:
            logging.warning("WARNING: Expected tune_vision=False but some vision parameters are trainable!")
            return False
        else:
            logging.info(f"Parameter freezing verified: tune_vision={expected_tune_vision}")
            return True
    
    logging.info(f"Vision parameters: {vision_trainable_count}/{vision_total_count} trainable")
    return True

# Find the <image> token in the first user message with type image
def find_and_replace_image_token(messages, replacement_image):
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            for content_item in message["content"]:
                if (
                    content_item.get("type") == "image"
                    and content_item.get("text") == "<image>"
                ):
                    content_item["image"] = replacement_image
                    return messages
    return messages


@dataclass
class DataArguments:
    """
    Arguments for data processing
    """
    max_pixels: int = field(
        default=1280 * 28 * 28,  # Reduced from 576*28*28 for memory efficiency
        metadata={"help": "Maximum pixels for image processing (H*W)"},
    )
    min_pixels: int = field(
        default=256 * 28 * 28,
        metadata={"help": "Minimum pixels for image processing (H*W)"},
    )
    tune_vision: bool = field(
        default=False,
        metadata={"help": "Whether to train vision encoder parameters (False freezes VIT)"}
    )

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, DataArguments))
    script_args, training_args, model_args, data_args = parser.parse_args_and_config()
    
    # Memory-saving configurations to prevent OOM
    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    model_args.attn_implementation = "flash_attention_2"
    # training_args.learning_rate = training_args.learning_rate * training_args.gradient_accumulation_steps # Linear scaling rule
    # training_args.learning_rate = training_args.learning_rate * (training_args.gradient_accumulation_steps ** 0.5) # square root scaling rule

    # training_args.deepspeed = "train/configs/ds_config_zero3.json"
    
    if training_args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_args.logging_dir = os.path.join(training_args.output_dir, "run_logs", f"run-{timestamp}")
        # Create the logging directory if it doesn't exist
        os.makedirs(training_args.logging_dir, exist_ok=True)
        logging.info(f"Logging directory set to: {training_args.logging_dir}")
    
    # Enable Weights & Biases reporting while keeping physical text logs
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "multimodal-reasoning" # TODO
    os.environ["WANDB_ENTITY"] = "aisg-arf" # TODO
    logging.info("Enabled Weights & Biases reporting with project: multimodal-reasoning")
    
    if training_args.logging_dir:
        log_file = os.path.join(training_args.logging_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Physical log file created at: {log_file}")

    logging.info("\n\nscript_args: %s", script_args)
    logging.info("\ntraining_args: %s", training_args)
    logging.info("\nmodel_args: %s", model_args)
    logging.info("\ndata_args: %s", data_args)
    logging.info("\n\n")

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = None # full parameter
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        max_pixels=data_args.max_pixels,
        min_pixels=data_args.min_pixels,
    )

    model.to("cuda")
    # Move model to GPU when using Flash Attention 2.0
    if model_args.attn_implementation == "flash_attention_2" and torch.cuda.is_available():
        model = model.to('cuda')
        logging.info("Model moved to GPU for Flash Attention 2.0")

    logging.info("\n\nmodel_kwargs: %s", model_kwargs)
    logging.info("\nprocessor: %s", processor)
    logging.info("\nmodel: %s", model)

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples): # I think this is called at a batch level so we can use Image.open() here to load the image as a PIL list
        # Get the texts and images, and apply the chat template with JIT replace <image> with actual image
        processed_messages = [
            find_and_replace_image_token(example["messages"], example["image"])
            for example in examples
        ]
        logging.info(f"processed_messages: {processed_messages}")
        texts = [
            processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=False,
            )
            for msg in processed_messages
        ]
        logging.info(f"CHECKING Prompt with image token in texts: {texts[0]}")
        images = [process_vision_info(msg)[0] for msg in processed_messages]
        logging.info(
            f"DEBUG: Image should be PIL Image as input to processor: {[type(img[0]) if img else 'None' for img in images]}"
        )

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Ignore ALL the prompt token indexes in the loss computation, as we only care about the PRM token losses
        # First, mask everything as -100
        labels[:, :] = -100

        good_token_id = processor.tokenizer.convert_tokens_to_ids(
            "+"
        )
        bad_token_id = processor.tokenizer.convert_tokens_to_ids(
            "-"
        )

        # Find positions of these tokens and unmask them
        assistant_token_mask = (batch["input_ids"] == good_token_id) | (
            batch["input_ids"] == bad_token_id
        )
        labels[assistant_token_mask] = batch["input_ids"][assistant_token_mask]

        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    if os.getenv("HF_TOKEN") is None:
        raise ValueError("HF_TOKEN is not set")
    else:
        logging.info(f"HF_TOKEN: {os.getenv('HF_TOKEN')[:5]}...")

    training_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, token=os.getenv("HF_TOKEN"))
    
    # You can now use data_args.max_pixels and data_args.min_pixels in your dataset processing
    logging.info(f"Using max_pixels: {data_args.max_pixels}, min_pixels: {data_args.min_pixels}")


    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=training_dataset["train"], # train on full dataset for now
        eval_dataset=None,
        processing_class=processor.tokenizer,
    )

    # Apply vision encoder freezing
    set_model_vision_freezing(model, data_args.tune_vision)
    log_vision_parameters(model)
    log_detailed_parameter_status(model)
    verify_parameter_freezing(model, data_args.tune_vision)

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
    
    trainer.accelerator.wait_for_everyone()