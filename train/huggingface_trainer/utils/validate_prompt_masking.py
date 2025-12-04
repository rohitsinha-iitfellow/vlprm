from PIL.Image import logging
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
)
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import torch

model_id = "./models/Qwen2.5-VL-7B-Instruct-updated-tokens-random-init-vals"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="cpu"
)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

# # Check if new token embeddings are initialized (not zeros/random)
# print(f"<-> embedding (151665): {model.language_model.embed_tokens.weight[151665]}")
# print(f"<+> embedding (151666): {model.language_model.embed_tokens.weight[151666]}")

# # Check if they're different from each other
# cos_sim = torch.nn.functional.cosine_similarity(
#     model.language_model.embed_tokens.weight[151665].unsqueeze(0),
#     model.language_model.embed_tokens.weight[151666].unsqueeze(0)
# )
# print(f"Cosine similarity between new tokens: {cos_sim}")

# exit()


messages = [{
    "content": [
      {
        "text": "You are a Visual Reasoning Teacher. Given a visual reasoning question with provided images and a student's solution, evaluate the visual interpretation accuracy, logical consistency of the current step, and whether it will lead to the correct final solution.",
        "type": "text"
      }
    ],
    "role": "system"
  },
  {
    'role': 'user',
    'content': [
        {'type': 'image',
      'image': '<image>'},
        {'type': 'text',
      'text': '### Question:\nWhat year was this picture taken? Answer the question using a single word or phrase.\n\n### Solution Process:\n[Visual Elements]\n<step_1>\nObserving several brown stuffed teddy bears, some with white paws, hanging close together, likely as display items.\n</step_1>'}]
  },
  {
    'role': 'assistant',
    'content': [{'type': 'text',
      'text': '<+>'}]
  },
  {
    'role': 'user',
    'content': [
        {'type': 'text',
      'text': '<step_2>\nThe second row contains three images: each has a large square, with a small black circle in the center. The black circle appears to decrease in size from left to right.\n</step_2>'}]
  },
  {
    'role': 'assistant',
    'content': [{'type': 'text',
      'text': '<->'}]
  }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)


# The labels are the input_ids, and we mask the padding tokens in the loss computation
labels = inputs["input_ids"].clone()

# Ignore ALL the prompt token indexes in the loss computation, as we only care about the PRM <+> and <-> token losses
# First, mask everything as -100
labels[:, :] = -100

# Now unmask only the assistant tokens (<+> and <->)
good_token_id = processor.tokenizer.convert_tokens_to_ids(
    "<+>"
)
bad_token_id = processor.tokenizer.convert_tokens_to_ids(
    "<->"
)

# verify tokens are single token ids
if not good_token_id == 151666 or not bad_token_id == 151665:
    raise ValueError(f"PRM tokens must be single tokens. Got: {good_token_id}, {bad_token_id}")

if processor.tokenizer.pad_token_id in [good_token_id, bad_token_id]:
    raise ValueError("Pad token ID conflicts with PRM token IDs")

# Find positions of these tokens and unmask them
assistant_token_mask = (inputs["input_ids"] == good_token_id) | (
    inputs["input_ids"] == bad_token_id
)
labels[assistant_token_mask] = inputs["input_ids"][assistant_token_mask]

inputs["labels"] = labels

# Debugging: Apply the debugging tips
print("=== DEBUGGING PROMPT MASKING ===")

# 1. Count how many tokens are not masked per example
non_masked_count = (labels != -100).sum(dim=1)
print(f"Non-masked tokens per example: {non_masked_count}")

# 2. Count PRM tokens specifically
good_count = (labels == good_token_id).sum(dim=1)
bad_count = (labels == bad_token_id).sum(dim=1)
print(f"Good tokens (<+>): {good_count}, Bad tokens (<->): {bad_count}")

# 3. Show the original input tokens and their labels side by side
print("\n=== TOKEN ANALYSIS ===")
for batch_idx in range(inputs["input_ids"].shape[0]):
    input_ids = inputs["input_ids"][batch_idx]
    label_ids = labels[batch_idx]
    
    print(f"\nBatch {batch_idx}:")
    print("Position | Input Token | Label | Token Text")
    print("-" * 50)
    
    for pos in range(len(input_ids)):
        input_token = input_ids[pos].item()
        label_token = label_ids[pos].item()
        
        # Decode the token
        try:
            token_text = processor.tokenizer.decode([input_token])
            # Clean up the token text for display
            token_text = repr(token_text)  # This will show special characters properly
        except:
            token_text = f"<UNKNOWN:{input_token}>"
        
        # Mark special tokens
        status = ""
        if label_token == -100:
            status = "MASKED"
        elif input_token == good_token_id:
            status = "GOOD_TOKEN"
        elif input_token == bad_token_id:
            status = "BAD_TOKEN"
        else:
            status = "UNMASKED"
        
        print(f"{pos:8d} | {input_token:11d} | {label_token:5d} | {token_text:20s} | {status}")

# 4. Verify no padding tokens are being treated as PRM tokens
if processor.tokenizer.pad_token_id is not None:
    pad_conflicts = (inputs["input_ids"] == processor.tokenizer.pad_token_id) & (labels != -100)
    if pad_conflicts.any():
        print(f"\nWARNING: Found padding tokens that are not masked: {pad_conflicts.sum()} tokens")
    else:
        print(f"\n✓ No padding token conflicts found")

# 5. Show the actual decoded text to understand context
print(f"\n=== FULL TEXT ANALYSIS ===")
full_text = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
print(f"Full decoded text: {full_text}")

# 6. Show only the tokens that will contribute to loss
non_masked_positions = labels[0] != -100
if non_masked_positions.any():
    contributing_tokens = inputs["input_ids"][0][non_masked_positions]
    contributing_text = processor.tokenizer.decode(contributing_tokens, skip_special_tokens=False)
    print(f"Tokens contributing to loss: {contributing_text}")
    print(f"Token IDs contributing to loss: {contributing_tokens.tolist()}")
else:
    print("WARNING: No tokens will contribute to loss!")

print(f"\nOriginal labels tensor: {inputs['labels']}")