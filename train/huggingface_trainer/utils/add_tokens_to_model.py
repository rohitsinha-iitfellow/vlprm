import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch

model_id = 'Qwen/Qwen2.5-VL-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    )

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu"
    )

existing_special_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])
GOOD_TOKEN='<+>'
BAD_TOKEN='<->'
# SEPERATOR_TOKEN='<extra>' # can use for MLM later
new_special_tokens = [GOOD_TOKEN, BAD_TOKEN]
updated_special_tokens = list(set(existing_special_tokens + new_special_tokens))
tokenizer.add_special_tokens({"additional_special_tokens": updated_special_tokens})

# Editing embedding of SEPERATOR_TOKEN
# with torch.no_grad():
#     extra_token_id = tokenizer.convert_tokens_to_ids(SEPERATOR_TOKEN)
#     base_token_id = tokenizer.pad_token_id  # or use "<mask>", "unused", etc.
#     model.get_input_embeddings().weight[extra_token_id] = model.get_input_embeddings().weight[base_token_id]

# Remove this if do not want to re-init token embeddings values
# embedding_matrix = model.language_model.embed_tokens.weight
# print(f"Embedding matrix shape: {embedding_matrix.shape}")
# print(f"Vocab size from embedding: {embedding_matrix.shape[0]}")
# print(f"Embedding dim from embedding: {embedding_matrix.shape[1]}")
# print(f"Model config hidden_size: {model.config.hidden_size}")

# # Check and initialize new token embeddings
# print("\n=== NEW TOKEN EMBEDDING INITIALIZATION ===")
# print(f"<-> embedding before: {embedding_matrix[151665][:5]}")  # First 5 values
# print(f"<+> embedding before: {embedding_matrix[151666][:5]}")

# # Initialize new token embeddings with small random values
# with torch.no_grad():
#     # Use the same initialization as the original embeddings
#     init_range = model.config.initializer_range  # Usually 0.02
#     embedding_matrix[151665:151667] = torch.randn(2, embedding_matrix.shape[1], 
#                                                  dtype=embedding_matrix.dtype,
#                                                  device=embedding_matrix.device) * init_range

# print(f"<-> embedding after: {embedding_matrix[151665][:5]}")
# print(f"<+> embedding after: {embedding_matrix[151666][:5]}")

# # Verify they're different now
# cos_sim = torch.nn.functional.cosine_similarity(
#     embedding_matrix[151665].unsqueeze(0),
#     embedding_matrix[151666].unsqueeze(0)
# )
# print(f"Cosine similarity after init: {cos_sim}")
# print("=" * 40)

save_directory = "./models/Qwen2.5-VL-7B-Instruct-updated-tokens"

os.makedirs(save_directory, exist_ok=True)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)