from transformers import AutoTokenizer

# Replace with your actual model/tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Decode the full sequence
input_ids = [151644, 151655, 151665, 151666]  # your full list
decoded_text = tokenizer.decode(input_ids)
print("Full text:", decoded_text)

# Decode individual tokens to see token boundaries
for i, token_id in enumerate(input_ids):
    token_text = tokenizer.decode([token_id])
    print(f"Position {i}: ID {token_id} = '{token_text}'")