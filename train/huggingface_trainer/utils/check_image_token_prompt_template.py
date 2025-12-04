import numpy as np
from PIL import Image
import transformers
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
# print transformers version
# print(f"Transformers version: {transformers.__version__}")

# Replace with your actual model/tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=1280*28*28, min_pixels=256*28*28)

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": 'You are an advanced AI assistant, designed to serve as a process supervision model. In this task, I will provide a problem statement followed by the first step of the solution process. For each subsequent turn, I will give you a new step in the solution. Your role is to assess whether the solution process is correct up to the current step.\n\n**Critical Note**: This is a vision-language task where the problem is intrinsically linked to an accompanying image. The correctness of each solution step must be evaluated against the specific visual information presented in the image. Any interpretation, calculation, or reasoning that contradicts or misrepresents the visual data in the image must be considered incorrect, regardless of whether the logic would be sound in isolation.\n\nYou will evaluate solution steps based on two critical dimensions:\n\n**[Visual Elements]**: The perceptual accuracy of visual interpretations, spatial relationships, object recognition, and visual data extraction directly from the provided image.\n\n**[Reasoning]**: The logical consistency of inferences, mathematical operations, deductive steps, and analytical conclusions as they relate to the visual information.\n\n## Task Structure:\n- **First round**: You will receive the problem statement and the first solution step.\n- **Subsequent rounds**: You will receive one new solution step per round.\n\n## Evaluation Criteria:\nFor each step, assess the **cumulative correctness** of the entire solution process up to and including the current step. Consider:\n\n1. **Perceptual Correctness**: Are all visual elements (shapes, colors, positions, quantities, spatial relationships) accurately identified and interpreted from the image across ALL steps? Does each step correctly reference what is actually shown in the image?\n\n2. **Reasoning Consistency**: Is the logical flow coherent? Do all inferences, calculations, and conclusions follow validly from the visual information in the image and previous steps? Are there any contradictions with earlier steps or the image content?\n\n3. **Image Fidelity**: Every claim, measurement, or observation must be verifiable against the provided image. Steps that assume information not visible in the image or misinterpret visible information must be marked as incorrect.\n\n## Response Format:\n- Respond **"+"** if BOTH visual elements AND reasoning are correct across the entire solution process up to this step, with all interpretations faithful to the image.\n- Respond **"-"** if EITHER visual elements OR reasoning contains any error in the current step or any previous step, or if any step deviates from what is shown in the image.\n\n**Important**: \n- Evaluate holistically - an error in any previous step affects the correctness of all subsequent steps.\n- Provide only "+" or "-" without explanations.\n- A step can only be marked "+" if the entire solution chain remains valid and grounded in the image.',
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "text": "<image>"},
            {
                "type": "text",
                "text": "### Question:\nThe puzzle you will receive is presented in a standard Raven's Progressive Matrices format: a 3-by-3 matrix of related images, with the bottom-right cell (the ninth tile) missing. There are eight possible answer choices provided separately, and your task is to decide which of those eight images correctly completes the 3-by-3 matrix pattern.\n\n### Solution Process:\nThe Raven's Progressive Matrix shows a 3x3 grid with the bottom right tile missing (third row, third column).\n\n",
            },
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "+"}]},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Each cell contains three geometric shapes, with variations in shape (triangle, pentagon, circle, hexagon, square, diamond), size, color (white, grey, black), and orientation.\n\n",
            }
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "+"}]},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "First row: \n- First tile: triangle, pentagon, hexagon. All are differently sized, two are black outline, one filled grey.\n- Second tile: same three shapes as first, but larger and positions changed. \n- Third tile: again same three shapes, further changed positions and orientation, all unfilled.\n\n",
            }
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "-"}]},
]


# Generate a random image
random_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
example_pil_image = Image.fromarray(random_array)


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


# Replace the <image> token with the example image
messages = find_and_replace_image_token(messages, example_pil_image)

print("replaced messages:", messages)

messages_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print("replaced messages_text:", messages_text)

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[messages_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print("inputs:", inputs)
# exit()
# Testing that the image token is replaced with the example image works

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful assistant that can answer questions and help with tasks."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "text": "<image>"},
            {"type": "text", "text": "What do you see in this image."}
        ]
    }
]

messages = find_and_replace_image_token(messages, example_pil_image)
print("messages after replace, should see 3 keys for image:", messages)
messages_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Query messages_text:", messages_text)

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[messages_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print("inputs:", inputs)

inputs = inputs.to("cuda")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)