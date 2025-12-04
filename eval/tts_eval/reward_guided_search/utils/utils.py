from typing import Dict, List, Tuple
from PIL import Image
from io import BytesIO
import base64
import re

def sample_to_images_list(
    sample: Dict,
) -> List[
    Image.Image
]:  # only covers formats for mathvista, mmmu, pls check if other datasets are used for evaluation
    if (
        "decoded_image" in sample.keys()
    ):  # specifically for mathvista dataset, which is strictly single image ONLY
        try:
            return [sample["decoded_image"].copy().convert("RGB")]
        except Exception as e:
            print(f"Error opening image for sample {sample}: {e}")
            return []
    
    max_visual_count = 16
    images_list = []
    for i in range(max_visual_count):
        if f"image_{i}" in sample:
            image = sample[f"image_{i}"]
            if image is None:
                continue  # Skip this image if it's None
            if isinstance(image, Image.Image):
                images_list.append(image.copy().convert("RGB")) # standardize for downstream usecases
            else:
                try:
                    # If the image is not already a PIL Image, try to open it if column image_{i} is local file path
                    images_list.append(Image.open(image).convert("RGB"))
                except Exception as e:
                    print(f"Error opening image_{i} for sample {sample}: {e}")
                    # Optionally, you can add a placeholder image or just continue
                    continue

    return images_list

def convert_images_to_base64(
    images_list: List[Image.Image],
    max_pixels: int = 5120*28*28, # configs from QWen-2.5-VL MMMU Eval Code: https://github.com/QwenLM/Qwen2.5-VL/blob/main/evaluation/mmmu/run_mmmu.py
    # max_pixels: int = 50176, # from training
    min_pixels: int = 1281*28*28,
    # min_pixels: int = 784, # from training
) -> List[str]:
    processed_visuals = []
    for image in images_list:
        if isinstance(image, Image.Image):  # Handle both single and multiple images
            base64_image = image.convert("RGB")
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": max_pixels, "min_pixels": min_pixels})
    return processed_visuals

def prepare_question_array_with_base64_image_strings(
    question: str, image_data_base64_list: List[str], interleave_image_tokens: bool
) -> Tuple[List[Dict], List[str]]:
    """
    Prepare the question in the format of a list of dictionaries, each containing a role and content.
    The content is a list of dictionaries, each containing a type and text.
    The type can be "text" or "image".
    The text is the text of the question.
    The image is the base64 encoded image string.

    Example:
    [
     {"type": "text", "text": "What is shown in "},
     {"type": "image", "data": "img1_base64"},
     {"type": "text", "text": "? Explain "},
     {"type": "image", "data": "img2_base64"},
     {"type": "text", "text": "."}
   ]
    """
    user_messages_array = []
    image_placeholders = re.findall(r"<image \d+>", question)

    if interleave_image_tokens is False: # found out for MathVista, image tokens are found in the options after appended to prompt  
    # if interleave_image_tokens is False or len(image_placeholders) == 0:  # for mathvista; and cases when there are no <image x> tokens in the question
        user_messages_array.append(  # TODO: Currently assumes Single Image only for PuzzleVQA because it is so, but to check MathVista if multiple images appear (in question, in options)
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{image_data_base64_list[0]}"},
                    {"type": "text", "text": question},
                ],  # for non-interlave questions, there will not be <image x> tokens in the question
            }
        )
        return user_messages_array, image_data_base64_list # we are assuming the image only appears once (for MathVista) - to revisit
    
    else:  # currently support find <image x> in the context, not <image_x> or <image> or any other format 
        output_image_data_base64_list = []
        image_placeholders = re.findall(r"<image \d+>", question)
        content_parts = []
        text_parts = re.split(r"<image \d+>", question)
        if text_parts[0]: # if text appears first, append it. Otherwise append first image first
            content_parts.append({"type": "text", "text": text_parts[0]})

        for i, placeholder in enumerate(image_placeholders):
            img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1 # converts <image x> to x-1 as in <image 1> to 0 (index)
            image_idx = min(img_idx, len(image_data_base64_list) - 1) if image_data_base64_list else 0 # index to reference the correspoding image in image_data_base64_list
            if image_data_base64_list and image_idx < len(image_data_base64_list):
                content_parts.append({"type": "image", "image": f"data:image/jpg;base64,{image_data_base64_list[image_idx]}"})
                output_image_data_base64_list.append(image_data_base64_list[image_idx])
            if i + 1 < len(text_parts) and text_parts[i + 1]:
                content_parts.append({"type": "text", "text": text_parts[i + 1]})

        user_messages_array.append(
            {
                "role": "user",
                "content": content_parts,
            }
        )


    return user_messages_array, output_image_data_base64_list

def prepare_question_array_with_base64_image_strings_mathvision(
    question: str, image_data_base64_list: List[str], interleave_image_tokens: bool
) -> Tuple[List[Dict], List[str]]:
    """
    Prepare the question in the format of a list of dictionaries, each containing a role and content.
    The content is a list of dictionaries, each containing a type and text.
    The type can be "text" or "image".
    The text is the text of the question.
    The image is the base64 encoded image string.

    Example:
    [
     {"type": "text", "text": "What is shown in "},
     {"type": "image", "data": "img1_base64"},
     {"type": "text", "text": "? Explain "},
     {"type": "image", "data": "img2_base64"},
     {"type": "text", "text": "."}
   ]
    """
    user_messages_array = []
    # image_placeholders = re.findall(r"<image \d+>", question)

    if interleave_image_tokens is False: # found out for MathVista, image tokens are found in the options after appended to prompt  
    # if interleave_image_tokens is False or len(image_placeholders) == 0:  # for mathvista; and cases when there are no <image x> tokens in the question
        user_messages_array.append(  # TODO: Currently assumes Single Image only for PuzzleVQA because it is so, but to check MathVista if multiple images appear (in question, in options)
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{image_data_base64_list[0]}"},
                    {"type": "text", "text": question.strip()},
                ],  # for non-interlave questions, there will not be <image x> tokens in the question
            }
        )
        return user_messages_array, image_data_base64_list # we are assuming the image only appears once (for MathVista) - to revisit
    
    else:  # currently support find <image x> in the context, not <image_x> or <image> or any other format 
        # output_image_data_base64_list = []
        # image_placeholders = re.findall(r"<image \d+>", question)
        # content_parts = []
        # text_parts = re.split(r"<image \d+>", question)
        # if text_parts[0]: # if text appears first, append it. Otherwise append first image first
        #     content_parts.append({"type": "text", "text": text_parts[0]})

        # for i, placeholder in enumerate(image_placeholders):
        #     img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1 # converts <image x> to x-1 as in <image 1> to 0 (index)
        #     image_idx = min(img_idx, len(image_data_base64_list) - 1) if image_data_base64_list else 0 # index to reference the correspoding image in image_data_base64_list
        #     if image_data_base64_list and image_idx < len(image_data_base64_list):
        #         content_parts.append({"type": "image", "image": f"data:image/jpg;base64,{image_data_base64_list[image_idx]}"})
        #         output_image_data_base64_list.append(image_data_base64_list[image_idx])
        #     if i + 1 < len(text_parts) and text_parts[i + 1]:
        #         content_parts.append({"type": "text", "text": text_parts[i + 1]})

        # user_messages_array.append(
        #     {
        #         "role": "user",
        #         "content": content_parts,
        #     }
        # )
        raise ValueError("Interleaving image tokens not implemented for MathVision")


def prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm(
    question: str, image_data_base64_list: List[str], interleave_image_tokens: bool, model_type: str = "minicpm"
) -> Tuple[List[Dict], List[str]]:
    """
    Prepare the question for InternVL and Mini-CPM models which expect image placeholders as text.
    
    Args:
        question: The question text
        image_data_base64_list: List of base64 encoded images
        interleave_image_tokens: Whether images are interleaved in the question
        model_type: Either "internvl" or "minicpm" to determine placeholder format
    
    Returns:
        Tuple of (user_messages_array, image_data_base64_list)
    """
    user_messages_array = []
    
    if interleave_image_tokens is False:
        # Generate placeholders based on model type
        if model_type == "minicpm":
            placeholders = "".join("(<image>./</image>)\n" for _ in image_data_base64_list)
            # placeholders = "".join(f"image {i}\n" for i in range(1, len(image_data_base64_list) + 1))
            # placeholders = "".join("<image>\n" for _ in range(1, len(image_data_base64_list) + 1))
        elif model_type == "internvl":  # internvl
            placeholders = "\n".join(f"Image-{i}: <image>\n" for i in range(1, len(image_data_base64_list) + 1))
        else:
            raise ValueError(f"Invalid model type set in prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm: {model_type}")
        
        # Combine placeholders with question as single text content
        user_messages_array.append({
            "role": "user",
            "content": f"{placeholders}\n{question}"
        })
        return user_messages_array, image_data_base64_list
    
    else:  # Handle interleaved images
        output_image_data_base64_list = []
        image_placeholders = re.findall(r"<image \d+>", question)
        modified_question = question
        
        for placeholder in image_placeholders:
            img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
            image_idx = min(img_idx, len(image_data_base64_list) - 1) if image_data_base64_list else 0
            
            if image_data_base64_list and image_idx < len(image_data_base64_list):
                output_image_data_base64_list.append(image_data_base64_list[image_idx])
                
                # Replace with model-specific placeholder
                if model_type == "minicpm":
                    img_num = int(re.search(r"<image (\d+)>", placeholder).group(1))
                    replacement = "(<image>./</image>)"
                    # replacement = f"image {img_num}"
                elif model_type == "internvl":  # internvl
                    # Use the original image number from the placeholder
                    img_num = int(re.search(r"<image (\d+)>", placeholder).group(1))
                    replacement = f"Image-{img_num}: <image>"
                else:
                    raise ValueError(f"Invalid model type set in prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm: {model_type}")
                
                modified_question = modified_question.replace(placeholder, replacement, 1)
        
        user_messages_array.append({
            "role": "user",
            "content": modified_question
        })
        
        return user_messages_array, output_image_data_base64_list

def extract_boxed(s):
    """
    Extract content from \boxed{...} expressions, handling corrupted escape sequences.
    
    Behavior:
    - Returns the LAST match content when multiple matches exist (preserving whitespace)
    - Returns None if no matches found
    - Handles nested braces correctly with depth tracking
    - Handles corrupted \b -> \x08 in LLM outputs
    
    This takes the last match because in mathematical problems, the final \boxed{}
    typically contains the answer after showing work in earlier \boxed{} expressions.
    """
    results = []
    i = 0
    
    # Define patterns to search for
    patterns = [
        r'\boxed{',  # Standard pattern
        '\x08oxed{'   # Corrupted \b -> \x08 pattern (from LLM outputs)
    ]
    
    while i < len(s):
        # Find the nearest occurrence of any pattern
        nearest_start = -1
        matched_pattern = None
        
        for pattern in patterns:
            pos = s.find(pattern, i)
            if pos != -1 and (nearest_start == -1 or pos < nearest_start):
                nearest_start = pos
                matched_pattern = pattern
        
        if nearest_start == -1:
            break
            
        # Advance past the matched pattern
        j = nearest_start + len(matched_pattern)
        depth = 1
        
        # Track brace depth to handle nested braces
        while j < len(s) and depth > 0:
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
            j += 1
            
        if depth == 0:
            # Extract content between the opening '{' and the matching '}'
            content = s[nearest_start + len(matched_pattern) : j - 1]
            results.append(content)
            i = j
        else:
            # Unbalanced braces: skip this occurrence and continue
            i = nearest_start + 1
    
    # Return the last match if any matches found (typically the final answer)
    if len(results) > 0:
        return results[-1]
    else:
        return None