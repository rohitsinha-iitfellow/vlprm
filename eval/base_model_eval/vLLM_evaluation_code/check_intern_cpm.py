from transformers import AutoTokenizer, AutoModel, GenerationConfig
from PIL import Image
from typing import List
import torch
import sys
sys.path.append('/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation')
from reward_guided_search.utils import prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm
from reward_guided_search.internvl_image_processing_utils import load_image
QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
]

models = [
    # "OpenGVLab/InternVL2_5-8B",
    "openbmb/MiniCPM-V-2_6",
]

for model in models:
    if model == "openbmb/MiniCPM-V-2_6": # use transformers=4.55.2
        print("-"*100)
        print(f"model: {model}")
        model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
        print(f"tokenizer.name_or_path: {tokenizer.name_or_path}")
        image1 = Image.open('test_images/duck.jpg').convert('RGB')
        image2 = Image.open('test_images/lion.jpg').convert('RGB')
        question = 'Compare image 1 and image 2, tell me about the differences between image 1 and image 2.'

        msgs = [{'role': 'user', 'content': [image1, image2, question]}]

        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            generation_config=GenerationConfig(num_beams=3, temperature=0.0, do_sample=False, max_new_tokens=16384)
        )
        print(answer)
    
    elif model == "OpenGVLab/InternVL2_5-8B": # use transformers=4.37.2
        print("-"*100)
        print(f"model: {model}")
        path = 'OpenGVLab/InternVL2_5-8B'
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.95)

        pixel_values1 = load_image('test_images/duck.jpg', max_num=12).to(torch.bfloat16).cuda()
        pixel_values2 = load_image('test_images/lion.jpg', max_num=12).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

        question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
        # only works with Transformers=4.37.2
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        print(f'User:\n{question}\nAssistant:\n{response}')