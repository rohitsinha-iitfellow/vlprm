import json
import torch
import torch.nn.functional as F
import os
from copy import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
from qwen_vl_utils import process_vision_info
from typing import List

from prompts import PRM_SYSTEM_PROMPT_NORMAL_TOK_V2, PRM_SYSTEM_PROMPT_NORMAL_TOK
from utils.utils import prepare_question_array_with_base64_image_strings
from utils.logger import log_info

# use for models with updated tokens
# POSITIVE_TOKEN = "<+>"
# NEGATIVE_TOKEN = "<->"

# use for base model with no updated tokens
POSITIVE_TOKEN = "+"
NEGATIVE_TOKEN = "-"


class VisualPRM:
    def __init__(self, model_path, model_init_kwargs=None):
        log_info(f"Loading model from {model_path}")

        if model_init_kwargs is None:
            model_init_kwargs = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
            }

        # Load corresponding reward model class based on the config.json file "architecture" key.
        # try:
        #     with open(os.path.join(model_path, "config.json"), "r") as f:
        #         config = json.load(f)
        #     architecture = config["architectures"][0]
        # except FileNotFoundError:
        #     raise FileNotFoundError(f"config.json not found in {model_path}")
        # except json.JSONDecodeError as e:
        #     raise ValueError(f"Invalid JSON in config.json: {e}")
        # except KeyError:
        #     raise KeyError("'architecture' key not found in config.json")

        # if architecture == "Qwen2_5_VLForConditionalGeneration":
        #     self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #         model_path, **model_init_kwargs
        #     )
        # elif architecture == "LlavaNextForConditionalGeneration":
        #     self.model = LlavaNextForConditionalGeneration.from_pretrained(
        #         model_path, **model_init_kwargs
        #     )
        # else:
        #     raise ValueError(
        #         f"Unsupported architecture in VisualPRM class: {architecture}"
        #     )
        # self.model_architecture = architecture
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **model_init_kwargs
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        # if self.model_architecture == "Qwen2_5_VLForConditionalGeneration":
        #     self.processor = AutoProcessor.from_pretrained(
        #         model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        #     )
        # elif self.model_architecture == "LlavaNextForConditionalGeneration":
        #     # Get patch_size from model config or use default
        #     patch_size = config.get("vision_config", {}).get("patch_size", 14)
        #     self.processor = LlavaNextProcessor.from_pretrained(
        #         model_path, patch_size=patch_size
        #     )
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        log_info("VisualPRM loaded successfully")
        self.pos_token_id = self.tokenizer.encode(POSITIVE_TOKEN)[0]
        self.neg_token_id = self.tokenizer.encode(NEGATIVE_TOKEN)[0]
        self.system_prompt = PRM_SYSTEM_PROMPT_NORMAL_TOK_V2 # TODO: Set based on positive/negative token used

    def inference_single(
        self, sample_input_messages_array_including_images_interweaved, logging=False
    ):
        # log_info(
        #     f"Starting inference with input: {sample_input_messages_array_including_images_interweaved}"
        # )

        text = self.processor.apply_chat_template(
            sample_input_messages_array_including_images_interweaved,
            tokenize=False,
            add_generation_prompt=True,
        )
        log_info(f"Reward model Text: {text}")

        # log_info("*" * 100)
        # log_info(sample_input_messages_array_including_images_interweaved)
        # log_info("*" * 100)
        # exit()
        
        image_inputs, video_inputs = process_vision_info(
            sample_input_messages_array_including_images_interweaved
        )

        log_info(f"CHECK: Len of Image inputs output from process_vision_info (should match len of base64 image list): {len(image_inputs)}")
        log_info(
            f"DEBUG: Image should be PIL Image as input to processor: {[type(img) if img else 'None' for img in image_inputs]}"
        )

        # if self.model_architecture == "Qwen2_5_VLForConditionalGeneration":
        #     message_ids = self.processor(
        #         text=[text],
        #         images=image_inputs,
        #         videos=video_inputs,
        #         padding=True,
        #         return_tensors="pt",
        #     ).to(self.model.device)
        # elif self.model_architecture == "LlavaNextForConditionalGeneration":
        #     message_ids = self.processor(
        #         images=image_inputs, text=text, return_tensors="pt"
        #     ).to(self.model.device)
        # else:
        #     raise ValueError(
        #         f"Unsupported architecture in VisualPRM class: {self.model_architecture}"
        #     )

        message_ids = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)        

        # message_ids = self.tokenizer.apply_chat_template(
        #     sample_input_messages_array_including_images_interweaved,
        #     tokenize=True,
        #     return_dict=True,
        #     return_tensors='pt'
        #     ).to(self.model.device)

        log_info(f"Tokenized message_ids shape: {message_ids['input_ids'].shape}")
        # log_info(f"Input tokens: {self.tokenizer.decode(message_ids['input_ids'][0])}")

        with torch.no_grad():
            outputs = self.model(
                **message_ids
            )  # [batch_size, seq_len, vocab_size] - double check size

        log_info(f"Model outputs logits shape: {outputs.logits.shape}")
        log_info(f"Model outputs logits device: {outputs.logits.device}")

        allowed_token_ids = torch.tensor([self.pos_token_id, self.neg_token_id], device=outputs.logits.device)  # shape: (2,)
        log_info(
            f"Allowed token IDs: {allowed_token_ids} (+ token: {self.pos_token_id}, - token: {self.neg_token_id})"
        )

        last_position_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        log_info(f"Last position logits shape: {last_position_logits.shape}")
        log_info(
            f"Last position logits for + and - tokens: {last_position_logits[:, allowed_token_ids]}"
        )

        masked_logits = last_position_logits[:, allowed_token_ids]  # [batch_size, 2]
        log_info(f"Masked logits shape: {masked_logits.shape}")
        log_info(f"Masked logits values: {masked_logits}")

        probs_pos_neg = F.softmax(masked_logits, dim=-1)
        log_info(f"Probabilities [pos, neg]: {probs_pos_neg}")

        predicted_indices = masked_logits.argmax(dim=-1)
        predicted_tokens = allowed_token_ids[predicted_indices]
        log_info(f"Predicted indices: {predicted_indices}")
        log_info(f"Predicted token IDs: {predicted_tokens}")

        decoded_tokens = [self.tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in predicted_tokens]
        log_info(f"Decoded predicted tokens: {decoded_tokens}")

        if logging:
            log_info(f"Decoded Labels (either + or -): {decoded_tokens}")

        positive_prob = probs_pos_neg[0][0].cpu().item()
        negative_prob = probs_pos_neg[0][1].cpu().item()
        
        if NEGATIVE_TOKEN in decoded_tokens:
            log_info("Negative prediction detected")
            reward_score = -1
        else:
            input_length = message_ids['input_ids'].shape[1]  # Total input tokens
            reward_score = (positive_prob ** 0.3) / (input_length ** 0.6)
            log_info(f"Normalized reward score: {reward_score}")

        result = {
            'prediction': 'negative' if NEGATIVE_TOKEN in decoded_tokens else 'positive',
            'positive_prob': positive_prob,
            'negative_prob': negative_prob,
            'reward_score': reward_score
        }
        
        log_info(f"Returning result: {result}")
        return result

    def get_reward(
        self,
        question: str,
        previous_steps: List[str],
        now_step: str,
        base64_image_list: List[str],
        interleave_image_tokens: bool = False,
    ) -> float:
        """
        Get reward score for a reasoning step given the question, previous steps, and current step.
        """

        messages_array_to_generate_reward = [
            # {"role": "system", "content": self.system_prompt}
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]  # mirrors training process


        if len(previous_steps) > 0:
            log_info(f"Previous steps > 0: {previous_steps}")
            for i, step in enumerate(previous_steps):
                if i == 0: # the first step requires to include question and set up solution process
                    standard_first_user_message = (
                        f"### Question:\n{question}\n\n### Solution Process:\n{step}"
                    )

                    standard_first_question_in_messages_array_format, standard_first_question_corresponding_image_data_base64_list = (
                        prepare_question_array_with_base64_image_strings(
                            standard_first_user_message,
                            base64_image_list,
                            interleave_image_tokens=interleave_image_tokens,
                        )
                    )

                    log_info(f"in VisualPRM length of standard_first_question_corresponding_image_data_base64_list: {len(standard_first_question_corresponding_image_data_base64_list)}")

                    messages_array_to_generate_reward += (
                        standard_first_question_in_messages_array_format
                    )
                    messages_array_to_generate_reward.append(
                        # {"role": "assistant", "content": POSITIVE_TOKEN}
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": POSITIVE_TOKEN}],
                        }
                    )
                else: # subsequent steps after first step, we can just paste the previous step
                    messages_array_to_generate_reward.append(
                        # {"role": "user", "content": step}
                        {"role": "user", "content": [{"type": "text", "text": step}]}
                    ) 
                    messages_array_to_generate_reward.append(
                        # {"role": "assistant", "content": POSITIVE_TOKEN}
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": POSITIVE_TOKEN}],
                        }
                    )
            
            messages_array_to_generate_reward.append(
                # {"role": "user", "content": now_step}
                {"role": "user", "content": [{"type": "text", "text": now_step}]}
            ) # set up for reward model to generate reward for the current step
                 
            # log_info(
            #     f"messages_array_to_generate_reward with multiple steps after step 1: {messages_array_to_generate_reward}"
            # )
        else:
            standard_first_user_message = (
            f"### Question:\n{question}\n\n### Solution Process:\n{now_step}"
        ) # reached only first step

            standard_first_question_in_messages_array_format, standard_first_question_corresponding_image_data_base64_list = (
                prepare_question_array_with_base64_image_strings(
                    standard_first_user_message,
                    base64_image_list,
                    interleave_image_tokens=interleave_image_tokens,
                )
            )

            log_info(f"in VisualPRM length of standard_first_question_corresponding_image_data_base64_list: {len(standard_first_question_corresponding_image_data_base64_list)}")

            messages_array_to_generate_reward += (
                standard_first_question_in_messages_array_format
            )
            

        # log_info(f"Reward model messages array: {messages_array_to_generate_reward}")
        log_info(f"Base64 image list length: {len(base64_image_list)}")
        result = self.inference_single(messages_array_to_generate_reward)

        return result