"""Minimal, self-contained VL-PRM evaluation example.

This script scores step-by-step reasoning traces with a Vision-Language
Process Reward Model (VL-PRM) on a custom JSON dataset. It can either
consume pre-written steps or draft candidate steps from a Qwen2.5-VL
policy model and let the VL-PRM choose the best one at each hop.

Dataset format (list of examples):
[
  {
    "image": "path/to/question.jpg",            # Required
    "question": "What is shown in the image?",  # Required
    "steps": ["Step 1", "Step 2"]               # Optional list of strings
  }
]

Usage:
    python -m eval.minimal_prm_evaluation \
        --model-path ob11/Qwen-VL-PRM-3B \
        --policy-model-path Qwen/Qwen2.5-VL-7B-Instruct \
        --data-path /path/to/my_dataset.json \
        --output-path /tmp/prm_scores.json

The script only relies on a single GPU for inference. It produces a JSON
list of scored examples with an ``average_step_score`` field in ``[0, 1]``.
"""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRM_SYSTEM_PROMPT = """**You are a process supervision model for visual reasoning tasks. You will receive an image and an image-based problem statement, followed by solution steps to evaluate.**

First round: problem statement and first solution step.
Subsequent rounds: one new step per round.

Assess the cumulative correctness of the entire solution up to each step.

## Evaluation Criteria:

1. **Visual Accuracy**: Are visual elements from the image correctly identified (shapes, colors, positions, quantities, spatial relationships)?

2. **Logical Validity**: Do all inferences and calculations follow correctly from the image and previous steps?

## Response:
- **"+"** if correct up to this step
- **"-"** if any error exists up to this step

Only respond with "+" or "-". No explanations.

An error in any step invalidates all subsequent steps."""

POLICY_SYSTEM_PROMPT = """You are a helpful visual reasoning assistant. Carefully read the question, look at the image, and produce the next concise reasoning step without repeating previous steps. Stop once you have enough information to deliver a final answer."""


def build_transform(input_size: int) -> T.Compose:
    """Standard image preprocessing used by VL-PRM."""

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: Iterable[tuple[int, int]], width: int, height: int, image_size: int
) -> tuple[int, int]:
    """Pick a target ratio that matches the original image best."""

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448) -> list[Image.Image]:
    """Split the image into square patches that the PRM expects."""

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    return processed_images


def load_image_from_base64(image_base64: str, input_size: int = 448, max_num: int = 12) -> tuple[torch.Tensor, list[int]]:
    """Decode a base64 image string into normalized pixel tensors."""

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, [pixel_values.shape[0]]


def encode_image_to_base64(image_path: str | Path) -> str:
    """Encode an image file into a base64 JPEG string."""

    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class QwenPolicyModel:
    """Minimal Qwen2.5-VL policy that proposes next steps."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str | None = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        ).eval().to(self.device)
        self.max_new_tokens = max_new_tokens

    def _build_messages(self, image: Image.Image, question: str, steps: Sequence[str]) -> list[dict]:
        step_history = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
        user_instruction = [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    f"Question: {question}\n\n"
                    "You are continuing the reasoning one step at a time."
                    + (f"\nExisting steps:\n{step_history}" if step_history else "")
                    + "\nProvide ONLY the next short reasoning step."
                ),
            },
        ]

        return [
            {"role": "system", "content": [{"type": "text", "text": POLICY_SYSTEM_PROMPT}]},
            {"role": "user", "content": user_instruction},
        ]

    def generate_candidates(
        self,
        image: Image.Image,
        question: str,
        steps: Sequence[str],
        num_candidates: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        messages = self._build_messages(image, question, steps)
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt], images=[image], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_candidates,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        prompt_length = inputs["input_ids"].shape[1]
        candidate_token_ids = outputs[:, prompt_length:]
        candidates = self.processor.batch_decode(candidate_token_ids, skip_special_tokens=True)
        return [c.strip() for c in candidates if c.strip()]


class MinimalVisualPRM:
    """Lightweight PRM wrapper used purely for inference."""

    def __init__(self, model_path: str, device: str | None = None, dtype: str = "bfloat16") -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = (
            AutoModel.from_pretrained(
                model_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch_dtype
            )
            .eval()
            .to(self.device)
        )

    def score_steps(self, image_b64: str, question: str, steps: Sequence[str]) -> float:
        """Return the average step reward in ``[0, 1]`` for one example."""

        pixel_values, num_patches_list = load_image_from_base64(image_b64)
        pixel_values = pixel_values.to(self.model.device, dtype=self.model.dtype)

        if pixel_values is not None and "<image>" not in question:
            question = "<image>\n" * len(num_patches_list) + question

        img_start = "<img>"
        img_end = "</img>"
        img_context = "<IMG_CONTEXT>"
        placeholder = "+"

        str2score = {"+": 1, "-": 0}
        candidate_tokens = []
        candidate_weights = []
        for token, weight in str2score.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            candidate_tokens.append(token_id)
            candidate_weights.append(weight)

        conversation_content = []
        for step_idx, step in enumerate(steps):
            if step_idx == 0:
                step_content = f"### Question:\n{question}\n\n### Solution Process:\n{step}"
            else:
                step_content = step
            conversation_content.append((step_content, placeholder))

        query_parts = [f"<|im_start|>system\n{PRM_SYSTEM_PROMPT}<|im_end|>"]
        for step_content, ph in conversation_content:
            query_parts.append(f"<|im_start|>user\n{step_content}<|im_end|>")
            query_parts.append(f"<|im_start|>assistant\n{ph}<|im_end|>")

        query = "\n".join(query_parts)

        num_image_token = 2
        for num_patches in num_patches_list:
            image_tokens = img_start + img_context * num_image_token * num_patches + img_end
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.model.device)
        attention_mask = model_inputs["attention_mask"].to(self.model.device)
        image_flags = torch.tensor([True] * pixel_values.shape[0], dtype=torch.long, device=self.model.device)

        placeholder_positions = []
        input_ids_list = input_ids[0].tolist()
        placeholder_token_id = self.tokenizer.convert_tokens_to_ids(placeholder)
        im_start_assistant_tokens = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        im_end_tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

        for i, token_id in enumerate(input_ids_list):
            if token_id != placeholder_token_id:
                continue
            has_prefix = i >= len(im_start_assistant_tokens) and input_ids_list[
                i - len(im_start_assistant_tokens) : i
            ] == im_start_assistant_tokens
            has_suffix = (i + len(im_end_tokens)) < len(input_ids_list) and input_ids_list[
                i + 1 : i + 1 + len(im_end_tokens)
            ] == im_end_tokens
            if has_prefix and has_suffix:
                placeholder_positions.append(i)

        with torch.no_grad():
            logits = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
            ).logits

        logits = logits[0][placeholder_positions, :][:, candidate_tokens]
        soft_scores = logits.softmax(dim=-1).tolist()

        scores = []
        for soft_score in soft_scores:
            score = sum(prob * weight for prob, weight in zip(soft_score, candidate_weights))
            scores.append(score)
        return float(sum(scores) / len(scores)) if scores else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VL-PRM scoring on a custom JSON dataset.")
    parser.add_argument("--model-path", required=True, help="Hugging Face model id or local path for the VL-PRM checkpoint.")
    parser.add_argument(
        "--policy-model-path",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen policy model used to draft candidate steps.",
    )
    parser.add_argument("--data-path", required=True, help="Path to the custom dataset JSON (see header for format).")
    parser.add_argument("--output-path", required=True, help="Where to write the scored JSON file.")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype to load weights with (e.g., float16, bfloat16).")
    parser.add_argument("--device", default=None, help="Optional device override (e.g., cuda:0, cpu).")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on processed samples for smoke tests.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum reasoning steps to roll out with the policy.")
    parser.add_argument(
        "--num-candidates", type=int, default=4, help="Number of candidate steps sampled from the policy per step."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the Qwen policy (0 switches to greedy).",
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value for the policy.")
    parser.add_argument(
        "--stop-phrase",
        default="final answer",
        help="If this phrase appears in the selected step, roll out stops early.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Maximum tokens generated for each policy candidate step."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_json(args.data_path)
    if not isinstance(dataset, list):
        raise ValueError("Dataset JSON must be a list of objects with image and question fields.")

    model = MinimalVisualPRM(args.model_path, device=args.device, dtype=args.dtype)
    policy = QwenPolicyModel(
        model_path=args.policy_model_path, device=args.device, dtype=args.dtype, max_new_tokens=args.max_new_tokens
    )
    results: List[dict] = []

    for idx, example in enumerate(dataset):
        if args.max_samples is not None and idx >= args.max_samples:
            break
        missing_fields = {k for k in ("image", "question") if k not in example}
        if missing_fields:
            raise ValueError(f"Example {idx} is missing fields: {missing_fields}")

        image_path = Path(example["image"])
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image_b64 = encode_image_to_base64(image_path)
        question = example["question"]
        # If manual steps are provided, score them directly; otherwise roll out with the policy.
        if "steps" in example and isinstance(example["steps"], list):
            steps = example["steps"]
            step_scores = [model.score_steps(image_b64, question, steps[: i + 1]) for i in range(len(steps))]
        else:
            steps = []
            step_scores = []
            for _ in range(args.max_steps):
                candidates = policy.generate_candidates(
                    image,
                    question,
                    steps,
                    num_candidates=args.num_candidates,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                if not candidates:
                    break

                scored_candidates = [
                    (candidate, model.score_steps(image_b64, question, steps + [candidate])) for candidate in candidates
                ]
                best_step, best_score = max(scored_candidates, key=lambda x: x[1])
                steps.append(best_step)
                step_scores.append(best_score)
                if args.stop_phrase and args.stop_phrase.lower() in best_step.lower():
                    break

        avg_score = float(sum(step_scores) / len(step_scores)) if step_scores else 0.0
        results.append({**example, "steps": steps, "step_scores": step_scores, "average_step_score": avg_score})

    save_json(results, args.output_path)


if __name__ == "__main__":
    main()
