# Instructions to set up virtual environment for evaluation
- We recommend using uv to create a virtual environment and following the sequence below to install the dependencies. The sequence of installations are important as it ensures compatibility of dependencies.
    - We had the most success with uv's standalone installer (not the pip version), and recommend this for this project as well.
    - ```curl -LsSf https://astral.sh/uv/install.sh | sh```

# Set Up Virtual Environment
- We recommend the following process to sync dependencies using our requirements.txt file. You will need to install flash-attn manually after syncing.
```bash
module load CUDA/12.6.0
uv venv -p 3.12 --seed
source .venv/bin/activate
uv pip sync requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
uv add flash-attn --no-build-isolation
```

- Alternatively, do the following for conda installation
```bash
conda create -n eval python=3.12
conda activate eval
pip install -r frozen_eval_requirements.txt
pip install "flash-attn<=2.8" --no-build-isolation
```

## Testing Environment
- To test if the environment is set up correctly, you can run the following command:
```bash
cd base_model_eval/vLLM_evaluation_code
./test_run_base_model_eval.sh
```

- Set an OpenAI API key in a .env file in the parent directory, as this evaluation requires a LLM Judge.
    - or ```export OPENAI_API_KEY=<your_api_key>``` before running the bash script
- This will run the evaluation script for a Qwen2.5-VL-3B-Instruct base model.
- If everything is installed correctly, you should get a score of 5/8 (62.5%) on the development set of MathVista evaluation (3/8 is the commmon score without flash-attn)
- If you get a score of less than 6/8, please check that flash-attn is installed correctly, because you will get accuracy degradation without it

# Running Greedy and Non-Greedy Search Evaluation

- To run the non-greedy search (one-shot) search evaluation, you need to comment out the following line in the vllm_bon_greedy_search_no_template.py file:
```python
stop=['<|im_end|>', '<|endoftext|>'], # TODO: Important when doing Greedy, not when Non-Greedy, check this again when running
include_stop_str_in_output=True,
```

- To run the greedy search evaluation, you should include the following lines in the vllm_bon_greedy_search_no_template.py file for the corresponding policy model:
```python
stop=[eos_token, NEWLINE_STEP_SEPERATOR], # TODO: Important when doing Greedy, not when Non-Greedy, check this again when running
include_stop_str_in_output=True,
```
This ensures reward is calculated at the step-by-step level.

Then run ```./run_bon_greedy_search_no_template.sh``` to run the evaluation.

or ```./vllm_lazy_greedy_search_no_template.sh``` if you have a PBS cluster.

# Minimal evaluation on your own dataset

If you just want to score step-by-step traces from your own data with a VL-PRM checkpoint, use the lightweight script below. It only requires a single GPU and does not depend on vLLM. When no steps are provided, it rolls out step candidates from a standard Qwen2.5-VL-7B policy and lets the VL-PRM pick the best one at every hop.

1. Format your dataset as a JSON list of objects:

```json
[
  {
    "image": "/path/to/image.jpg",
    "question": "What is shown in the image?",
    "steps": ["First reasoning step", "Second reasoning step"]  // Optional. If omitted, steps are generated.
  }
]
```

2. Run the minimal evaluator:

```bash
python -m eval.minimal_prm_evaluation \
  --model-path ob11/Qwen-VL-PRM-3B \
  --policy-model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --data-path /path/to/my_dataset.json \
  --output-path /tmp/prm_scores.json
```

The output JSON mirrors the input and adds:

- `steps`: the scored step trajectory (either provided or generated)
- `step_scores`: the PRM score assigned after each chosen step
- `average_step_score`: the mean PRM score in `[0, 1]`

# Running LLM Judge Evaluation for MathVision
- MathVision involves commonly answering questions with LaTex involved, hence we need to support SymPy answer validators for accurate output evaluation.

- Run ```python mathvision_helper_functions.py <results_file.json>``` to run this evaluation manually and inspect the results to verify that the outputs are evaluated correctly.
