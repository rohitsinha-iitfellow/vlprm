# Rolling Out and Step-Wise Verification

1. Transfer dataset images and their corresponding JSON paths containing absolute image paths to input_data_files/{dataset_name} 
    - refer to VQAv2 example provided
2. Specify the Azure endpoints, deployment and config in ```rollout.py```
    - Edit the parse_answer and check_answer functions in ```rollout.py``` to match the answer format of the dataset
        - parse_answer is configured by setting scoring_mode, depending on the format of the answer. See parsing functions in ```accuracy_reward.py```
        - check_answer is configured by setting prompt_format_version, depending on the data type of the answer (e.g. integer only, string only exact match, LLM as judge string match). Trace accuracy_reward.py to see how different modes are configured.
    - run ```./run_rollout.sh (no parameters)``` to generate rollouts
3. Transfer rollouts outputs to rollout_data_generation/llm_judge_verification/rollout_output_data_files/{dataset_name}/{split_name}
    - we recommend to keep the raw rollout outputs untouched, and make a copy to process in the next step in case for processing errors that might "corrupt" the raw rollout outputs
4. Prepare batches using ```merge_and_map_batch.py```, then run ```./run_batch_processor.sh``` in rollout_data_generation/llm_judge_verification to verify the rollouts, with the required parameters expected in ```batch_processor.py```
    - Usage: ```python merge_and_map_batch.py --split="vqav2_4k" --model="gpt-4.1-nano-3"```
        - Azure Batch API expects the model to be specified in the JSONL file along with prompt*
        - ```merge_and_map_batch.py``` prepares input batch files that fulfil the Azure Batch API requirements e.g. max tokens per batch, max file size, lines per batch. (see Azure Batch API documentation for more details)
    - Usage: ```AZURE_API_KEY='your-key' ./run_batch_processor.sh [screen_session_name] [batch_start_index] [batch_end_index] [split] [azure_endpoint] [check_interval] [model]```

# Known Issues
- For datasets like AI2D, InfoVQA, PuzzleTest, Azure will return content filter errors and are unavoidable, unfortunately we cannot use such samples for training. We drop these rollouts in the training data.

- If content filter error thrown, API will return response "Content filter error", which results in a "fail to parse" error, and the prompt will be skipped, with error "fail to parse" in the log.

    Example: infovqa_run1_open_ans_9K_v1_subset_raven_rollouts_1289_1610_streaming has 8 "failed to parse" errors, which are all content filter errors, as can see in log, "failed to parse: 4/60 rollouts" twice. 

- For the scale of rollout generation and batch verification we designed, we recommend utilizing Azure to maximize the throughput limits that OpenAI native API cannot provide. You effectively get ~22x the throughput with Azure compared to being constrained to the limits of a single OpenAI API endpoint.
    - We deployed the maximum of 22 API endpoints and batch endpoints (1 for every available Azure region with GPT-4.1 and o4-mini models available) throughout the project to maximize the throughput limits. The core bottleneck is the Tokens Per Minute (TPM) limit of the API endpoints, which you can adjust the hyperparameters accordingly to fit the rate limits.