#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Array of DIALOGUE_END_INDEX values
DIALOGUE_END_INDICES=(6 8 10 12)

# Base paths
BASE_FORMAT_EPISODE_PATH="data/formatted_episodes/taskeval_v3/taskeval_1757_llama_fewshot_end_turns"
BASE_INTENT_RESPONSE_PATH="data/intent_response/taskeval_v3/taskeval_1757_llama_fewshot_end_turns"
BASE_INTENT_VALUE_PATH="data/intent_value/taskeval_v3/taskeval_1757_llama_fewshot_end_turns"

# Source bashrc
source ~/.bashrc

for DIALOGUE_END_INDEX in "${DIALOGUE_END_INDICES[@]}"; do
    echo "Processing DIALOGUE_END_INDEX=$DIALOGUE_END_INDEX"

    # Set paths with DIALOGUE_END_INDEX
    FORMAT_EPISODE_PATH="${BASE_FORMAT_EPISODE_PATH}_${DIALOGUE_END_INDEX}.json"
    INTENT_RESPONSE_PATH="${BASE_INTENT_RESPONSE_PATH}_${DIALOGUE_END_INDEX}.jsonl"
    INTENT_VALUE_PATH="${BASE_INTENT_VALUE_PATH}_${DIALOGUE_END_INDEX}.json"

    echo "Activating sotopia environment..."
    conda activate sotopia

    echo "Running get_intent_episode.py ..."
    python get_intent_episode.py --tag taskeval_fewshot --tgt_path="$FORMAT_EPISODE_PATH"

    echo "Running intent_prediction.py ..."
    python intent_prediction.py --res_path="$FORMAT_EPISODE_PATH" --tgt_path="$INTENT_RESPONSE_PATH" --split_begin=0 --split_end None --end_turn="$DIALOGUE_END_INDEX"

    echo "Activating diplomacy_cicero environment..."
    conda activate diplomacy_cicero

    echo "Running intent_value_evaluate.py ..."
    python intent_value_evaluate.py --task_eval --move --res_path="$INTENT_RESPONSE_PATH" --tgt_path="$INTENT_VALUE_PATH"

    echo "Completed processing for DIALOGUE_END_INDEX=$DIALOGUE_END_INDEX"
    echo "----------------------------------------"
done

echo "Script execution completed."