
# Exit immediately if a command exits with a non-zero status
set -e

# Here the parse means the v2
DIALOGUE_END_INDEX=6
FORMAT_EPISODE_PATH="data/formatted_episodes/taskeval_v4_fewshot_plausible/taskeval_llama_70b_parse.json"
INTENT_RESPONSE_PATH="data/intent_response/taskeval_v4_fewshot_plausible/taskeval_llama_70b_parse.jsonl"
INTENT_VALUE_PATH="data/intent_value/taskeval_v4_fewshot_plausible/taskeval_llama_70b_parse.json"

source ~/.bashrc

echo "Activating sotopia environment..."
conda activate sotopia

echo "Running get_intent_episode.py ..."
# python get_intent_episode.py --tag taskeval_fewshot_plausible_parse --tgt_path=$FORMAT_EPISODE_PATH
# python add_actual_intent_episode.py --env_tag te_1757_with_previous --tgt_path=$FORMAT_EPISODE_PATH

echo "Running intent_prediction.py ..."
python intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=0 --split_end None --end_turn 8

echo "Activating diplomacy_cicero environment..."
conda activate diplomacy_cicero

echo "Running intent_value_evaluate.py ..."
python intent_value_evaluate.py --task_eval --move --res_path=$INTENT_RESPONSE_PATH --tgt_path=$INTENT_VALUE_PATH

echo "Script execution completed."



# Test Intent Prediction
# python intent_prediction.py --res_path data/formatted_episodes/taskeval_v3/taskeval_1757_llama_fewshot.json --tgt_path data/intent_response/taskeval_v3/test.jsonl --split_begin=0 --split_end None
