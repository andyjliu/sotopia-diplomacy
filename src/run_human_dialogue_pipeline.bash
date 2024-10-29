
# Exit immediately if a command exits with a non-zero status
set -e

FORMAT_EPISODE_PATH="data/formatted_episodes/taskeval_v6/taskeval_actual_movement_coop_without_flausible.json"
INTENT_RESPONSE_PATH="data/intent_response/taskeval_v6/taskeval_actual_movement_coop_without_flausible.jsonl"
INTENT_VALUE_PATH="data/intent_value/taskeval_v6/taskeval_actual_movement_coop_without_flausible.json"

source ~/.bashrc

# echo "Activating sotopia environment..."
conda activate sotopia

# echo "Running add_actual_intent_episode.py ..."
CUDA_VISIBLE_DEVICES=0 python get_actual_intent_episode.py --env_tag coop_without_flausible_v2 --tgt_path=$FORMAT_EPISODE_PATH

echo "Running intent_prediction.py ..."
CUDA_VISIBLE_DEVICES=0 python intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=0 --split_end None
# python intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=200 --split_end=300
# python intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=900 --split_end=1350
# python intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=1350 --split_end None

# echo "Activating diplomacy_cicero environment..."
conda activate diplomacy_cicero

# echo "Running intent_value_evaluate.py ..."
python intent_value_evaluate.py --task_eval --move --res_path=$INTENT_RESPONSE_PATH --tgt_path=$INTENT_VALUE_PATH

echo "Script execution completed."
