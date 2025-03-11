
# Exit immediately if a command exits with a non-zero status
set -e

FORMAT_EPISODE_PATH="../data/formatted_episodes/taskeval_negoeval/actual_movement.json"
INTENT_RESPONSE_PATH="../data/intent_response/taskeval_negoeval/actual_movement.jsonl"
INTENT_VALUE_PATH="../data/intent_value/taskeval_negoeval/actual_movement.json"

source ~/.bashrc

# echo "Activating sotopia environment..."
conda activate sotopia

# env_tag: coop_with_actual
# echo "Running add_actual_intent_episode.py ..."
CUDA_VISIBLE_DEVICES=0 python ../get_actual_intent_episode.py --env_tag finetune_format_with_previous --tgt_path=$FORMAT_EPISODE_PATH

echo "Running intent_prediction.py ..."
CUDA_VISIBLE_DEVICES=0 python ../intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=0 --split_end None
# --split_begin=0 --split_end=200
# CUDA_VISIBLE_DEVICES=0 python intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=200 --split_end None
# python intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=900 --split_end=1350
# python intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=1350 --split_end None

# echo "Activating diplomacy_cicero environment..."
conda activate diplomacy_cicero

# echo "Running intent_value_evaluate.py ..."
python ../intent_value_evaluate.py --task_eval --move --res_path=$INTENT_RESPONSE_PATH --tgt_path=$INTENT_VALUE_PATH

echo "Script execution completed."
