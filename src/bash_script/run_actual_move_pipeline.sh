
# Exit immediately if a command exits with a non-zero status
set -e

FORMAT_EPISODE_PATH="../data/formatted_episodes/taskeval_negoeval_whole_finetune/intent_human_dialogue_cut.json"
INTENT_RESPONSE_PATH="../data/intent_response/taskeval_negoeval_whole_finetune/intent_human_dialogue_cut.jsonl"
INTENT_VALUE_PATH="../data/intent_value/taskeval_negoeval_whole_finetune/intent_human_dialogue_cut.json"

source ~/.bashrc

echo "Activating sotopia environment..."
conda activate sotopia

echo "Running add_actual_intent_episode.py ..."
python ../get_actual_intent_episode.py --env_tag whole_finetune_format_without_diplomacy_background --tgt_path=$FORMAT_EPISODE_PATH

echo "Running intent_prediction.py in parallel on 4 GPUs..."
CUDA_VISIBLE_DEVICES=0 python ../intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=0 --split_end=250 --cut --end_turn=0 &
CUDA_VISIBLE_DEVICES=1 python ../intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=250 --split_end=500 --cut --end_turn=0 &
CUDA_VISIBLE_DEVICES=2 python ../intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=500 --split_end=750 --cut --end_turn=0 &
CUDA_VISIBLE_DEVICES=3 python ../intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=750 --split_end=1000 --cut --end_turn=0 &

# Wait for all background processes to finish
wait

# echo "Activating diplomacy_cicero environment..."
conda activate diplomacy_cicero

# echo "Running intent_value_evaluate.py ..."
python ../intent_value_evaluate.py --task_eval --move --res_path=$INTENT_RESPONSE_PATH --tgt_path=$INTENT_VALUE_PATH

echo "Script execution completed."



# Format Episode
# python get_actual_intent_episode.py --env_tag finetune_format_with_previous --tgt_path="data/formatted_episodes/taskeval_negoeval/actual_movement.json"

# Intent prediction
# python intent_prediction.py --res_path="data/formatted_episodes/taskeval_negoeval/actual_movement_v2.json" --tgt_path="data/intent_response/taskeval_negoeval/test.jsonl" --split_begin=0 --split_end None

# Intent value evaluate
# python intent_value_evaluate.py --task_eval --res_path="data/intent_response/taskeval_negoeval_whole_finetune/intent_human_dialogue.jsonl" --tgt_path="data/intent_value/taskeval_negoeval_whole_finetune/actual_movement.json"