
# Exit immediately if a command exits with a non-zero status
set -e

# Here the parse means the v2
EPI_TAG="test_demo_v4"
FORMAT_EPISODE_PATH="data/formatted_episodes/human_ai_interaction/${EPI_TAG}.json"
INTENT_RESPONSE_PATH="data/intent_response/human_ai_interaction/${EPI_TAG}.json"
INTENT_VALUE_PATH="data/intent_value/human_ai_interaction/${EPI_TAG}.json"


source ~/.bashrc

conda activate sotopia
PYTHONWARNINGS="ignore" python human_ai_interaction.py --game_id 74152 --env_tag demo_v2 --phase_name F1904M --c1 Turkey --c2 Germany --epi_tag=$EPI_TAG

echo "Calculating TaskEval Score..."
python get_llm_intent_episode.py --tag=$EPI_TAG --tgt_path=$FORMAT_EPISODE_PATH

# echo "Running intent_prediction.py ..."
python intent_prediction.py --res_path=$FORMAT_EPISODE_PATH --tgt_path=$INTENT_RESPONSE_PATH --split_begin=0 --split_end None

conda activate diplomacy_cicero

# echo "Running intent_value_evaluate.py ..."
python intent_value_evaluate.py --task_eval --move --res_path=$INTENT_RESPONSE_PATH --tgt_path=$INTENT_VALUE_PATH

python read_taskeval.py --file_name=$INTENT_VALUE_PATH

