source ~/.bashrc
conda activate sotopia

# Human data
# python llm_judgement_for_human_data.py \
#     --model /data/models/huggingface/meta-llama/Llama-3.1-70B-Instruct \
#     --input_file expert_annotation/wenkai_subsample_expert_annotation_4_n_each_feature.csv \
#     --processed_output_file expert_annotation/llama3_70b_judge_processed.json &

# python llm_judgement_for_human_data.py \
#     --model /data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct/ \
#     --input_file expert_annotation/wenkai_subsample_expert_annotation_4_n_each_feature.csv \
#     --processed_output_file expert_annotation/llama3_8b_judge_processed.json &

# python llm_judgement_for_human_data.py \
#     --model /compute/babel-14-33/wenkail/Qwen3-8B/ \
#     --input_file expert_annotation/wenkai_subsample_expert_annotation_4_n_each_feature.csv \
#     --processed_output_file expert_annotation/qwen_8b_judge_processed.json &

# python llm_judgement_for_human_data.py \
#     --model /compute/babel-14-33/wenkail/DeepSeek-R1-Distill-Llama-8B/ \
#     --input_file expert_annotation/wenkai_subsample_expert_annotation_4_n_each_feature.csv \
#     --processed_output_file expert_annotation/r1_llama_8b_judge_processed.json &

# LLM data
python llm_judgement_for_llm_per_message.py --model /compute/babel-14-33/wenkail/Qwen3-8B/ --epi_tag qwen3_8b_finetune_format_v6 --process --processed_output_file qwen3_8b_finetune_format_v6_llama3_8b_judgement.jsonl &

python llm_judgement_for_llm_per_message.py --model /compute/babel-14-33/wenkail/Qwen3-8B/ --epi_tag llama_8b_finetune_format_v6 --process --processed_output_file llama_8b_finetune_format_v6_llama3_8b_judgement.jsonl &

python llm_judgement_for_llm_per_message.py --model /compute/babel-14-33/wenkail/Qwen3-8B/ --epi_tag r1_distill_llama3_8b_finetune_format_v6 --process --processed_output_file r1_distill_llama3_8b_finetune_format_v6_llama3_8b_judgement.jsonl &

wait