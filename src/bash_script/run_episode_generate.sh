source ~/.bashrc

conda activate sotopia
# [llama3-70b-analysis, gpt-4-analysis, specific_human_anno_llama3_70b, specific_human_anno_gpt_4, specific_human_anno_gpt_4_new]

python generate_episode.py --epi_tag demo_llama --env_tag demo_v2 --model llama3_70b
python generate_episode.py --epi_tag demo_gpt_4o --env_tag demo_v2 --model gpt-4o
# gpt-4o-mini
# llama3_70b

# --env_tag test_finetune_v2 

# Without fluasible Moves:
# coop_without_flausible_move_v2

# --env_tag tv3
# --split_begin=53 --split_end=80
# python generate_episode.py --epi_tag te_n_with_previous_llama3_70b --split_begin=80 --split_end None

# --model gpt-4-turbo

# python generate_episode.py --epi_tag new_taskeval_llama3_within_10_turns --split_begin=0 --split_end None 

# --model gpt-4-turbo

# --env_tag ntaske