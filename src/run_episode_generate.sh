source ~/.bashrc

conda activate sotopia
# [llama3-70b-analysis, gpt-4-analysis, specific_human_anno_llama3_70b, specific_human_anno_gpt_4, specific_human_anno_gpt_4_new]
# python generate_episode.py --model gpt-4-turbo --tag gpt-4-analysis --sample_size 10 --sub_sample 
# python generate_episode.py --epi_tag te_n_with_previous_llama3_70b --split_begin=0 --split_end=26
# python generate_episode.py --epi_tag te_n_with_previous_llama3_70b --split_begin=26 --split_end=53
python generate_episode.py --epi_tag coop_with_flausible_move_v4

# Without fluasible Moves:
# coop_without_flausible_move_v2

# --env_tag tv3
# --split_begin=53 --split_end=80
# python generate_episode.py --epi_tag te_n_with_previous_llama3_70b --split_begin=80 --split_end None

# --model gpt-4-turbo

# python generate_episode.py --epi_tag new_taskeval_llama3_within_10_turns --split_begin=0 --split_end None 

# --model gpt-4-turbo

# --env_tag ntaske