source ~/.bashrc

conda activate sotopia
# [llama3-70b-analysis, gpt-4-analysis, specific_human_anno_llama3_70b, specific_human_anno_gpt_4, specific_human_anno_gpt_4_new]

# 1.
# python ../generate_episode.py --epi_tag test_llama3.1_v2 --env_tag finetune --model llama3_8b --split_begin=0 --split_end=3
# 2. 
# python generate_episode.py --epi_tag llama3.18b_lora --env_tag finetune_format --model llama3_8b --split_begin=0 --split_end None
# 3. 
# python generate_episode.py --epi_tag llama3.18b --env_tag finetune_format --model llama3_8b --split_begin=0 --split_end None
# 4. 
python generate_episode.py --epi_tag llama3.170b_v1 --env_tag finetune_format --model llama3_70b --split_begin=0 --split_end None




# python generate_episode.py --epi_tag demo_gpt_4o --env_tag demo_v2 --model gpt-4o

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