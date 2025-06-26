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
# python generate_episode.py --epi_tag llama3.170b_v1 --env_tag finetune_format --model llama3_70b --split_begin=0 --split_end None

# Compare with/without finetune in Negotiation Evaluation Version 2
# python generate_episode.py --epi_tag llama_8b_negoeval_v2_whole_v4 --env_tag whole_finetune_format_without_diplomacy_background --model llama3_8b --agent_model=gpt-4o-mini --split_begin=0 --split_end None --picked_envs /home/wenkail/diplomacy/sotopia-diplomacy/src/data/intent_value/taskeval_negoeval_whole_finetune/envs.txt
# 2. python generate_episode.py --epi_tag llama_8b_lora_negoeval_whole --env_tag whole_finetune_format_without_diplomacy_background --model llama3_8b_lora --agent_model=llama3_70b --split_begin=0 --split_end None
# python generate_episode.py --epi_tag llama_70b_negoeval_v2_whole_v4 --env_tag whole_finetune_format_without_diplomacy_background --model llama3_70b --agent_model=gpt-4o-mini --split_begin=0 --split_end None --picked_envs /home/wenkail/diplomacy/sotopia-diplomacy/src/data/intent_value/taskeval_negoeval_whole_finetune/envs.txt
python ../generate_episode.py --epi_tag llama_8b_sft_finetune_v3 --env_tag finetune_format_latest --model llama3_8b --agent_model=llama3_8b --split_begin=0 --split_end 250 &
python ../generate_episode.py --epi_tag llama_8b_sft_finetune_v3 --env_tag finetune_format_latest --model llama3_8b --agent_model=llama3_8b --split_begin=250 --split_end 500 &
python ../generate_episode.py --epi_tag llama_8b_sft_finetune_v3 --env_tag finetune_format_latest --model llama3_8b --agent_model=llama3_8b --split_begin=500 --split_end 750 &
python ../generate_episode.py --epi_tag llama_8b_sft_finetune_v3 --env_tag finetune_format_latest --model llama3_8b --agent_model=llama3_8b --split_begin=750 --split_end None &
wait

# python generate_episode.py --epi_tag llama_8b_sft_finetune_format_v2 --env_tag finetune_format_latest --model llama3_8b --agent_model=llama3_8b --split_begin=0 --split_end None

# python generate_episode.py --epi_tag llama_8b_finetune_format_v2 --env_tag finetune_format_latest --model llama3_8b --agent_model=llama3_8b --split_begin=0 --split_end None


# New generation
# python generate_episode.py --epi_tag qwen3_8b_finetune_format_v6 --env_tag finetune_format_latest --model qwen3_8b --split_begin=0 --agent_model=qwen3_8b --split_end None

# python generate_episode.py --epi_tag llama_8b_finetune_format_v6 --env_tag finetune_format_latest --model llama3_8b --split_begin=0 --agent_model=llama3_8b --split_end None

# python generate_episode.py --epi_tag r1_distill_llama3_8b_finetune_format_v6 --env_tag finetune_format_latest --model r1_distill_llama3_8b --split_begin=0 --agent_model=r1_distill_llama3_8b --split_end None