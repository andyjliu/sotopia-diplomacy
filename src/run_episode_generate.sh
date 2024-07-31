source ~/.bashrc

conda activate sotopia
# [llama3-70b-analysis, gpt-4-analysis, specific_human_anno_llama3_70b, specific_human_anno_gpt_4, specific_human_anno_gpt_4_new]
# python generate_episode.py --model gpt-4-turbo --tag gpt-4-analysis --sample_size 10 --sub_sample 
python generate_episode.py --model llama3_70b --tag llama3-70b-analysis --sample_size 10 --sub_sample 