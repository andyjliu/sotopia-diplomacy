#!/bin/bash

# 固定的路径
FORMAT_EPISODE_PATH="../data/formatted_episodes/taskeval_negoeval/llama_8b_lora_negoeval_plus.json"

source ~/.bashrc
echo "Activating sotopia environment for get_intent_episode..."
conda activate sotopia

echo "Running get_intent_episode.py ..."
python ../get_llm_intent_episode.py --tag llama_8b_lora_negoeval_plus --tgt_path=$FORMAT_EPISODE_PATH

# 定义 DIALOGUE_END_INDEX 数组
DIALOGUE_END_INDEX_ARRAY=(4 6 8 10 14 18)
# 设备编号计数器
i=0

# 针对每个 DIALOGUE_END_INDEX 分别执行 intent_prediction 和 intent_value_evaluate
for DIALOGUE_END_INDEX in "${DIALOGUE_END_INDEX_ARRAY[@]}"; do
    # 分配 GPU 设备，利用 i mod 6（依次分配0~5）
    GPU_DEVICE=$(( i % 6 ))
    ((i++))
    
    # 根据当前 DIALOGUE_END_INDEX 生成目标文件路径，在原文件名末尾添加后缀
    INTENT_RESPONSE_PATH="../data/intent_response/taskeval_negoeval/llama_8b_lora_negoeval_plus_cut_${DIALOGUE_END_INDEX}.jsonl"
    INTENT_VALUE_PATH="../data/intent_value/taskeval_negoeval/llama_8b_lora_negoeval_plus/llama_8b_lora_negoeval_plus_cut_${DIALOGUE_END_INDEX}.json"

    # 并行运行 intent_prediction（使用 sotopia 环境）
    (
        source ~/.bashrc
        echo "Activating sotopia environment for intent_prediction with DIALOGUE_END_INDEX=${DIALOGUE_END_INDEX} on GPU ${GPU_DEVICE}..."
        conda activate sotopia
        echo "Running intent_prediction.py with DIALOGUE_END_INDEX=${DIALOGUE_END_INDEX} on GPU ${GPU_DEVICE}..."
        CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python ../intent_prediction.py \
            --res_path=$FORMAT_EPISODE_PATH \
            --tgt_path=$INTENT_RESPONSE_PATH \
            --split_begin=0 \
            --split_end None \
            --cut \
            --end_turn=$DIALOGUE_END_INDEX 

        echo "Activating diplomacy_cicero environment for intent_value_evaluate with DIALOGUE_END_INDEX=${DIALOGUE_END_INDEX} on GPU ${GPU_DEVICE}..."
        conda activate diplomacy_cicero
        echo "Running intent_value_evaluate.py with DIALOGUE_END_INDEX=${DIALOGUE_END_INDEX} on GPU ${GPU_DEVICE}..."
        CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python ../intent_value_evaluate.py \
            --task_eval \
            --move \
            --res_path=$INTENT_RESPONSE_PATH \
            --tgt_path=$INTENT_VALUE_PATH
    ) &
done

wait
echo "Script execution completed."
