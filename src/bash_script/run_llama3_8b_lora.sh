# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b_lora.sh

source ~/.bashrc
conda activate inf

MODEL_DIR="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct/"
LORA_DIR="name=/data/user_data/wenkail/sotopia_diplomacy/LLaMA-Factory/checkpoints/llama3_8b_lora_sft"
test -d "$MODEL_DIR"
CUDA_VISIBLE_DEVICES=0 python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3639 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --lora_modules=$LORA_DIR \
    --enable-lora \
    --max-lora-rank 32 \
    --chat-template "../chat_templates/llama3.jinja" \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.9 \
    --dtype bfloat16 \
    --max-model-len 115824

    # 3636
    # 3640