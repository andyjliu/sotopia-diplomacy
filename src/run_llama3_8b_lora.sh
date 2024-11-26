# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b_lora.sh

source ~/.bashrc
conda activate inf

MODEL_DIR="/data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/"
LORA_DIR="name=/data/user_data/wenkail/sotopia_diplomacy/8b_sft_lora_checkpoints_1e-4/checkpoint-300/"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3639 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --lora_modules=$LORA_DIR \
    --chat-template "chat_templates/llama3.jinja" \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.9 \
    --dtype bfloat16 \
    --max-num-seqs 32 \
    --max-num-batched-tokens=8192

    # 3636
    # 3640