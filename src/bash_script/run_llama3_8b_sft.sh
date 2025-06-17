# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b.sh > logs/stdout_llama3_8b.txt 2> logs/stderr_llama3_8b.txt

source ~/.bashrc
conda activate inf

# The baseline model
# MODEL_DIR="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct/"

# The negotiation model
MODEL_DIR="/compute/babel-14-33/wenkail/checkpoints/llama3_8b_full_sft/checkpoint-1500/"
test -d "$MODEL_DIR"
CUDA_VISIBLE_DEVICES=0,1 python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3638 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --tensor-parallel-size=2 \
    --gpu-memory-utilization=0.9 \
    --dtype bfloat16 \
    --max-model-len 115824 \
    --chat-template "../chat_templates/llama3.jinja"

    # 3636
    # 3640

    # --chat-template "../chat_templates/llama3.jinja" \
