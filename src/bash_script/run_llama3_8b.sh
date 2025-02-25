# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b.sh > logs/stdout_llama3_8b.txt 2> logs/stderr_llama3_8b.txt

source ~/.bashrc
conda activate inf

MODEL_DIR="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct/"
test -d "$MODEL_DIR"
CUDA_VISIBLE_DEVICES=0 python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3638 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "../chat_templates/llama3.jinja" \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.9 \
    --dtype bfloat16 \
    --max-model-len 115824 \

    # 3636
    # 3640