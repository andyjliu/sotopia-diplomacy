# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b.sh

source ~/.bashrc
conda activate inf

MODEL_DIR="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct/"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3638 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "../chat_templates/llama3.jinja" \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.98 \
    --dtype bfloat16
    # --max-model-len 40960