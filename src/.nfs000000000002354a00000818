# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b.sh > logs/stdout_llama3_8b.txt 2> logs/stderr_llama3_8b.txt

source ~/.bashrc
conda activate inf

MODEL_DIR="/data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3639 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "chat_templates/llama3.jinja" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=8192

    # 3636
    # 3640