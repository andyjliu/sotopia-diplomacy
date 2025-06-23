# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b.sh > logs/stdout_llama3_8b.txt 2> logs/stderr_llama3_8b.txt

source ~/.bashrc
conda activate inf

MODEL_DIR="/compute/babel-14-33/wenkail/Qwen3-8B/"
test -d "$MODEL_DIR"
CUDA_VISIBLE_DEVICES=0,1 python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3638 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --gpu-memory-utilization=0.9 \
    --dtype bfloat16 \
    --max-model-len 40960

    # 3636
    # 3640