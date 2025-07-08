# Qwen 8b
# CUDA_VISIBLE_DEVICES=1 bash run_qwen_8b.sh

source ~/.bashrc
conda activate inf

MODEL_DIR="/compute/babel-14-33/wenkail/Qwen3-8B/"

test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3640 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --gpu-memory-utilization=0.9 \
    --dtype bfloat16 \
    --max-model-len 40960