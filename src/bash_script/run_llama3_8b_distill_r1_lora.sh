# llama3 8b distill r1
# CUDA_VISIBLE_DEVICES=2 bash run_llama3_8b_distill_r1.sh

source ~/.bashrc
conda activate inf

MODEL_DIR="/compute/babel-14-33/wenkail/DeepSeek-R1-Distill-Llama-8B/"
LORA_DIR="name=/compute/babel-14-33/wenkail/checkpoints/r1_distill_llama3_8b_lora_sft"

test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3648 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --lora_modules=$LORA_DIR \
    --enable-lora \
    --max-lora-rank 32 \
    --gpu-memory-utilization=0.98 \
    --dtype bfloat16 \
    --max-model-len 115824
