# llama3 70b instruct
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_llama3_70b.sh > logs/stdout_llama3_70b.txt 2> logs/stderr_llama3_70b.txt

source ~/.bashrc
conda activate inf

MODEL_DIR="/data/models/huggingface/meta-llama/Llama-3.1-70B-Instruct/"
LORA_DIR="name=/compute/babel-14-33/wenkail/checkpoints/llama3_70b_lora_sft_ds3_full_800/checkpoint-800"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=9571 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --lora_modules=$LORA_DIR \
    --enable-lora \
    --max-lora-rank 32 \
    --tensor-parallel-size=4 \
    --dtype bfloat16 \
    --gpu-memory-utilization=0.95 \
    --max-model-len 115824

# sources: https://github.com/vllm-project/vllm/pull/2249


# huggingface-cli download google/gemma-3n-E4B-it --local-dir ./gemma-3n-E4B
