# Qwen 8b
# CUDA_VISIBLE_DEVICES=0 bash run_qwen_8b.sh

source ~/.bashrc
conda activate inf

MODEL_DIR="/compute/babel-14-33/wenkail/Qwen3-8B/"
LORA_DIR="name=/data/user_data/wenkail/sotopia_diplomacy/LLaMA-Factory/checkpoints/qwen_8b_lora_sft"

test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3641 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --lora_modules=$LORA_DIR \
    --enable-lora \
    --max-lora-rank 32 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.9 \
    --dtype bfloat16 \
    --max-model-len 40960

    # 3636
    # 3640