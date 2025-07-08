# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b.sh > logs/stdout_llama3_8b.txt 2> logs/stderr_llama3_8b.txt

source ~/.bashrc
conda activate inf

# The baseline model
# MODEL_DIR="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct/"

# The negotiation model
MODEL_DIR="/compute/babel-14-33/wenkail/checkpoints/llama3_8b_full_sft/checkpoint-1500/"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3637 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.95 \
    --dtype bfloat16
