# llama3 70b instruct
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_llama3_70b.sh > logs/stdout_llama3_70b.txt 2> logs/stderr_llama3_70b.txt
# source ~/.bashrc
# conda activate inf

MODEL_DIR="/data/models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct/"

test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=9570 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "chat_templates/llama3.jinja" \
    --tensor-parallel-size=4 \
    --max-num-batched-tokens=8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32

# sources: https://github.com/vllm-project/vllm/pull/2249

#  lora_modules