# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b.sh > logs/stdout_llama3_8b.txt 2> logs/stderr_llama3_8b.txt

source ~/.bashrc
conda activate sotopia
MODEL_DIR="/data/user_data/wenkail/.cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3636 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "llama3.jinja" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=8192