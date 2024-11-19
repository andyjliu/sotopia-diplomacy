# llama3 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama3_8b.sh > logs/stdout_llama3_8b.txt 2> logs/stderr_llama3_8b.txt

source ~/.bashrc
conda activate lp
# MODEL_DIR="/data/user_data/wenkail/.cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"

# MODEL_DIR="/data/user_data/wenkail/llm_personality/llama_big_five"
# MODEL_DIR="/compute/babel-5-23/jiaruil5/personality/checkpoints/word5_lr1e-4/checkpoint-3000"
MODEL_DIR="/data/models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct/"
LORA_DIR=""
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3639 \
    --model=$MODEL_DIR \
    --tokenizer=$LORA_DIR \
    --chat-template "chat_templates/llama3.jinja" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=8192

    # 3636
    # 3640