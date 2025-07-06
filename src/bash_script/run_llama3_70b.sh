# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_llama3_70b.sh
source ~/.bashrc
conda activate inf

MODEL_DIR="/data/models/huggingface/meta-llama/Llama-3.1-70B-Instruct/"
test -d "$MODEL_DIR"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m vllm.entrypoints.openai.api_server \
  --port 9570 \
  --model $MODEL_DIR \
  --tokenizer $MODEL_DIR \
  --chat-template ../chat_templates/llama3.jinja \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 60000 \
  --max_num_seqs 1 \
  --swap-space 16 \
  --cpu-offload-gb 4 \
  --block-size 16
  # --max-num-batched-tokens 4096
  

# sources: https://github.com/vllm-project/vllm/pull/2249

#  lora_modules