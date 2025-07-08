# Mistral 2506 24B
# CUDA_VISIBLE_DEVICES=0,1 bash run_mistral.sh

source ~/.bashrc
conda activate inf

MODEL_DIR="/compute/babel-14-33/wenkail/Magistral-Small-2506-24B/"

test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3645 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization=0.9 \
    --dtype bfloat16 \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --tool-call-parser mistral \
    --max-model-len 40960