import random

def get_model_config(model):
    if 'gpt' in model:
        api_key = None
        # org_id = random.sample([0, 1], 1)
        org_id = 1
        model_path = None
    elif model == 'llama3_8b':
        api_key = "EMPTY"
        org_id = "http://127.0.0.1:3636/v1"
        model_path = "/data/user_data/wenkail/.cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"
    elif model == 'llama2_7b':
        api_key = "EMPTY"
        org_id = "http://127.0.0.1:2525/v1"
        model_path = "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf"
    return {
        'api_key': api_key, 
        'org_id': org_id, 
        'model_path': model_path
    }