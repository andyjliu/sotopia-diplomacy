import json
import os
import argparse
from tqdm import tqdm
from openai import OpenAI
from instruction_prompt import InstructionPrompt

api_key = os.getenv("OPENAI_API_KEY")

class Evaluate:
    def __init__(self, model):
        self.model = model
        if "llama" in model.lower() or "qwen" in model.lower():
            self.client = OpenAI(
                api_key="EMPTY",
                base_url="http://127.0.0.1:3638/v1",
            )
        else:
            self.client = OpenAI(api_key=api_key)

    def call_llm(
        self,
        prompt,
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

def process_message(evaluate, instruction_prompt, msg):
    # Remove 'llm_judgement_matches' if present
    if 'llm_judgement_matches' in msg:
        del msg['llm_judgement_matches']
    prompt = instruction_prompt.final_prompt(msg['message'])
    try:
        response = evaluate.call_llm(prompt)
    except Exception as e:
        response = f"Error: {e}"
    msg['llm_judgement'] = response
    return msg

def process_item(item, evaluate, instruction_prompt):
    if 'phases' in item:
        for phase in tqdm(item['phases'], desc=f"Phases for game {item['id']}"):
            messages = phase.get('messages', [])
            for msg in messages:
                process_message(evaluate, instruction_prompt, msg)
    return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct/")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # Load the interaction data
    # 读取json格式
    if args.data_file.endswith('.json'):
        with open(args.data_file, 'r') as f:
            data = json.load(f)
    # 读取jsonl格式
    elif args.data_file.endswith('.jsonl'):
        with open(args.data_file, 'r') as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError("Unsupported file format. Please use .json or .jsonl")

    instruction_prompt = InstructionPrompt()
    evaluate = Evaluate(args.model)
    with open(args.output_file, 'w') as fout:
        for item in tqdm(data, desc="Annotating items"):
            item = process_item(item, evaluate, instruction_prompt)
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            
# python llm_judgement_for_cicero_data.py --model /data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct/ --data_file /data/user_data/wenkail/sotopia_diplomacy/processed_data_analysis/500_sample_with_taskeval_supply_center_qwen_judge.jsonl --output_file /data/user_data/wenkail/sotopia_diplomacy/processed_data_analysis/500_sample_with_taskeval_supply_center_llama_8b_judge.jsonl