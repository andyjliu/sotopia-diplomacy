from openai import OpenAI
import json
from instruction_prompt import InstructionPrompt
from instruction_prompt_baseline import InstructionPromptBaseline
from instruction_prompt_fewshot import InstructionPromptFewShot
import os
import argparse
import pandas as pd
from tqdm import tqdm, trange
import re
from collections import defaultdict, Counter
from typing import Literal, Dict
from pydantic import BaseModel, Field, ValidationError

api_key = os.getenv("OPENAI_API_KEY")

# Define strict yes/no type
YesNo = Literal['yes', 'no']

class LLM8dim(BaseModel):
    one: YesNo = Field(..., alias='1gamemove.yes')
    two: YesNo = Field(..., alias='2reasoning.yes')
    three: YesNo = Field(..., alias='3rapport.yes')
    four: YesNo = Field(..., alias='3a_apologies.yes')
    five: YesNo = Field(..., alias='3a_compliment.yes')
    six: YesNo = Field(..., alias='3a_personalthoughts.yes')
    seven: YesNo = Field(..., alias='3a_reassurance.yes')
    eight: YesNo = Field(..., alias='4shareinformation.yes')

    class Config:
        allow_population_by_field_name = True

    def to_int_dict(self) -> Dict[str, int]:
        return {
            '1gamemove.yes': 1 if self.one == 'yes' else 0,
            '2reasoning.yes': 1 if self.two == 'yes' else 0,
            '3rapport.yes': 1 if self.three == 'yes' else 0,
            '3a_apologies.yes': 1 if self.four == 'yes' else 0,
            '3a_compliment.yes': 1 if self.five == 'yes' else 0,
            '3a_personalthoughts.yes': 1 if self.six == 'yes' else 0,
            '3a_reassurance.yes': 1 if self.seven == 'yes' else 0,
            '4shareinformation.yes': 1 if self.eight == 'yes' else 0,
        }

def parse_llm_response_structured(resp: str) -> Dict[str, int]:
    tmp = {}
    for line in resp.strip().splitlines():
        m = re.match(r'(\d+)\.\s*(yes|no)', line.strip(), re.IGNORECASE)
        if m:
            tmp[m.group(1)] = m.group(2).lower()
    if len(tmp) != 8:
        raise ValueError(f"Expected 8 items, got {len(tmp)}")
    data = {
        '1gamemove.yes': tmp['1'],
        '2reasoning.yes': tmp['2'],
        '3rapport.yes': tmp['3'],
        '3a_apologies.yes': tmp['4'],
        '3a_compliment.yes': tmp['5'],
        '3a_personalthoughts.yes': tmp['6'],
        '3a_reassurance.yes': tmp['7'],
        '4shareinformation.yes': tmp['8'],
    }
    model = LLM8dim(**data)
    return model.to_int_dict()

class Evaluate:
    def __init__(self, model, output_file):
        self.model = model
        self.output_file = output_file
        if 'r1' in model.lower():
            self.client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:3642/v1")
        else:
            if "llama" in model.lower():
                if "70b" in model.lower():
                    self.client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:9570/v1")
                elif "8b" in model.lower():
                    self.client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:3638/v1")
            elif "qwen" in model.lower():
                self.client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:3640/v1")
            else:
                self.client = OpenAI(api_key=api_key)

    def call_llm(self, prompt, system_prompt="You are a helpful assistant.", temperature=1):
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

    def call_llm_and_parse(self, prompt, max_retries=1000, temperature=1):
        for attempt in range(max_retries):
            resp = self.call_llm(prompt, temperature=temperature)
            try:
                return parse_llm_response_structured(resp)
            except (ValueError, ValidationError) as e:
                print(f"[Attempt {attempt+1}/{max_retries}] parse failed: {e}. Retrying...")
        raise RuntimeError("Failed to get valid 8-dim yes/no response")

    def multi_trial_responses(self, prompt, system_prompt="You are a helpful assistant.",
                              temperature_list=[0.2, 0.5, 0.7, 1.0], num_trials=5, show_progress=True):
        all_results = {}
        for temp in temperature_list:
            temp_results = []
            iterator = trange(num_trials, desc=f"Temp {temp}", leave=False) if show_progress else range(num_trials)
            for _ in iterator:
                temp_results.append(self.call_llm(prompt, system_prompt, temp))
            all_results[temp] = temp_results
        return all_results

def process_and_save(data, output_file, is_csv=False):
    """
    对llm_response进行解析和处理，保存为结构化json文件，确保8个维度均为yes/no并合法。
    """
    processed_data = []
    for d in tqdm(data, desc="Processing and saving"):
        # 如果是DataFrame Series就转换为字典
        if is_csv and hasattr(d, "to_dict"):
            d = d.to_dict()
        try:
            structured = parse_llm_response_structured(d['llm_response'])
        except (ValueError, ValidationError) as e:
            print(f"Skipping invalid response: {d['llm_response']}\nError: {e}")
            continue
        d['Input.full_text'] = d['Input.full_text'].replace('\\n', '\n')
        d.update(structured)
        d.pop('llm_response', None)
        processed_data.append(d)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    if processed_data:
        print("Example processed item:\n", json.dumps(processed_data[0], indent=2, ensure_ascii=False))
    print(f"\nTotal processed items: {len(processed_data)}")

def compute_consistency(multi_responses):
    consistency_dict, details = {}, {}
    for temp, responses in multi_responses.items():
        parsed = []
        for resp in responses:
            try:
                parsed.append(parse_llm_response_structured(resp))
            except Exception:
                parsed.append(None)
        valid = [p for p in parsed if p]
        if len(valid) < 2:
            consistency_dict[temp] = None
            details[temp] = parsed
            continue
        agree_count = total = 0
        for i in range(1, 9):
            key = list(valid[0].keys())[i - 1]
            answers = [p[key] for p in valid if key in p]
            if not answers:
                continue
            most, freq = Counter(answers).most_common(1)[0]
            agree_count += freq
            total += len(answers)
        consistency_dict[temp] = agree_count / total if total else None
        details[temp] = parsed
    return consistency_dict, details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--consistency", action="store_true")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--temperatures", type=str, default="0.2,0.5,0.7,1.0")
    parser.add_argument("--max_retries", type=int, default=1000)
    args = parser.parse_args()

    if args.baseline:
        instruction_prompt = InstructionPromptBaseline()
    elif args.fewshot:
        instruction_prompt = InstructionPromptFewShot()
    else:
        instruction_prompt = InstructionPrompt()
        
    evaluate = Evaluate(args.model, args.output_file or "unused.json")

    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
        data = df.to_dict(orient='records')
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f) if args.input_file.endswith('.json') else [json.loads(l) for l in f]

    if args.consistency:
        results = []
        temps = [float(t) for t in args.temperatures.split(",")]
        for row in tqdm(data, desc="Consistency eval"):
            prompt = instruction_prompt.final_prompt(row['Input.full_text'])
            multi_results = evaluate.multi_trial_responses(prompt, temperature_list=temps, num_trials=args.num_trials)
            labeled = {str(t): [{"example_id": i+1, "response": r} for i, r in enumerate(multi_results[t])] for t in temps}
            row['llm_multi_responses'] = labeled
            row['llm_consistency'], _ = compute_consistency(multi_results)
            results.append(row)
        out_file = args.output_file.replace(".jsonl", "_consistency.jsonl")
        with open(out_file, 'w', encoding='utf-8') as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        return

    results = []
    for row in tqdm(data, desc="Evaluating"):
        prompt = instruction_prompt.final_prompt(row['Input.full_text'])
        try:
            structured = evaluate.call_llm_and_parse(prompt, max_retries=args.max_retries)
            row.update(structured)
        except (RuntimeError, ValueError, ValidationError) as e:
            print(f"Skipping due to error: {e}")
            continue
        results.append(row)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
            
if __name__ == "__main__":
    main()
