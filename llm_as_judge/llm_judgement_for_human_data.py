from openai import OpenAI
import json
from instruction_prompt import InstructionPrompt
import os
import argparse
import pandas as pd
from tqdm import tqdm, trange
import re
from collections import defaultdict, Counter

api_key = os.getenv("OPENAI_API_KEY")

def parse_llm_response(response):
    """
    将llm的response解析为一个map，key为序号（int），value为yes/no的1/0
    例如: "1. yes\n2. no\n3. yes" -> {1: 1, 2: 0, 3: 1}
    如果答案数不足8个，则返回空dict
    """
    result = {}
    lines = response.strip().split('\n')
    for line in lines:
        m = re.match(r'(\d+)\.\s*(yes|no)', line.strip(), re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            val = 1 if m.group(2).lower() == 'yes' else 0
            result[idx] = val
    if len(result) != 8:
        return {}
    return result

def process_and_save(data, output_file, is_csv=False):
    """
    对llm_response进行解析和处理，保存为json文件
    """
    llm_response_mapping = {
        1: "1gamemove.yes",
        2: "2reasoning.yes",
        3: "3rapport.yes",
        4: "3a_apologies.yes",
        5: "3a_compliment.yes",
        6: "3a_personalthoughts.yes",
        7: "3a_reassurance.yes",
        8: "4shareinformation.yes"
    }
    processed_data = []
    for d in tqdm(data, desc="Processing and saving"):
        # Only call to_dict if d is a pandas Series (i.e., not already a dict)
        if is_csv and hasattr(d, "to_dict"):
            d = d.to_dict()
        parsed = parse_llm_response(d['llm_response'])
        if parsed and len(parsed) == 8:
            d['Input.full_text'] = d['Input.full_text'].replace('\\n', '\n')
            for i in range(1, 9):
                d[llm_response_mapping[i]] = parsed[i]
            if 'llm_response' in d:
                del d['llm_response']
            processed_data.append(d)
        else:
            print(f"Invalid response format or not 8 items: {d['llm_response']}")
            print(f"Parsed: {parsed}")
            print(f"Skipping this item")
            print(f"--------------------------------")
    # 保存为json文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    if processed_data:
        print("Example of processed data:")
        print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))
    else:
        print("No valid processed data found.")
    print(f"\nTotal processed items: {len(processed_data)}") 

class Evaluate:
    def __init__(self, model, output_file):
        self.model = model
        self.output_file = output_file
        if "Llama" in model or "Qwen" in model:
            self.client = OpenAI(
                api_key = "EMPTY",
                base_url = "http://127.0.0.1:9570/v1",
                )
        else:
            self.client = OpenAI(api_key=api_key)

    def call_llm(
        self,
        prompt,
        system_prompt="Your are a helpful assistant.",
        temperature=1,
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

    def multi_trial_responses(
        self,
        prompt,
        system_prompt="Your are a helpful assistant.",
        temperature_list=[0.2, 0.5, 0.7, 1.0],
        num_trials=5,
        show_progress=True,
    ):
        """
        对于每个temperature，进行num_trials次实验，返回每个temperature下的response列表
        返回格式:
        {
            temperature1: [response1, response2, ...],
            temperature2: [response1, response2, ...],
            ...
        }
        """
        all_results = {}
        for temp in temperature_list:
            temp_results = []
            iterator = trange(num_trials, desc=f"Temp {temp}", leave=False) if show_progress else range(num_trials)
            for trial_idx in iterator:
                resp = self.call_llm(prompt, system_prompt=system_prompt, temperature=temp)
                temp_results.append(resp)
            all_results[temp] = temp_results
        return all_results

def compute_consistency(multi_responses):
    """
    multi_responses: dict, temperature -> list of response strings
    Returns:
        consistency_dict: temperature -> consistency (float, 0~1)
        details: temperature -> list of (parsed dict or None)
    """
    consistency_dict = {}
    details = {}
    for temp, responses in multi_responses.items():
        parsed_list = []
        for resp in responses:
            parsed = parse_llm_response(resp)
            parsed_list.append(parsed if parsed else None)
        # Only consider valid parsed responses
        valid = [p for p in parsed_list if p is not None]
        if not valid or len(valid) < 2:
            consistency_dict[temp] = None
            details[temp] = parsed_list
            continue
        # For each of the 8 questions, compute agreement
        agree_count = 0
        total = 0
        for i in range(1, 9):
            answers = [p[i] for p in valid if i in p]
            if len(answers) < 2:
                continue
            # Count the most common answer
            most_common, freq = Counter(answers).most_common(1)[0]
            agree_count += freq
            total += len(answers)
        consistency = agree_count / total if total > 0 else None
        consistency_dict[temp] = consistency
        details[temp] = parsed_list
    return consistency_dict, details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/models/huggingface/meta-llama/Llama-3.1-70B-Instruct/")
    parser.add_argument("--output_file", type=str, default="annotation_data/o3_judge_new_sample.jsonl")
    parser.add_argument("--input_file", type=str, default="annotation_data/wenkai_subsample_expert_annotation_4_n_each_feature.csv")
    parser.add_argument("--process", action="store_true", help="Whether to process llm_response and output processed json")
    parser.add_argument("--processed_output_file", type=str, default="annotation_data/o3_judge_processed.json", help="Output file for processed results")
    parser.add_argument("--consistency", action="store_true", help="Whether to run multi-trial consistency experiments")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials per temperature for consistency")
    parser.add_argument("--temperatures", type=str, default="0.2,0.5,0.7,1.0", help="Comma separated list of temperatures")
    args = parser.parse_args()

    instruction_prompt = InstructionPrompt()
    evaluate = Evaluate(args.model, args.output_file)

    # 判断输入文件类型
    if args.input_file.endswith('.csv'):
        data = pd.read_csv(args.input_file)
        results = []
        if args.consistency:
            # Consistency mode: 多温度多次实验
            temperature_list = [float(t) for t in args.temperatures.split(",")]
            all_consistency_results = []
            for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Main Progress"):
                input_text = row['Input.full_text']
                prompt = instruction_prompt.final_prompt(input_text)
                multi_results = evaluate.multi_trial_responses(
                    prompt,
                    temperature_list=temperature_list,
                    num_trials=args.num_trials,
                    show_progress=True
                )
                # 标记好每个temperature的example 1,2,3,4,5
                labeled_multi_results = {}
                for temp in temperature_list:
                    labeled_multi_results[str(temp)] = [
                        {"example_id": i+1, "response": multi_results[temp][i]} for i in range(len(multi_results[temp]))
                    ]
                # 计算一致性
                consistency_dict, _ = compute_consistency(multi_results)
                row_dict = row.to_dict()
                row_dict['llm_multi_responses'] = labeled_multi_results
                row_dict['llm_consistency'] = {str(k): v for k, v in consistency_dict.items()}
                all_consistency_results.append(row_dict)
            # 保存consistency实验结果
            output_file = args.output_file.replace(".jsonl", "_consistency.jsonl")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for row_dict in all_consistency_results:
                    f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
            print(f"Consistency results saved to {output_file}")
            # 打印整体一致性统计
            all_cons = []
            for row in all_consistency_results:
                for temp, cons in row['llm_consistency'].items():
                    if cons is not None:
                        all_cons.append((temp, cons))
            if all_cons:
                print("Average consistency per temperature:")
                temp2vals = defaultdict(list)
                for temp, cons in all_cons:
                    temp2vals[temp].append(cons)
                for temp in sorted(temp2vals, key=lambda x: float(x)):
                    vals = temp2vals[temp]
                    print(f"  Temp {temp}: {sum(vals)/len(vals):.3f} (n={len(vals)})")
        else:
            for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Main Progress"):
                input_text = row['Input.full_text']
                prompt = instruction_prompt.final_prompt(input_text)
                response = evaluate.call_llm(prompt)
                row_dict = row.to_dict()
                row_dict['llm_response'] = response
                results.append(row_dict)
            if args.process:
                # Pass is_csv=False because results is a list of dicts, not Series
                process_and_save(results, args.processed_output_file, is_csv=False)
            else:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for row_dict in results:
                        f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
    elif args.input_file.endswith('.json') or args.input_file.endswith('.jsonl'):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_file.endswith('.json'):
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f]
        results = []
        if args.consistency:
            temperature_list = [float(t) for t in args.temperatures.split(",")]
            all_consistency_results = []
            for item in tqdm(data, desc="Main Progress"):
                input_text = item['Input.full_text']
                prompt = instruction_prompt.final_prompt(input_text)
                multi_results = evaluate.multi_trial_responses(
                    prompt,
                    temperature_list=temperature_list,
                    num_trials=args.num_trials,
                    show_progress=True
                )
                labeled_multi_results = {}
                for temp in temperature_list:
                    labeled_multi_results[str(temp)] = [
                        {"example_id": i+1, "response": multi_results[temp][i]} for i in range(len(multi_results[temp]))
                    ]
                consistency_dict, _ = compute_consistency(multi_results)
                item['llm_multi_responses'] = labeled_multi_results
                item['llm_consistency'] = {str(k): v for k, v in consistency_dict.items()}
                all_consistency_results.append(item)
            output_file = args.output_file.replace(".jsonl", "_consistency.jsonl")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for item in all_consistency_results:
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Consistency results saved to {output_file}")
            # 打印整体一致性统计
            all_cons = []
            for item in all_consistency_results:
                for temp, cons in item['llm_consistency'].items():
                    if cons is not None:
                        all_cons.append((temp, cons))
            if all_cons:
                print("Average consistency per temperature:")
                temp2vals = defaultdict(list)
                for temp, cons in all_cons:
                    temp2vals[temp].append(cons)
                for temp in sorted(temp2vals, key=lambda x: float(x)):
                    vals = temp2vals[temp]
                    print(f"  Temp {temp}: {sum(vals)/len(vals):.3f} (n={len(vals)})")
        else:
            for item in tqdm(data, desc="Main Progress"):
                # 假设json格式下输入字段为'Input.full_text'
                input_text = item['Input.full_text']
                prompt = instruction_prompt.final_prompt(input_text)
                response = evaluate.call_llm(prompt)
                item['llm_response'] = response
                results.append(item)
            if args.process:
                process_and_save(results, args.processed_output_file)
            else:
                with open(args.output_file, 'w', encoding='utf-8') as f_out:
                    for item in results:
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise ValueError("只支持csv, json, jsonl格式的输入文件")