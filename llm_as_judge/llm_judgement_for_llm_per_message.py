from openai import OpenAI
import json
from instruction_prompt import InstructionPrompt
import os
import argparse
import pandas as pd
from tqdm import tqdm, trange
import re
from collections import defaultdict, Counter
from rich import print
import sys
sys.path.append("../../")
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile

api_key = os.getenv("OPENAI_API_KEY")

def parse_llm_response(response):
    """
    解析llm的response为一个map，key为序号（int），value为yes/no的1/0
    例如: "1. yes\n2. no\n3. yes" -> {1: 1, 2: 0, 3: 1}
    如果答案数不足8个，则返回空dict

    新增：如果response中包含</think>，则只解析</think>之后的内容
    """
    # 如果有</think>，只取其后的内容
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
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

def process_single_result(result):
    """
    对单条result进行解析和处理，返回处理后的dict（如果无效则返回None）
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
    parsed = parse_llm_response(result['llm_response'])
    if parsed and len(parsed) == 8:
        for i in range(1, 9):
            result[llm_response_mapping[i]] = parsed[i]
        if 'llm_response' in result:
            del result['llm_response']
        return result
    else:
        print(f"Invalid response format or not 8 items: {result['llm_response']}")
        print(f"Parsed: {parsed}")
        print(f"Skipping this item")
        print(f"--------------------------------")
        return None

class Evaluate:
    def __init__(self, model):
        self.model = model
        if "llama" in model.lower() or "qwen" in model.lower():
            self.client = OpenAI(
                api_key = "EMPTY",
                base_url = "http://127.0.0.1:3640/v1",
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
    parser.add_argument("--output_file", type=str, required=False, help="Output file for raw results")
    parser.add_argument("--epi_tag", type=str, required=True, help="Episode tag")
    parser.add_argument("--process", action="store_true", help="Whether to process llm_response and output processed json")
    parser.add_argument("--processed_output_file", type=str, required=True, help="Output file for processed results")
    parser.add_argument("--consistency", action="store_true", help="Whether to run multi-trial consistency experiments")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials per temperature for consistency")
    parser.add_argument("--temperatures", type=str, default="0.2,0.5,0.7,1.0", help="Comma separated list of temperatures")
    args = parser.parse_args()

    instruction_prompt = InstructionPrompt()
    evaluate = Evaluate(args.model)
    all_epi_pks = list(EpisodeLog.all_pks())
    epis = []
    for pk in tqdm(all_epi_pks):
        epi = EpisodeLog.get(pk)
        if epi.tag == args.epi_tag:
            epis.append(epi)
    # import pdb; pdb.set_trace()
    if args.consistency:
        # Consistency mode: 多温度多次实验
        temperature_list = [float(t) for t in args.temperatures.split(",")]
        all_consistency_results = []
        for epi in tqdm(epis, desc="Main Progress"):
            for i in range(1, len(epi.messages)):
                input_text = epi.messages[i][0][2]
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
                row_dict = epi.to_dict()
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
        # 直接对每一条生成的result进行处理并写入jsonl
        if args.process:
            num_processed = 0
            example_printed = False
            with open(args.processed_output_file, 'w', encoding='utf-8') as f:
                for epi in tqdm(epis, desc=f"Main Progress on {args.epi_tag} Results"):
                    for i in tqdm(range(1, len(epi.messages)), desc=f"Epi Progress on {args.epi_tag} Results"):
                        input_text = epi.messages[i][0][2]
                        result = {}
                        prompt = instruction_prompt.final_prompt(input_text)
                        response = evaluate.call_llm(prompt)
                        result['epi_pk'] = epi.pk
                        result['turn_idx'] = i - 1
                        result['epi_tag'] = epi.tag
                        result['epi_messages'] = input_text
                        result['llm_response'] = response
                        processed = process_single_result(result)
                        if processed is not None:
                            f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                            num_processed += 1
                            if not example_printed:
                                print("Example of processed data:")
                                print(json.dumps(processed, indent=2, ensure_ascii=False))
                                example_printed = True
                print(f"\nTotal processed items: {num_processed}")
        else:
            results = []
            for epi in tqdm(epis, desc=f"Main Progress on {args.epi_tag} Results"):
                for i in tqdm(range(1, len(epi.messages)), desc=f"Epi Progress on {args.epi_tag} Results"):
                    input_text = epi.messages[i][0][2]
                    result = {}
                    prompt = instruction_prompt.final_prompt(input_text)
                    response = evaluate.call_llm(prompt)
                    result['epi_pk'] = epi.pk
                    result['turn_idx'] = i - 1
                    result['epi_tag'] = epi.tag
                    result['epi_messages'] = input_text
                    result['llm_response'] = response
                    results.append(result)
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for row_dict in results:
                    f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
                    
# python llm_judgement_for_llm.py --model /compute/babel-14-33/wenkail/Qwen3-8B/ --epi_tag llama_8b_sft_lora_finetune_format_v2 --process --processed_output_file llama_8b_sft_lora_finetune_format_v2_qwen3_8b_judgement.jsonl

# python llm_judgement_for_llm.py --model /compute/babel-14-33/wenkail/Qwen3-8B/ --epi_tag llama_8b_finetune_format_v2 --process --processed_output_file llama_8b_finetune_format_v2_qwen3_8b_judgement.jsonl

# python llm_judgement_for_llm_per_message.py --model /compute/babel-14-33/wenkail/Qwen3-8B/ --epi_tag qwen3_8b_finetune_format_v6 --process --processed_output_file qwen3_8b_finetune_format_v6_llama3_8b_judgement.jsonl

# python llm_judgement_for_llm_per_message.py --model /compute/babel-14-33/wenkail/Qwen3-8B/ --epi_tag llama_8b_finetune_format_v6 --process --processed_output_file llama_8b_finetune_format_v6_llama3_8b_judgement.jsonl

# python llm_judgement_for_llm_per_message.py --model /compute/babel-14-33/wenkail/Qwen3-8B/ --epi_tag r1_distill_llama3_8b_finetune_format_v6 --process --processed_output_file r1_distill_llama3_8b_finetune_format_v6_llama3_8b_judgement.jsonl