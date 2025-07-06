#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM judgement (EpisodeLog version) — pydantic rewrite

用法示例：
python llm_judgement_for_llm_per_message.py \
    --model /compute/babel-14-33/wenkail/Qwen3-8B/ \
    --epi_tag qwen3_8b_lora_finetune_format_v6 \
    --process \
    --processed_output_file epis_judgement/qwen3_8b_lora_finetune_format_v6_qwen3_8b_judgement.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Literal, Sequence, Set, Tuple

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from rich import print
from tqdm import tqdm, trange

sys.path.append("../../")  # for sotopia
from sotopia.database import EpisodeLog  # noqa: E402
from instruction_prompt import InstructionPrompt  # noqa: E402

# --------------------------------------------------------------------------- #
#                         1.  Strict Pydantic schema                          #
# --------------------------------------------------------------------------- #

YesNo = Literal["yes", "no"]


class LLM8dim(BaseModel):
    """Eight‐dimensional yes/no judgement with field-alias mapping."""

    one: YesNo = Field(..., alias="1gamemove.yes")
    two: YesNo = Field(..., alias="2reasoning.yes")
    three: YesNo = Field(..., alias="3rapport.yes")
    four: YesNo = Field(..., alias="3a_apologies.yes")
    five: YesNo = Field(..., alias="3a_compliment.yes")
    six: YesNo = Field(..., alias="3a_personalthoughts.yes")
    seven: YesNo = Field(..., alias="3a_reassurance.yes")
    eight: YesNo = Field(..., alias="4shareinformation.yes")

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



# --------------------------------------------------------------------------- #
#                       2.  response → structured parser                      #
# --------------------------------------------------------------------------- #


_THINK_SPLIT = re.compile(r"</think>", flags=re.IGNORECASE)
_LINE_RE = re.compile(r"(\d+)\.\s*(yes|no)", flags=re.IGNORECASE)


def parse_llm_response_structured(resp: str) -> Dict[str, int]:
    """
    ① 若出现 </think>，仅取其后的内容；
    ② 解析 1–8 行 "n. yes/no"；
    ③ 使用 Pydantic 校验并返回 0/1 int-dict。
    """
    # (1) strip CoT
    if _THINK_SPLIT.search(resp):
        resp = _THINK_SPLIT.split(resp, maxsplit=1)[-1]

    tmp: Dict[str, str] = {}
    for line in resp.strip().splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            idx, ans = m.group(1), m.group(2).lower()
            tmp[idx] = ans
    if len(tmp) != 8:
        raise ValueError(f"Expected 8 items, got {len(tmp)}")

    # (2) align to field aliases
    data = {
        "1gamemove.yes": tmp["1"],
        "2reasoning.yes": tmp["2"],
        "3rapport.yes": tmp["3"],
        "3a_apologies.yes": tmp["4"],
        "3a_compliment.yes": tmp["5"],
        "3a_personalthoughts.yes": tmp["6"],
        "3a_reassurance.yes": tmp["7"],
        "4shareinformation.yes": tmp["8"],
    }
    model = LLM8dim(**data)
    return model.to_int_dict()


# --------------------------------------------------------------------------- #
#                        3.  LLM wrapper  +  retry parse                      #
# --------------------------------------------------------------------------- #


class Evaluate:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = self._init_client(model_name)

    @staticmethod
    def _init_client(model_name: str) -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        if "llama" in model_name.lower():
            # toy port mapping示例，按需修改
            if "70b" in model_name.lower():
                return OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:9570/v1")
            return OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:3638/v1")
        if "qwen" in model_name.lower():
            return OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:3640/v1")
        return OpenAI(api_key=api_key)

    # -------- basic call -------- #
    def call_llm(self, prompt: str, system_prompt="You are a helpful assistant.", temperature=1) -> str:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(model=self.model_name,
                                                   messages=msgs,
                                                   temperature=temperature)
        return resp.choices[0].message.content.strip()

    # -------- call + parse (with retry) -------- #
    def call_llm_and_parse(self, prompt: str, *, max_retries: int = 1000, temperature: float = 1) -> Tuple[Dict[str, int], str]:
        """
        调用LLM并解析响应，返回解析结果和原始响应
        """
        for attempt in range(max_retries):
            raw = self.call_llm(prompt, temperature=temperature)
            try:
                parsed = parse_llm_response_structured(raw)
                return parsed, raw
            except (ValueError, ValidationError) as e:
                print(f"[Attempt {attempt+1}/{max_retries}] parse failed: {e}. Retrying…")
        raise RuntimeError("Failed to obtain valid 8-dim yes/no response")


# --------------------------------------------------------------------------- #
#                    4.  Consistency metric  (unchanged)                      #
# --------------------------------------------------------------------------- #


def compute_consistency(multi_responses: Dict[float, Sequence[str]]):
    """
    给定 {temperature: [raw_resp, …]} 计算一致性分数。
    """
    cons, details = {}, {}
    for temp, resps in multi_responses.items():
        parsed = []
        for r in resps:
            try:
                parsed.append(parse_llm_response_structured(r))
            except Exception:
                parsed.append(None)

        valid = [p for p in parsed if p]
        if len(valid) < 2:
            cons[temp] = None
            details[temp] = parsed
            continue

        agree = total = 0
        for key in valid[0]:
            answers = [p[key] for p in valid]
            most, freq = Counter(answers).most_common(1)[0]
            agree += freq
            total += len(answers)
        cons[temp] = agree / total if total else None
        details[temp] = parsed
    return cons, details


# --------------------------------------------------------------------------- #
#                        5.  Utility: Load existing results                   #
# --------------------------------------------------------------------------- #


def load_existing_epi_pks(output_file: str) -> Set[str]:
    """
    从已存在的output文件中加载已处理的epi_pk集合
    """
    existing_pks = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        if "epi_pk" in record:
                            existing_pks.add(record["epi_pk"])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load existing results from {output_file}: {e}")
    return existing_pks


# --------------------------------------------------------------------------- #
#                              6.  Main routine                               #
# --------------------------------------------------------------------------- #


def run_process_mode(args, episodes, ip: InstructionPrompt, evaluator: Evaluate):
    # 加载已存在的结果
    existing_pks = load_existing_epi_pks(args.processed_output_file)
    
    # 统计过滤结果
    total_episodes = len(episodes)
    filtered_episodes = [epi for epi in episodes if epi.pk not in existing_pks]
    num_already_processed = len(existing_pks)
    num_to_process = len(filtered_episodes)
    
    print(f"Total episodes for tag '{args.epi_tag}': {total_episodes}")
    print(f"Episodes already processed: {num_already_processed}")
    print(f"Episodes to process: {num_to_process}")
    print(f"Filtered out: {num_already_processed} episodes")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.processed_output_file), exist_ok=True)
    
    num_ok = 0
    processed_epis_in_session = set()  # 追踪本次运行中已处理的episode
    
    with open(args.processed_output_file, "a", encoding="utf-8") as fout:  # 使用 append 模式
        for epi in tqdm(filtered_episodes, desc=f"Main Progress on {args.epi_tag}"):
            # 确保每个episode在本次运行中只处理一次
            if epi.pk in processed_epis_in_session:
                continue
                
            for turn in range(1, len(epi.messages)):
                user_msg = epi.messages[turn][0][2]
                prompt = ip.final_prompt(user_msg)
                try:
                    parsed, raw_response = evaluator.call_llm_and_parse(prompt, max_retries=1000)
                except RuntimeError:
                    continue  # skip
                record = {
                    "epi_pk": epi.pk,
                    'env_pk': epi.environment,
                    "turn_idx": turn - 1,
                    "epi_tag": epi.tag,
                    "epi_messages": user_msg,
                    "llm_raw_response": raw_response,
                    **parsed,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_ok += 1
            
            # 标记该episode已在本次运行中处理
            processed_epis_in_session.add(epi.pk)
    
    print(f"\nTotal processed items: {num_ok}")
    print(f"Episodes processed in this session: {len(processed_epis_in_session)}")


def run_consistency_mode(args, episodes, ip: InstructionPrompt, evaluator: Evaluate):
    # 对于consistency模式，也可以检查已存在的结果
    out_path = args.output_file.replace(".jsonl", "_consistency.jsonl")
    existing_pks = load_existing_epi_pks(out_path)
    
    # 统计过滤结果
    total_episodes = len(episodes)
    filtered_episodes = [epi for epi in episodes if epi.pk not in existing_pks]
    num_already_processed = len(existing_pks)
    num_to_process = len(filtered_episodes)
    
    print(f"Total episodes for tag '{args.epi_tag}': {total_episodes}")
    print(f"Episodes already processed for consistency: {num_already_processed}")
    print(f"Episodes to process for consistency: {num_to_process}")
    print(f"Filtered out: {num_already_processed} episodes")
    
    temps = [float(t) for t in args.temperatures.split(",")]
    all_out = []
    processed_epis_in_session = set()  # 追踪本次运行中已处理的episode
    
    for epi in tqdm(filtered_episodes, desc="Consistency eval"):
        # 确保每个episode在本次运行中只处理一次
        if epi.pk in processed_epis_in_session:
            continue
            
        for turn in range(1, len(epi.messages)):
            prompt = ip.final_prompt(epi.messages[turn][0][2])
            multi = {}
            for t in temps:
                it = trange(args.num_trials, desc=f"Temp {t}", leave=False)
                multi[t] = [evaluator.call_llm(prompt, temperature=t) for _ in it]

            labelled = {str(t): [{"example_id": i + 1, "response": r}
                                 for i, r in enumerate(multi[t])] for t in temps}
            cons, _ = compute_consistency(multi)
            row = epi.to_dict()
            row["llm_multi_responses"] = labelled
            row["llm_consistency"] = {str(k): v for k, v in cons.items()}
            all_out.append(row)
        
        # 标记该episode已在本次运行中处理
        processed_epis_in_session.add(epi.pk)

    # 如果有新结果，追加到文件
    if all_out:
        with open(out_path, "a", encoding="utf-8") as f:  # 使用 append 模式
            for row in all_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    print(f"Consistency results saved to {out_path}")
    print(f"New consistency results: {len(all_out)}")
    print(f"Episodes processed in this session: {len(processed_epis_in_session)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epi_tag", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=False,
                        help="Dummy path when not used")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--processed_output_file", type=str, default="")
    parser.add_argument("--consistency", action="store_true")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--temperatures", type=str, default="0.2,0.5,0.7,1.0")
    args = parser.parse_args()

    # 选择 prompt 模板（如需 baseline 可自行扩展）
    ip = InstructionPrompt()
    evaluator = Evaluate(args.model)

    # 过滤 EpisodeLog
    epis = [EpisodeLog.get(pk) for pk in EpisodeLog.all_pks()
            if EpisodeLog.get(pk).tag == args.epi_tag]

    if args.consistency:
        run_consistency_mode(args, epis, ip, evaluator)
        return

    if args.process:
        if not args.processed_output_file:
            raise ValueError("--processed_output_file must be filled")
        run_process_mode(args, epis, ip, evaluator)
        return

    print("[ERROR] Neither --process nor --consistency specified, nothing to do.")


if __name__ == "__main__":
    main()
