import argparse
import json
import logging
import re
import os

from typing import List, Tuple, Union, Dict
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field, validator
from beartype import beartype

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)


"""
================================================================
1. Define data structures
================================================================
"""

class EvaluationBySocialDimensions(BaseModel):
    ethos: tuple[str, int] = Field(
        ...,
        description=
            "Ethos\n"
            "- Does this utterance convey the speaker's plans? (e.g. I'm attempting to make that deal with Russia now)\n"
            "- Does this utterance convey the speaker's thoughts? (e.g. I think me and England are really on the same page at this point)\n"
            "- Does this utterance convey the speaker's goals? (e.g. I am committed to supporting Munich holding)\n"
            "- Does this utterance share information about other player moves on the board? (e.g., France held out for a long time)\n"
            "- Does this utterance propose a plan of action? (e.g., make sure you don't move Munich so it can take my support)\n"
            "- Does this utterance have a counter-offer? (e.g., Well, are you willing to humor my question about the Aegean, anyway?)\n"
            "- Does this utterance seek clarification? (e.g., are you willing to tell me what your plans are for the Tri unit?)\n\n"
            "Scoring Breakdown (0-10):\n"
            " - Minimal Ethos (0-2): Lacks clarity in conveying plans, thoughts, or goals.\n"
            " - Basic Ethos (3-5): Conveys some plans or thoughts but lacks depth.\n"
            " - Strong Ethos (6-8): Clearly conveys plans, thoughts, and goals with some strategic sharing.\n"
            " - Exemplary Ethos (9-10): Highly effective in conveying plans, thoughts, and goals with strategic clarity.\n\n"
            "Output your reasoning to the string portion, then an integer score (0-10) in the tuple."
        
    )
    logos: tuple[str, int] = Field(
        ...,
        description=
            "Logos\n"
            "- Does this sentence have justification? (e.g., if you took Marseilles, I would be stronger against England)\n"
            "- Does this sentence have an if-then or if-else logic flow? (e.g., if you took Marseilles, I would be stronger against England)\n"
            "- Does this sentence show reflection upon hindsights? (e.g., You could have advised me that supporting Mun-Bur was more important than Kie-Ruh)\n\n"
            "Scoring Breakdown (0-10):\n"
            " - Illogical (0-2): Lacks logical coherence or justification.\n"
            " - Basic Logic (3-5): Some logical structure but with gaps.\n"
            " - Strong Logic (6-8): Well-structured logic with clear justification.\n"
            " - Exemplary Logic (9-10): Exceptionally coherent and justified with robust logical flow.\n\n"
            "Output your reasoning to the string portion, then an integer score (0-10) in the tuple."
        
    )
    pathos: tuple[str, int] = Field(
        ...,
        description=
            "Pathos\n"
            "- Does this utterance share personal information?\n"
            "- Does this utterance reassure the recipient? (e.g., you are my favorite)\n"
            "- Does this utterance compliment or thank the recipient? (e.g., Thanks!)\n"
            "- Does this utterance have an apology? (e.g., Ha! So sorry!!)\n\n"
            "Scoring Breakdown (0-10):\n"
            " - Emotionally Flat (0-2): Lacks emotional engagement or warmth.\n"
            " - Basic Engagement (3-5): Some emotional cues but limited resonance.\n"
            " - Strong Engagement (6-8): Engages emotionally with warmth and empathy.\n"
            " - Exemplary Engagement (9-10): Deeply engaging with profound emotional impact.\n\n"
            "Output your reasoning to the string portion, then an integer score (0-10) in the tuple."
        
    )
    goal: tuple[str, int] = Field(
        ...,
        description="Please first reiterate agent's social goals. "
        "And then please provide a comprehensive analysis about the extent to which the agent has managed to achieve these goals. "
        "In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from 0 and 10 in the 'score' field. 0 represents minimal goals achievement, 10 represents complete goal achievement, and a higher score indicates that the agent is making progress towards their social goals. Almost Not Finishing Any Goal (0-3): Scores from 0 to 3 indicate almost not finishing any goal, suggesting a minimal level of goal achievement. This range signifies either no progress or only a very rudimentary level of advancement towards the completion of set goals. Finishing Less Than 50% of Goals (4-6): A score between 4 and 6 suggests finishing less than 50% of the goals, indicating a moderate level of goal completion. This range represents partial success, with some goals being met while a significant portion remains unachieved. Finishing More Than 50%, But Not All Goals (7-8): Scores in the 7 to 8 range indicate finishing more than 50% but not all of the goals. This suggests a high level of achievement, where the majority of set goals are met, but some goals still remain incomplete. Finishing All Goals (9-10): A score between 9 and 10 signifies finishing all goals, representing the highest level of achievement in goal completion. This range indicates that all set objectives have been met, signifying complete success in achieving the targeted goals.",
    )


    @validator("ethos", "logos", "pathos", "goal")
    def zero_to_ten_validator(cls, v: tuple[str, int]) -> tuple[str, int]:
        if not (0 <= v[1] <= 10):
            raise ValueError("Score must be an integer in the range [0, 10].")
        return v


class EnvResponse(BaseModel):
    """
    The GPT-4 response must strictly match this structure:
    {
      "agent_1_evaluation": {
        "ethos": [<reasoning string>, <score 0-10>],
        "logos": [<reasoning string>, <score 0-10>],
        "pathos": [<reasoning string>, <score 0-10>],
        "goal": [<reasoning string>, <score 0-10>]
      },
      "agent_2_evaluation": {
        ...
      }
    }
    """
    agent_1_evaluation: EvaluationBySocialDimensions
    agent_2_evaluation: EvaluationBySocialDimensions


class Message:
    """Minimal message class."""
    def __init__(self, content: str):
        self.content = content

    def to_natural_language(self) -> str:
        return self.content


class AgentAction(Message):
    """Adds an action_type (e.g. 'talk')."""
    def __init__(self, content: str, action_type: str = "talk"):
        super().__init__(content)
        self.action_type = action_type


"""
================================================================
2. Simple aggregator for EnvResponse -> ScriptEnvironmentResponse
================================================================
"""

class ScriptEnvironmentResponse(BaseModel):
    """
    Our final aggregated structure, after we compute average or store details.
    """
    terminated: bool
    p1_rate: Union[None, Tuple[float, Dict[str, Union[float, int]]]] = None
    p2_rate: Union[None, Tuple[float, Dict[str, Union[float, int]]]] = None
    comments: str = ""


def _reduce(
    responses_per_reducer: List[Tuple[Tuple[str, Union[int, float]], str]]
) -> Tuple[Dict[str, Union[int, float]], str]:
    """
    Convert a list of dimension scores into a single dictionary with an average,
    plus combined textual reasons.
    Example of responses_per_reducer:
    [
      (("ethos", 7), "<ethos> reasoning..."),
      (("logos", 8), "<logos> reasoning..."),
      ...
    ]
    """
    responses_dict = defaultdict(list)
    comments_dict: Dict[str, str] = defaultdict(str)
    reduced_dict: Dict[str, Union[int, float]] = {}

    for (dimension_name, dimension_value), reasoning in responses_per_reducer:
        responses_dict[dimension_name].append(dimension_value)
        comments_dict[dimension_name] += reasoning + "\n"

    for dim_name, values in responses_dict.items():
        if len(values) > 0 and all(isinstance(v, (int, float)) for v in values):
            reduced_dict[dim_name] = sum(values) / len(values)
        else:
            reduced_dict[dim_name] = 0

    # compute overall_score
    numeric_for_overall = [
        v for v in reduced_dict.values()
        if isinstance(v, (int, float))
    ]
    if numeric_for_overall:
        reduced_dict["overall_score"] = sum(numeric_for_overall) / len(numeric_for_overall)
    else:
        reduced_dict["overall_score"] = 0

    # combine textual comments
    combined_comments = ""
    for k, v in comments_dict.items():
        combined_comments += f"{k}:\n{v}\n"

    return reduced_dict, combined_comments


def unweighted_aggregate_evaluate(
    env_response: EnvResponse
) -> ScriptEnvironmentResponse:
    """
    Convert an EnvResponse into the standard aggregator output format:
    [
      ("agent_1", (("ethos", #), "reasoning...")),
      ...
    ]
    Then do a simple aggregator.
    """
    # build the tuples
    result_list: List[Tuple[str, Tuple[Tuple[str, Union[int, float]], str]]] = []

    # agent_1
    a1 = env_response.agent_1_evaluation
    result_list.append(("agent_1", (("ethos", a1.ethos[1]), a1.ethos[0])))
    result_list.append(("agent_1", (("logos", a1.logos[1]), a1.logos[0])))
    result_list.append(("agent_1", (("pathos", a1.pathos[1]), a1.pathos[0])))
    result_list.append(("agent_1", (("goal", a1.goal[1]), a1.goal[0])))

    # agent_2
    a2 = env_response.agent_2_evaluation
    result_list.append(("agent_2", (("ethos", a2.ethos[1]), a2.ethos[0])))
    result_list.append(("agent_2", (("logos", a2.logos[1]), a2.logos[0])))
    result_list.append(("agent_2", (("pathos", a2.pathos[1]), a2.pathos[0])))
    result_list.append(("agent_2", (("goal", a2.goal[1]), a2.goal[0])))

    # now group by agent
    grouped = defaultdict(list)
    for who, data in result_list:
        grouped[who].append(data)

    # reduce
    agent_1_reduced = _reduce(grouped["agent_1"]) if "agent_1" in grouped else ({}, "")
    agent_2_reduced = _reduce(grouped["agent_2"]) if "agent_2" in grouped else ({}, "")

    # build final
    p1_rate, c1 = None, ""
    if agent_1_reduced[0]:
        p1_dict = dict(agent_1_reduced[0])
        p1_overall = p1_dict.pop("overall_score", 0)
        p1_rate = (p1_overall, p1_dict)
        c1 = agent_1_reduced[1]

    p2_rate, c2 = None, ""
    if agent_2_reduced[0]:
        p2_dict = dict(agent_2_reduced[0])
        p2_overall = p2_dict.pop("overall_score", 0)
        p2_rate = (p2_overall, p2_dict)
        c2 = agent_2_reduced[1]

    comments = f"Agent1:\n{c1}\nAgent2:\n{c2}"
    return ScriptEnvironmentResponse(
        terminated=False,  # no rule-based termination here
        p1_rate=p1_rate,
        p2_rate=p2_rate,
        comments=comments
    )


"""
================================================================
3. GPT-4 Evaluator that uses openai.ChatCompletion.create()
   and PydanticOutputParser
================================================================
"""

class GPT4EnvEvaluator:
    """
    Calls GPT-4 directly using openai.ChatCompletion.create(), 
    passing a system + user message. Then we parse using LangChain’s
    PydanticOutputParser to ensure the JSON matches EnvResponse.
    """

    def __init__(self, openai_api_key: str, model_name: str = "gpt-4"):
        self.client = OpenAI()
        self.model_name = model_name
        self.parser = PydanticOutputParser(pydantic_object=EnvResponse)

        self.system_template = (
            "You are an evaluator that must produce JSON strictly matching this pydantic schema:\n\n"
            "{schema}\n\n"
            "Do not add any extra keys. No additional commentary. Only valid JSON.\n"
        )

    def evaluate(self, conversation: str) -> EnvResponse:
        """
        Synchronously call GPT-4 with the conversation text, 
        parse the response into EnvResponse using PydanticOutputParser.
        """
        # Construct the system and user messages
        system_prompt = self.system_template.format(
            schema=self.parser.get_format_instructions()
        )
        user_prompt = f"Conversation:\n{conversation}\n\nPlease provide your evaluation."

        # Build the message list
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call GPT-4
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=1.0
            )
            assistant_content = response.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"OpenAI API call failed: {e}")
            raise ValueError("Failed to call OpenAI GPT-4 API.") from e

        # Parse the GPT output using PydanticOutputParser
        try:
            env_response_obj = self.parser.parse(assistant_content)
            return env_response_obj
        except Exception as e:
            log.error(f"Parsing error: {e}")
            raise ValueError("Failed to parse GPT output into EnvResponse.") from e


"""
================================================================
4. Utility: format_messages
   Convert multiline text into a list of (agent, AgentAction)
================================================================
"""

def format_messages(raw_data: str) -> List[Tuple[str, AgentAction]]:
    """
    Assumes lines like:
      1  Alice -> Bob: Hello, how are you?
    We capture the sender + content.
    The first discovered sender => agent_1, the next => agent_2, etc.
    """
    messages_format = []
    mapping = {}
    pattern = r"^\d+\s+(\S+)\s+->\s+(\S+):\s+(.*)$"
    for line in raw_data.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            sender, receiver, message = match.groups()
            if not mapping:
                # first discovered => agent_1, second => agent_2
                mapping[sender] = "agent_1"
                mapping[receiver] = "agent_2"
            if sender not in mapping:
                mapping[sender] = f"agent_{len(mapping)+1}"
            messages_format.append((mapping[sender], AgentAction(message)))
    return messages_format


"""
================================================================
5. Main script
================================================================
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file (jsonl).")
    parser.add_argument("--envs_file", type=str, required=False, help="Path to envs file.")
    parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI model name (e.g. gpt-4).")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set OPENAI_API_KEY in environment variables.")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.envs_file:
        with open(args.envs_file, "r", encoding="utf-8") as f:
            envs = f.read().splitlines()
            
    match_data = [i for i in data if i['env_uuid'] in envs]
    
    # 如果 input JSON 就是一个list，那我们逐条处理:
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input JSON must be a non-empty list.")

    # 初始化 evaluator
    evaluator = GPT4EnvEvaluator(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name=args.model
    )

    # 打开输出文件（.jsonl格式），准备逐条写出
    with open(args.output, "w", encoding="utf-8") as fout:
        # 遍历每条数据
        for item in tqdm(match_data):
            raw_dialogue = item.get("intent_dialogue", "")
            if not raw_dialogue:
                # 如果没对话文本，可根据需求选择跳过或写出空结果
                continue

            # 解析出 (agent_1, AgentAction(...)) 的列表，以及一个 sender->agent_1/2 的映射
            messages = []
            mapping = {}
            pattern = r"^\d+\s+(\S+)\s+->\s+(\S+):\s+(.*)$"
            
            for line in raw_dialogue.splitlines():
                line = line.strip()
                if not line:
                    continue
                match = re.match(pattern, line)
                if match:
                    sender, receiver, msg = match.groups()
                    # 动态建立 sender/receiver => agent_1/agent_2 的映射
                    if sender not in mapping:
                        mapping[sender] = f"agent_{len(mapping)+1}"
                    if receiver not in mapping:
                        mapping[receiver] = f"agent_{len(mapping)+1}"
                    messages.append((mapping[sender], AgentAction(msg)))
            # 把对话重新拼成给 GPT-4 评估的字符串
            conversation_text = ""
            for who, msg in messages:
                conversation_text += f"{who}: {msg.to_natural_language()}\n"

            # 评估
            env_response = evaluator.evaluate(conversation_text)
            final_result = unweighted_aggregate_evaluate(env_response)

            # ---------------------------------------------------
            # 把 aggregator 输出结果改成以原始sender名字为key
            # 先获取 mapping 的反向映射: agent_1 -> 原始名字
            # 如果只有两个角色，则 mapping 里通常是 {'Alice': 'agent_1', 'Bob': 'agent_2'} 这种
            # 我们反转后: {'agent_1': 'Alice', 'agent_2': 'Bob'}
            # 注: 如果不需要改成原始sender名称，可跳过本步
            reverse_mapping = {v: k for k, v in mapping.items()}

            # 将 aggregator 返回的 p1_rate, p2_rate 分别装进新的 dict
            agent_results = {}
            if final_result.p1_rate is not None:
                # final_result.p1_rate 是 (overall_score, { 'ethos':..., 'logos':..., 'pathos':... })
                agent_name_1 = reverse_mapping.get("agent_1", "agent_1")
                agent_results[agent_name_1] = {
                    "overall_score": final_result.p1_rate[0],
                    "detailed_scores": final_result.p1_rate[1],  # ethos, logos, pathos, goal
                }
            if final_result.p2_rate is not None:
                agent_name_2 = reverse_mapping.get("agent_2", "agent_2")
                agent_results[agent_name_2] = {
                    "overall_score": final_result.p2_rate[0],
                    "detailed_scores": final_result.p2_rate[1],
                }

            # 你也可以把 final_result.comments 放进去
            # agent_results["comments"] = final_result.comments

            # ---------------------------------------------------
            # 把原始条目的所有字段 + 新的评估结果合并，做成一个完整输出
            merged_output = {
                **item,              # 原始数据
                "evaluation": agent_results  # 新增评估结果
            }

            # ---------------------------------------------------
            # 写入 .jsonl 文件，一条JSON占一行
            fout.write(json.dumps(merged_output, ensure_ascii=False) + "\n")

    print(f"Evaluation complete. Results written to {args.output}")
    
    
if __name__ == "__main__":
    main()
    
    
# python negoeval_for_actual.py --input data/intent_value/taskeval_negoeval_whole_finetune/intent_human_dialogue.json --model gpt-4o-mini --output data/intent_value/taskeval_negoeval_whole_finetune/intent_human_dialogue_negoeval.jsonl --envs_file data/intent_value/taskeval_negoeval_whole_finetune/envs_matched.txt