import argparse
import random
from rich import print
import sys
sys.path.append('../')
# print(sys.path)
from tqdm import tqdm
from episode_utils import get_countries_from_agent
from parlai.core.opt import Opt
from parlai.core.agents import create_agent
import json
import re
import os
import pdb
import rich

def int_or_none(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")

def get_intent_agent(dir_path, intent_model_path):
    intent_model_opt = Opt.load_init(intent_model_path)
    intent_model_opt['model_file'] = dir_path + intent_model_opt['model_file']
    intent_model_opt['dict_file'] = dir_path + intent_model_opt['dict_file']
    intent_agent = create_agent(intent_model_opt)
    return intent_agent

def cutoff_dialogue(dialogue, end_turn):
    pattern = r'(\d+)\s+([A-Z]+)\s+->\s+([A-Z]+):\s+(.*?)(?=\n\d+\s+[A-Z]+\s+->\s+[A-Z]+:|$)'
    matches = re.findall(pattern, dialogue, re.DOTALL)
    filtered_turns = [match for match in matches if 0 <= int(match[0]) <= end_turn]
    result = '\n'.join([f"{turn[0]} {turn[1]} -> {turn[2]}: {turn[3].strip()}" for turn in filtered_turns])
    return result

def generate_whole_prompt(episode, country1, country2, end_turn, cut):
    prompt = ""
    prompt += f"{episode['phase_name']}\n"
    if cut:
        cuttoffed_dialogue = cutoff_dialogue(episode['intent_dialogue'], end_turn)
    else:
        cuttoffed_dialogue = episode['intent_dialogue']
    prompt += f"{cuttoffed_dialogue}\n"
    # pdb.set_trace()
    prompt += f"{episode['unit_center']}\n"
    prompt += f"{episode['phase_name']} {country1} 5 ANON 5min WTA two powers for {country2}:"
    return prompt

def get_intent_response(intent_agent, prompt):
    intent_agent.skip_generation = False
    intent_agent.observe(dict(text=prompt, episode_done=False))
    intent_response = intent_agent.act()
    return intent_response

def extract_units_by_country(units_string, country):
    country = country.upper()
    country_units = units_string.split(";")
    for entry in country_units:
        if entry.strip().startswith(country):
            return entry.split(":", 1)[1].strip()
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", default="/data/user_data/wenkail/", type=str, required=False, help="Choose the value model dir")
    parser.add_argument("--res_path", default="data/formatted_episodes/taskeval_1757_intent_episode_llama3.json", type=str, required=False, help="The resource path for file")
    parser.add_argument("--tgt_path", default="data/intent_response/taskeval_1757_intent_prediction_llama3.json", type=str, required=False, help="The target path for file")
    parser.add_argument("--intent_model_path", default="/data/user_data/wenkail/models/imitation_intent.opt", type=str, required=False, help="Choose the env model")
    parser.add_argument("--split_begin", type=int, required=True, help="The begin index of the sub samples")
    parser.add_argument("--split_end", type=int_or_none, required=True, help="The end index of the sub samples")
    parser.add_argument("--end_turn", type=int_or_none, required=False, help = "The end index of the sub dialogue")
    parser.add_argument("--cut", action='store_true', help="Whether use the actual move")

    args = parser.parse_args()

    # TODO: Replace Hard Code country into agents match
    # countries = ["England", "Germany"]
    with open(args.res_path, 'r') as f:
        formatted_episodes = json.load(f)

    # pdb.set_trace()
    intent_agent = get_intent_agent(args.dir_path, args.intent_model_path)

    split_formatted_episodes = formatted_episodes[args.split_begin: args.split_end]

    if not os.path.exists(os.path.dirname(args.tgt_path)):
        os.makedirs(os.path.dirname(args.tgt_path))
        
    with open(args.tgt_path, 'a') as f:
        for episode in tqdm(split_formatted_episodes):
            response = {}
            countries = get_countries_from_agent(episode["agents"])
            response["game_id"] = episode["game_id"]
            response["phase_name"] = episode["phase_name"]
            response["env_uuid"] = episode["env_uuid"]
            response["countries"] = countries
            response[f"{countries[0]}_units"] = extract_units_by_country(episode['unit_center'].split('\n')[0], countries[0])
            response[f"{countries[1]}_units"] = extract_units_by_country(episode['unit_center'].split('\n')[0], countries[1])
            response[f"{countries[0]}_response"] = get_intent_response(intent_agent, generate_whole_prompt(episode, countries[0], countries[1], args.end_turn, args.cut))
            response[f"{countries[1]}_response"] = get_intent_response(intent_agent, generate_whole_prompt(episode, countries[1], countries[0], args.end_turn, args.cut))
            if args.cut:
                response["intent_dialogue"] = cutoff_dialogue(episode['intent_dialogue'], args.end_turn)
            else:
                response["intent_dialogue"] = episode["intent_dialogue"]

            response[f"{countries[0]}_response"].pop("metrics", None)
            response[f"{countries[1]}_response"].pop("metrics", None)

            json.dump(response, f)
            f.write('\n')

    print(f"Additional data successfully appended to {args.tgt_path}")


    
if __name__ == "__main__":
    main()