import argparse
import random
from rich import print
import sys
# sys.path.append('../')
# print(sys.path)
from tqdm import tqdm
from evaluate_utils import (
    get_intent_agent,
)
import json

def generate_whole_prompt_without_dialogue(episode, country1, country2):
    prompt = ""
    prompt += f"{episode['phase_name']}\n"
    # prompt += f"{episode['intent_dialogue']}\n"
    prompt += f"{episode['unit_center']}\n"
    prompt += f"{episode['phase_name']} {country1} 5 ANON 5min WTA two powers for {country2}:"
    return prompt

def get_intent_response(intent_agent, prompt):
    intent_agent.skip_generation = False
    intent_agent.observe(dict(text=prompt, episode_done=False))
    intent_response = intent_agent.act()
    return intent_response

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_file", default= "data/formatted_whole_episodes_for_intent.json", type=str, required=False, help="The resource file")
    parser.add_argument("--target_file", default= "data/intent_responses_whole_without_dialogue.json", type=str, required=False, help="The target file")
    parser.add_argument("--dir_path", default="/data/user_data/wenkail/", type=str, required=False, help="Choose the value model dir")
    parser.add_argument("--intent_model_path", default="/data/user_data/wenkail/models/imitation_intent.opt", type=str, required=False, help="Choose the env model")

    args = parser.parse_args()
    countries = ["England", "Germany"]
    intent_agent = get_intent_agent(args.dir_path, args.intent_model_path)
    with open(args.res_file, 'r') as f:
        formatted_episodes = json.load(f)
        
    intent_responses = []
    for episode in tqdm(formatted_episodes):
        response = {}
        response["game_id"] = episode["game_id"]
        response["phase_name"] = episode["phase_name"]
        response["env_uuid"] = episode["env_uuid"]
        response[countries[0]] = get_intent_response(intent_agent,generate_whole_prompt_without_dialogue(episode, countries[0], countries[1]))
        response[countries[1]] = get_intent_response(intent_agent,generate_whole_prompt_without_dialogue(episode, countries[1], countries[0]))
        response[countries[0]].pop("metrics")
        response[countries[1]].pop("metrics")
        intent_responses.append(response)
    
    json.dump(intent_responses, open(args.target_file, 'w'))

    
if __name__ == "__main__":
    main()