import pandas as pd
import numpy as np
import rich
import sys
import os
sys.path.append("/home/wenkail/diplomacy/diplomacy_cicero")
sys.path.append("../")
from parlai.core.opt import Opt
from parlai.core.agents import create_agent
import re
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy import pydipcc 
import json
import inspect
import pdb
import argparse
from evaluate_utils import (
    get_intent_agent,
    get_pydipcc_game
)
from tqdm import tqdm

def get_country(c):
    country_maps = {'A': 'AUSTRIA', 'E': 'ENGLAND', 'F': 'FRANCE', 'G': 'GERMANY', 'I': 'ITALY', 'R': 'RUSSIA', 'T': 'TURKEY'}
    return country_maps[c]

def message_sample(phase, c1, c2):
    stored_message=[]
    for m in phase["messages"]:
        if m['sender'] == get_country(c1) and m['recipient'] == get_country(c2):
            stored_message.append(m)
    return stored_message

def get_dialogue(messages):
    dialogue = ""
    for i in range(len(messages)):
        dialogue += f"{i} {messages[i]['sender']} -> {messages[i]['sender']}: {messages[i]['message']}\n"
    return dialogue

def get_format_unit_center(units):
    prompt = ""
    for key in units:
        prompt += f"{key}: "
        for u in units[key]:
            prompt += f"{u}, "
        prompt = prompt[:-2]
        prompt += "; "
    return prompt

def get_unit_center(phase):
    unit_center = "units: "
    unit_center += get_format_unit_center(phase['state']['units'])
    unit_center += "\n"
    unit_center += "centers: "
    unit_center += get_format_unit_center(phase['state']['centers'])
    return unit_center

def generate_whole_prompt(phase_name, dialogue, unit_center, country1, country2):
    prompt = ""
    prompt += f"{phase_name}\n"
    prompt += f"{dialogue}\n"
    prompt += f"{unit_center}\n"
    prompt += f"{phase_name} {country1} 5 ANON 5min WTA two powers for {country2}:"
    return prompt

def phase_whole_prompt(game, phase_name, c1, c2):
    for phase in game["phases"]:
        if phase['name'] == phase_name:
            current_phase = phase
            # print("Current phase found")
    messages = message_sample(current_phase, c1, c2)
    # Get the input dialogue for intent model
    dialogue = get_dialogue(messages)
    unit_center = get_unit_center(current_phase)
    country1 = get_country(c1)
    country2 = get_country(c2)
    whole_prompt = generate_whole_prompt(phase_name, dialogue, unit_center, country1, country2)
    return whole_prompt

def get_intent_response(intent_agent, prompt):
    intent_agent.skip_generation = False
    intent_agent.observe(dict(text=prompt, episode_done=False))
    intent_response = intent_agent.act()
    # pdb.set_trace()
    return intent_response['text']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", default="/data/user_data/wenkail/", type=str, required=False, help="Choose the value model dir")
    parser.add_argument("--intent_model_path", default="/data/user_data/wenkail/models/imitation_intent.opt", type=str, required=False, help="Choose the env model")
    parser.add_argument("--game_file_path", default="/data/user_data/wenkail/sotopia_diplomacy/games/53480.json", type=str, required=False, help="Choose the game file")
    parser.add_argument("--human_annotation", default="manual_annotations.csv", type=str, required=False, help="Choose the human annotation file")
    parser.add_argument("--tgt_path", default="../data/analysis/human_annotation_intent_prediction.json", type=str, required=False, help="Choose the intent response")
    args = parser.parse_args()


    # Given Phase name here
    # TODO: Remove it from csv file
    game_name = "53480"
    intent_agent= get_intent_agent(args.dir_path, args.intent_model_path)
    with open(args.game_file_path, 'r') as f:
        game = json.load(f)
    
    # ha = pd.read_csv(args.human_annotation)
    ha = pd.read_csv(args.human_annotation)
    phases_intent = []
    for row in tqdm(ha.itertuples()):
        intent = {}
        intent['game'] = row[1]
        intent['phase'] = row[2]
        intent['power1'] = row[3]
        intent['power2'] = row[4]
        intent['actual_order'] = row[5]
        intent_prompt_p1 = phase_whole_prompt(game, row[2], row[3], row[4])
        intent_prompt_p2 = phase_whole_prompt(game, row[2], row[4], row[3])
        intent['predict_order_p1'] = get_intent_response(intent_agent, intent_prompt_p1)
        intent['predict_order_p2'] = get_intent_response(intent_agent, intent_prompt_p2)
        phases_intent.append(intent)

    json.dump(phases_intent, open(args.tgt_path, 'w'),indent=4)


if __name__ == '__main__':
    main()



