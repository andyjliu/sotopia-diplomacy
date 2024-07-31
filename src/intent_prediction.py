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
import pdb


def get_intent_agent(dir_path, intent_model_path):
    intent_model_opt = Opt.load_init(intent_model_path)
    intent_model_opt['model_file'] = dir_path + intent_model_opt['model_file']
    intent_model_opt['dict_file'] = dir_path + intent_model_opt['dict_file']
    intent_agent = create_agent(intent_model_opt)
    return intent_agent


def get_intent_agent(dir_path, intent_model_path):
    intent_model_opt = Opt.load_init(intent_model_path)
    intent_model_opt['model_file'] = dir_path + intent_model_opt['model_file']
    intent_model_opt['dict_file'] = dir_path + intent_model_opt['dict_file']
    intent_agent = create_agent(intent_model_opt)
    return intent_agent

def generate_whole_prompt(episode, country1, country2):
    prompt = ""
    prompt += f"{episode['phase_name']}\n"
    prompt += f"{episode['intent_dialogue']}\n"
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
# units_string = 'AUSTRIA: A SER, A BUD, F GRE, A VIE, F NAP; ENGLAND: F LON, F NTH, A HOL, F IRI; FRANCE: F BRE, A SPA, A PAR, A BUR; GERMANY: A DEN, A BEL, A SIL, A RUH, F KIE; ITALY: A VEN, F MAO, F LYO, A MAR; RUSSIA: A WAR, F SWE, A NWY, A GAL, F BLA, A RUM, A UKR; TURKEY: A BUL, F CON, A ANK, F EAS'
# country = 'ITALY'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", default="/data/user_data/wenkail/", type=str, required=False, help="Choose the value model dir")
    parser.add_argument("--res_path", default="data/formatted_episodes/formatted_episodes_for_random_sample_100_games.json", type=str, required=False, help="Choose the intent response")
    parser.add_argument("--tgt_path", default="data/intent_response/intent_response_random_sample_100_games.json", type=str, required=False, help="Choose the intent response")
    parser.add_argument("--intent_model_path", default="/data/user_data/wenkail/models/imitation_intent.opt", type=str, required=False, help="Choose the env model")

    args = parser.parse_args()

    # TODO: Replace Hard Code country into agents match
    # countries = ["England", "Germany"]
    with open(args.res_path, 'r') as f:
        formatted_episodes = json.load(f)
    # pdb.set_trace()

    intent_responses = []
    intent_agent = get_intent_agent(args.dir_path, args.intent_model_path)

    for episode in tqdm(formatted_episodes):
        response = {}
        countries = get_countries_from_agent(episode["agents"])
        response["game_id"] = episode["game_id"]
        response["phase_name"] = episode["phase_name"]
        response["env_uuid"] = episode["env_uuid"]
        response["countries"] = countries
        response[f"{countries[0]}_units"] = extract_units_by_country(episode['unit_center'].split('\n')[0], countries[0])
        response[f"{countries[1]}_units"] = extract_units_by_country(episode['unit_center'].split('\n')[0], countries[1])
        response[f"{countries[0]}_response"] = get_intent_response(intent_agent,generate_whole_prompt(episode, countries[0], countries[1]))
        response[f"{countries[1]}_response"] = get_intent_response(intent_agent,generate_whole_prompt(episode, countries[1], countries[0]))
        response[f"{countries[0]}_response"].pop("metrics")
        response[f"{countries[1]}_response"].pop("metrics")
        intent_responses.append(response)
    
    json.dump(intent_responses, open(args.tgt_path, 'w'))

    
if __name__ == "__main__":
    main()