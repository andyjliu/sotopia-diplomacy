import argparse
import random
from rich import print
import sys
# sys.path.append('../')
# print(sys.path)
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
from episode_utils import (
    process_conversation, 
    format_diplomacy_data, 
    get_game_phase_env_from_episode,
    replace_names_with_countries,
    process_conversation_to_intent
)
import json

def generate_whole_prompt(episode, country1, country2):
    prompt = ""
    prompt += f"{episode['env'].phase_name}\n"
    prompt += f"{episode['intent_dialogue']}\n"
    prompt += f"{episode['unit_center']}\n"
    prompt += f"{episode['env'].phase_name} {country1} 5 ANON 5min WTA two powers for {country2}:"
    return prompt

def get_intent_response(intent_agent, prompt):
    intent_agent.skip_generation = False
    intent_agent.observe(dict(text=prompt, episode_done=False))
    intent_response = intent_agent.act()
    return intent_response

    # intent_responses = []
    # for episode in formatted_episodes:
    #     response = {}
    #     response[countries[0]] = get_intent_response(intent_agent,generate_whole_prompt(episode, countries[0], countries[1]))
    #     response[countries[1]] = get_intent_response(intent_agent,generate_whole_prompt(episode, countries[1], countries[0]))
    #     intent_responses.append(response)
    # print(intent_responses)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", default="/data/user_data/wenkail/", type=str, required=False, help="Choose the value model dir")
    parser.add_argument("--intent_model_path", default="/data/user_data/wenkail/models/imitation_intent.opt", type=str, required=False, help="Choose the env model")

    args = parser.parse_args()
    countries = ["England", "Germany"]

    with open('data/formatted_episodes_for_intent.json', 'r') as f:
        formatted_episodes = json.load(f)
        
    intent_responses = []
    for episode in formatted_episodes:
        response = {}
        response[countries[0]] = get_intent_response(intent_agent,generate_whole_prompt(episode, countries[0], countries[1]))
        response[countries[1]] = get_intent_response(intent_agent,generate_whole_prompt(episode, countries[1], countries[0]))
        intent_responses.append(response)
    print(intent_responses)

    
if __name__ == "__main__":
    main()