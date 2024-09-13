import random
from rich import print
import sys
import json
import pdb
from typing import Any
import argparse
from tqdm import tqdm
sys.path.append("../")
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
from episode_utils import get_game_phase_env_from_episode, get_phases_from_envs, get_actual_dialogue
from episode_utils import (
    process_conversation, 
    format_diplomacy_data, 
    get_game_phase_env_from_episode,
    replace_names_with_countries,
    process_conversation_to_intent
)

def get_agents(countries):
    agents_list = []
    all_character_pks = list(AgentProfile.all_pks())
    for pk in all_character_pks:
        profile = AgentProfile.get(pk)
        for c in countries:
            if profile.country == c:
                agents_list.append(pk)
    return agents_list

def get_intent_dialogue(messages):
    dialogue = ""
    num = 0
    for m in messages:
        dialogue += f"{str(num)} {m['sender']} -> {m['recipient']}: {m['message']}\n"
        num += 1
    return dialogue


def format_episode(envs, phases):
    game_phase_messages = []
    for i in range(len(phases)):
        messages = get_actual_dialogue(envs[i], phases[i])
        if messages != []:
            # pdb.set_trace()
            store = {}
            store['game_id'] = envs[i].game_id
            store['agents'] = get_agents(envs[i].agent_powers)
            store['phase_name'] = phases[i]["name"]
            store['env_uuid'] = envs[i].pk
            
            store['intent_dialogue'] = get_intent_dialogue(messages)
            store['unit_center'] = format_diplomacy_data(envs[i].scenario)
            game_phase_messages.append(store)
    return game_phase_messages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games_dir", default="/data/user_data/wenkail/sotopia_diplomacy/whole_filter_games_100", type=str, required=False, help="Choose the evaluate model, the name can be seen in config")
    # parser.add_argument("--src_epi_tag", type=str, required=True, help = "Choose a tag for access the database")
    parser.add_argument("--env_tag", type=str, required=True, help = "Choose a tag for access the database")
    parser.add_argument("--tgt_path", default="data/formatted_episodes/taskeval_1757_intent_episode_.json", type=str, required=False, help="Store formatted intent episode")
    args = parser.parse_args()

    all_epi_pks = list(EpisodeLog.all_pks())
    all_env_pks = list(EnvironmentProfile.all_pks())
    epis = []
    envs = []

    # Get source Episodes 
    # for pk in all_epi_pks:
    #     epi = EpisodeLog.get(pk)
    #     if epi.tag == args.env_tag:
    #         epis.append(epi)
    # # Get Environment Profiles
    # for epi in epis:
    #     envs.append(get_game_phase_env_from_episode(epi))
    # TODO: Test
    for env_pks in all_env_pks:
        env = EnvironmentProfile.get(env_pks)
        if env.env_tag == args.env_tag:
            envs.append(env)
    
    phases = get_phases_from_envs(args.games_dir, envs)
    intent_dialogue = format_episode(envs, phases)
    print(f"Intent Dialogue Length: {len(intent_dialogue)}")
    with open(args.tgt_path, 'w') as f:
        json.dump(intent_dialogue, f, indent = 4)

if __name__ == "__main__":
    main()