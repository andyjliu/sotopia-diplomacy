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
import pdb

def get_episodes(tag):
    all_task_pks = list(EpisodeLog.all_pks())
    episodelogs = []
    for pk in all_task_pks:
        episo = EpisodeLog.get(pk)
        if episo.tag == tag:
            # pdb.set_trace()
            env = get_game_phase_env_from_episode(episo)
            episodelogs.append({"game_id": env.game_id, "agents": episo.agents, "phase_name": env.phase_name, "env_uuid": env.pk, "env": env, "episode": episo})
    return episodelogs

def format_episode(episodes):
    new_episodes = []
    agent_profiles = []
    all_character_pks = list(AgentProfile.all_pks())
    for pk in all_character_pks:
        agent_profiles.append(AgentProfile.get(pk))
    for episode in episodes:
        # pdb.set_trace()
        episode['dialogue'] = replace_names_with_countries(process_conversation(episode["episode"].messages), agent_profiles)
        episode['intent_dialogue'] = process_conversation_to_intent(episode['dialogue'])
        episode['unit_center'] = format_diplomacy_data(episode["env"].scenario)
        new_episodes.append(episode)
    return new_episodes


    

def main():
    valid_countries = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", default="/data/user_data/wenkail/", type=str, required=False, help="Choose the value model dir")
    parser.add_argument("--tag", type=str, required=True, help = "Choose a tag for access the database")
    parser.add_argument("--tgt_path", default="data/formatted_episodes/taskeval_1757_intent_episode_llama3.json", type=str, required=False, help="Store formatted intent episode")
    args = parser.parse_args()

    episodes = get_episodes(args.tag)
    formatted_episodes = format_episode(episodes)
    intent_episodes = []
    # pdb.set_trace()
    for formatted_episode in formatted_episodes:
        formatted_episode.pop('episode')
        formatted_episode.pop('env')
        formatted_episode.pop('dialogue')
        intent_episodes.append(formatted_episode)

    with open(args.tgt_path, 'w') as f:
        json.dump(formatted_episodes, f, indent = 4)


if __name__ == "__main__":
    main()