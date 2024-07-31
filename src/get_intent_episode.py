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
        episode['dialogue'] = replace_names_with_countries(process_conversation(episode["episode"].messages), agent_profiles)
        episode['intent_dialogue'] = process_conversation_to_intent(episode['dialogue'])
        episode['unit_center'] = format_diplomacy_data(episode["env"].scenario)
        new_episodes.append(episode)
    return new_episodes


    

def main():
    valid_countries = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", default="/data/user_data/wenkail/", type=str, required=False, help="Choose the value model dir")
    parser.add_argument("--intent_model_path", default="/data/user_data/wenkail/models/imitation_intent.opt", type=str, required=False, help="Choose the env model")
    # parser.add_argument("--c1", type = str, choices = valid_countries, required=True, help="Choose the first country")
    # parser.add_argument("--c2", type = str, choices = valid_countries, required=True, help="Choose the second country")
    parser.add_argument("--tag", type=str, required=True, help = "Choose a tag for access the database")
    parser.add_argument("--tgt_dir", default="data/formatted_episodes/", type=str, required=False, help="Choose the env model")
    args = parser.parse_args()

    # TODO: replace c1 and c2 into the episode content into the real angent by getting its index
    # countries = [c1, c2]

    episodes = get_episodes(args.tag)
    formatted_episodes = format_episode(episodes)
    intent_episodes = []
    # pdb.set_trace()
    for formatted_episode in formatted_episodes:
        formatted_episode.pop('episode')
        formatted_episode.pop('env')
        intent_episodes.append(formatted_episode)
    with open(f'{args.tgt_dir}formatted_episodes_for_{args.tag}.json', 'w') as f:
        json.dump(formatted_episodes, f)


if __name__ == "__main__":
    main()