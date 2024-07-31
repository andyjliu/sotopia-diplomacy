import random
import sys
import json
sys.path.append("../../")
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
import argparse

def get_episodes(tag):
    epis = []
    all_pks = list(EpisodeLog.all_pks())
    for pk in all_pks:
        if EpisodeLog.get(pk).tag == tag:
            epis.append(EpisodeLog.get(pk))
    return epis

def get_episode_dict(epi):
    episode = {}
    episode['pk'] = epi.pk
    episode['environment'] = epi.environment
    episode['agents'] = epi.agents
    episode['tag'] = epi.tag
    episode['models'] = epi.models
    # TODO: Add message 
    episode['messages'] = []
    messages = epi.messages
    for i in range(len(messages)):
        if i == 0:
            episode['messages'].append(messages[i][0][2])
            episode['messages'].append(messages[i][1][2])
        else:
            episode['messages'].append(messages[i][0][2])
    episode['reasoning'] = epi.reasoning
    episode['rewards'] = epi.rewards
    episode['rewards_prompt'] = epi.rewards_prompt
    return episode

def main():
    parser = argparse.ArgumentParser()
    # tag: [gpt-4-analysis, llama3-70b-analysis]
    parser.add_argument("--tag", default="", type=str, required=True, help = "Give a tag for episode choosen")
    parser.add_argument("--out_file", default="", type=str, required=True, help = "Give a output file path")
    args = parser.parse_args()
    epis = get_episodes(args.tag)
    epis_dict_list = []
    for epi in epis:
        epis_dict_list.append(get_episode_dict(epi))
    
    with open(args.out_file, 'w') as f:
        json.dump(epis_dict_list, f, indent=4)

if __name__ == '__main__':
    main()