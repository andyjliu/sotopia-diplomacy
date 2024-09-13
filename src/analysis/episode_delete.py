import redis
import random
import sys
import json
sys.path.append("../../")
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
import argparse

def delete_episode(tag):
    num = 0
    all_pks = list(EpisodeLog.all_pks())
    pks = []
    for pk in all_pks:
        if EpisodeLog.get(pk).tag == tag:
            pks.append(pk)
    r = redis.Redis(host='localhost', port=6379, db=0)
    episode_index = ":sotopia.database.logs.EpisodeLog:"
    with r.pipeline() as pipe:
        for key in pks:
            num += 1
            pipe.delete(episode_index + key)
        pipe.execute()
    print(f"Delete Episode Num: {num}")

def main():
    parser = argparse.ArgumentParser()
    # [llama3-70b-analysis, gpt-4-analysis, specific_human_anno_llama3_70b, specific_human_anno_gpt_4]
    parser.add_argument("--tag", default="", type=str, required=True, help = "Give a tag for episode choosen")
    args = parser.parse_args()

    delete_episode(args.tag)

if __name__ == '__main__':
    main()