import redis
import random
import sys
import json
sys.path.append("../../")
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
import argparse

def delete_env(tag):
    num = 0
    all_pks = list(EnvironmentProfile.all_pks())
    pks = []
    for pk in all_pks:
        if EnvironmentProfile.get(pk).env_tag == tag:
            pks.append(pk)
    r = redis.Redis(host='localhost', port=6379, db=0)
    env_index = ":sotopia.database.logs.EnvironmentProfile:"
    with r.pipeline() as pipe:
        for key in pks:
            pipe.delete(env_index + key)
            num += 1
        pipe.execute()
    return num



def main():
    parser = argparse.ArgumentParser()
    # [llama3-70b-analysis, gpt-4-analysis, specific_human_anno_llama3_70b, specific_human_anno_gpt_4]
    parser.add_argument("--tag", default="", type=str, required=True, help = "Give a tag for episode choosen")
    args = parser.parse_args()

    delete_num = delete_env(args.tag)
    print(f"Delete Environment Profiles Number: {delete_num}")

if __name__ == '__main__':
    main()