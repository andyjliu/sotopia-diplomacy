import asyncio
import argparse
from typing import List, Literal, Set
from sotopia.agents import LLMAgent, BaseAgent
from sotopia.database import AgentProfile, EnvironmentProfile, EpisodeLog
from sotopia.envs import ParallelSotopiaEnv
from sotopia.messages import AgentAction, Observation
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator, ReachGoalLLMEvaluator
from sotopia.samplers import EnvAgentCombo
from profile_utils import get_env_pks_by_tag
from sotopia.server import run_async_server
import sys
from tqdm import tqdm

def int_or_none(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")

def get_existing_env_uuids(epi_tag: str) -> Set[str]:
    """
    根据epi_tag获取已经生成好的episode对应的environment UUID集合
    """
    existing_env_uuids = set()
    all_episode_pks = list(EpisodeLog.all_pks())
    for pk in all_episode_pks:
        episode = EpisodeLog.get(pk)
        if episode.tag == epi_tag:
            existing_env_uuids.add(episode.environment)
    return existing_env_uuids

def get_agents(args, countries):
    agents_list = []
    all_character_pks = list(AgentProfile.all_pks())
    for pk in all_character_pks:
        profile = AgentProfile.get(pk)
        if profile.country in countries:
            agent = LLMAgent(
                agent_name=profile.first_name,
                agent_profile=profile,
                model_name=args.model
            )
            agents_list.append(agent)
            if len(agents_list) == 2:
                break
    if len(agents_list) != 2:
        raise ValueError("Two agents are required.")
    return agents_list

def get_env_countries(env):
    profile = EnvironmentProfile.get(env)
    if profile is None:
        raise ValueError(f"Environment not found for ID: {env}")
    return profile.agent_powers

def create_env_agent_combo(env_model: str, env_uuid: str, agents: List[BaseAgent[Observation, AgentAction]],
                           action_order: Literal["simutaneous", "round-robin", "random"] = "round-robin") -> EnvAgentCombo[Observation, AgentAction]:
    env = ParallelSotopiaEnv(
        model_name=env_model,
        evaluators = [RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2)],
        terminal_evaluators=[ReachGoalLLMEvaluator(model_name=env_model)],
        action_order=action_order,
        uuid_str=env_uuid,
    )
    return (env, agents)

async def episode_generation(tag, env_agent_combo_list: List[EnvAgentCombo[Observation, AgentAction]]):
    total = len(env_agent_combo_list)
    for idx, env_agent_combo in enumerate(tqdm(env_agent_combo_list, desc="Generating episodes", unit="episode")):
        try:
            await run_async_server(
                    env_agent_combo_list=[env_agent_combo],
                    omniscient=False,
                    script_like=False,
                    json_in_script=False,
                    tag=tag,
                    push_to_db=True,
                    using_async=True
            )
        except Exception as e:
            print(f"Error generating episode {idx+1}/{total}: {str(e)}", file=sys.stderr, flush=True)
            continue

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games_dir", default="/data/user_data/wenkail/sotopia_diplomacy/clean_global_whole_games", type=str, required=False)
    parser.add_argument("--picked_envs", default="", type=str, required=False)
    parser.add_argument("--model", default="llama3_8b", type=str, required=False)
    parser.add_argument("--agent_model", default="llama3_8b", type=str, required=False)
    parser.add_argument("--env_tag", type=str, required=False)
    parser.add_argument("--epi_tag", type=str, required=True)
    parser.add_argument("--split_begin", type=int, default=0, required=False)
    parser.add_argument("--split_end", type=int_or_none, default=None, required=False)
    args = parser.parse_args()

    # 获取已经生成好的episode对应的environment UUID集合
    existing_env_uuids = get_existing_env_uuids(args.epi_tag)
    print(f"Found {len(existing_env_uuids)} existing episodes for tag '{args.epi_tag}', will skip their environments.")

    if args.picked_envs == "":
        uuid_dict_list = get_env_pks_by_tag(args.env_tag)
        uuid_list = [uuid_dict['uuid'] for uuid_dict in uuid_dict_list][args.split_begin: args.split_end]
    else:
        with open(args.picked_envs, 'r') as f:
            uuid_list = [line.strip() for line in f.readlines()]
    uuid_list = uuid_list[args.split_begin: args.split_end]

    # 过滤掉已经生成过episode的environment
    original_count = len(uuid_list)
    uuid_list = [uuid for uuid in uuid_list if uuid not in existing_env_uuids]
    skipped_count = original_count - len(uuid_list)
    print(f"Skipped {skipped_count} environments that already have episodes generated.")
    print(f"Remaining environments to process: {len(uuid_list)}")

    agents_list = []
    for idx, uuid in enumerate(tqdm(uuid_list, desc="Getting agents", unit="env")):
        try:
            countries = get_env_countries(uuid)
            agents = get_agents(args, countries)
            agents_list.append(agents)
        except Exception as e:
            print(f"[{idx+1}/{len(uuid_list)}] Failed to get agents for env_uuid={uuid}: {e}", file=sys.stderr, flush=True)
            continue

    env_agent_combo_list = []
    model = args.model
    agent_model = args.agent_model
    for i in tqdm(range(len(uuid_list)), desc="Creating env-agent combos", unit="env"):
        try:
            env_agent_combo = create_env_agent_combo(model, uuid_list[i], agents_list[i])
            env_agent_combo_list.append(env_agent_combo)
        except Exception as e:
            print(f"[{i+1}/{len(uuid_list)}] Failed to create env_agent_combo for env_uuid={uuid_list[i]}: {e}", file=sys.stderr, flush=True)
            continue

    print(f"Total episodes to generate: {len(env_agent_combo_list)}", flush=True)
    await episode_generation(args.epi_tag, env_agent_combo_list)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"__main__: Unhandled exception: {e}", file=sys.stderr, flush=True)
        
        
        
# python generate_episode.py --epi_tag llama_8b_finetune_format_v5 --env_tag finetune_format_latest --model llama3_8b --split_begin=0 --split_end None

# python generate_episode.py --epi_tag test --env_tag finetune_format_latest --model llama3_8b --split_begin=0 --split_end 1