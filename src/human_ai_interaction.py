import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import cast, Literal, List, Tuple
from sotopia.agents import Agents, LLMAgent, HumanAgent, RedisAgent, ScriptWritingAgent, BaseAgent
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
from sotopia.envs import ParallelSotopiaEnv
from sotopia.messages import AgentAction, Observation
from sotopia.messages.message_classes import ScriptBackground
from sotopia.generation_utils.generate import LLM_Name, agenerate_action, agenerate_goal, agenerate_script
from sotopia.samplers import EnvAgentCombo
from profile_utils import adjacency_selection, get_env_pks, random_country_adjacency_selection, get_env_pk, get_env_pks_by_tag
from tqdm import tqdm
import argparse
from sotopia.server import run_async_server
import logging
from logging.handlers import RotatingFileHandler
from sotopia.envs.evaluators import (
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    unweighted_aggregate_evaluate,
)
import os
import pdb
import random
import warnings
# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*PEP 484 type hint.*"
)

# Suppress all warnings from `beartype`
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="beartype.*"
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*`LLMChain` was deprecated.*"
)

def int_or_none(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")

def get_agents(args, countries, is_human):
    agents_list = []
    all_character_pks = list(AgentProfile.all_pks())
    c1, c2 = countries 
    for pk in all_character_pks:
        profile = AgentProfile.get(pk)
        if profile.country == c1:
            agent = LLMAgent(
                agent_name=profile.first_name,
                agent_profile=profile,
                model_name=args.agent_model
            )
            agents_list.append(agent)
        
        if profile.country == c2:
            agent = HumanAgent(
                agent_name="Human",
                agent_profile=profile,
                uuid_str=profile.pk,
                frozen_action = 'speak'
            )
            agents_list.append(agent)
        if len(agents_list) == 2:  # Ensure we only get two agents
            break

    if len(agents_list) != 2:
        raise ValueError("Two agents are required.")
    return agents_list

def get_env_countries(env):
    try:
        profile = EnvironmentProfile.get(env)
        if profile is None:
            raise ValueError(f"Environment not found for ID: {env}")
        return profile.agent_powers
    except Exception as e:
        raise RuntimeError(f"Error accessing environment with ID: {env}. Original error: {str(e)}")

def create_env_agent_combo(env_model: str, agent_model: str, env_uuid: str, agents: List[BaseAgent[Observation, AgentAction]],action_order: Literal["simutaneous", "round-robin", "random"] = "round-robin") -> EnvAgentCombo[Observation, AgentAction]:
    env = ParallelSotopiaEnv(
        model_name=env_model,
        action_order=action_order,
        evaluators=[
            RuleBasedTerminatedEvaluator()
        ],
        terminal_evaluators=[
            ReachGoalLLMEvaluator(env_model),
        ],
        uuid_str=env_uuid,
    )
    return (env, agents)

def get_env(uuids, env_tag, game_id, phase_name, countries):
    for uuid in uuids:
        env = EnvironmentProfile.get(uuid)
        if env.game_id == game_id and env.phase_name == phase_name and set(env.agent_powers) == set(countries) and env.env_tag == env_tag:
            return env.pk
    return None

async def episode_generation(tag, env_agent_combo_list: List[EnvAgentCombo[Observation, AgentAction]], frozen_action = None):
    
    for env_agent_combo in env_agent_combo_list:
        try:
            # import pdb; pdb.set_trace()
            logging.info("Starting episode generation")
            messages = await run_async_server(
                env_agent_combo_list=[env_agent_combo],
                omniscient=False,
                script_like=False,
                json_in_script=False,
                # tag="whole", # First generation results
                tag=tag,
                push_to_db=True,
                using_async=True
            )
            logging.info(f"Finished a run_async_server")
        except Exception as e:
            logging.exception(f"An error occurred: {e}")
            
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_dir", default="/data/user_data/wenkail/", type=str, required=False, help="Choose the value model dir")
    parser.add_argument("--env_model", default="gpt-4o", type=str, required=False, help="The env model")
    parser.add_argument("--agent_model", default="gpt-4o", type=str, required=False, help="The agent model")
    parser.add_argument("--env_tag", type=str, required=False, help="The env tag")
    parser.add_argument("--game_id", type=str, required=False, help="The game id")
    parser.add_argument("--c1", type=str, required=False, help="The first country")
    parser.add_argument("--c2", type=str, required=False, help="The second country")
    parser.add_argument("--phase_name", type=str, required=False, help="The phase name")
    parser.add_argument("--epi_tag", type=str, required=False, help="The episode tag")
    parser.add_argument("--human", action='store_true', help="Whether is the human interaction")
    args = parser.parse_args()
    uuid_dict_list = get_env_pks_by_tag(args.env_tag)
    previous_uuid_list = [uuid_dict['uuid'] for uuid_dict in uuid_dict_list]
    uuid_list = [get_env(uuids=previous_uuid_list, env_tag=args.env_tag, game_id=args.game_id, phase_name=args.phase_name, countries=[args.c1, args.c2])]
    agents_list = [get_agents(args,get_env_countries(env=uuid), args.human) for uuid in uuid_list]
    env_agent_combo_list = []
    
    # print("Choose uuids count: ", len(uuid_list))
    for i in range(len(uuid_list)):
        env_agent_combo_list.append(create_env_agent_combo(args.env_model, args.agent_model, uuid_list[i], agents_list[i]))
        
    # TODO: For human interation, doesn't print the agent's response in terminal, should be fixed
    print("Start The Conversation: ")
    print("Leave your messages: ")
    await episode_generation(args.epi_tag, env_agent_combo_list, frozen_action=2)
    
if __name__ == "__main__":
    asyncio.run(main())