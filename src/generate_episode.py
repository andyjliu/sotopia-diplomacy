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

# def pick_two_adjacent_countries(adjacency):
#     country = random.choice(list(adjacency.keys()))
#     adjacent_country = random.choice(adjacency[country])
#     return country, adjacent_country
def int_or_none(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")

def get_agents(args,countries):
    agents_list = []
    all_character_pks = list(AgentProfile.all_pks())
    for pk in all_character_pks:
        profile = AgentProfile.get(pk)
        if profile.country in countries:
            agent = LLMAgent(
                agent_name=profile.first_name,
                agent_profile=profile,
                # TODO: After testing, change to args.agent_model
                model_name=args.model
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

def create_env_agent_combo(env_model: str, agent_model: str, env_uuid: str, agents: List[BaseAgent[Observation, AgentAction]],
                           action_order: Literal["simutaneous", "round-robin", "random"] = "round-robin") -> EnvAgentCombo[Observation, AgentAction]:
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

async def episode_generation(tag, env_agent_combo_list: List[EnvAgentCombo[Observation, AgentAction]]):
    for env_agent_combo in tqdm(env_agent_combo_list, desc = "Processing Eposide Generation: "):
        try:
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
            # for episode_messages in messages:
            #     for message in episode_messages:
            #         logging.debug(f"Message: {message}")
        except Exception as e:
            logging.exception(f"An error occurred: {e}")

async def main():

    
    logging.info("Starting the program")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--games_dir", default="/data/user_data/wenkail/sotopia_diplomacy/whole_filter_games_100", type=str, required=False, help="Choose the evaluate model, the name can be seen in config")
    parser.add_argument("--model", default="llama3_70b", type=str, required=False, help="Choose the env model")
    parser.add_argument("--env_model", default="llama3_70b", type=str, required=False, help="Choose the env model")
    parser.add_argument("--agent_model", default="llama3_70b", type=str, required=False, help="Choose the agent model")
    parser.add_argument("--env_tag", type=str, required=False, help="Choose a environment tag for getting the choosen environment")
    parser.add_argument("--epi_tag", type=str, required=True, help = "Choose a tag for access the database")
    parser.add_argument("--split_begin", type=int, default=0, required=False, help="The begin index of the sub samples")
    parser.add_argument("--split_end", type=int_or_none, default=None, required=False, help="The end index of the sub samples")
    args = parser.parse_args()

    valid_countries = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
 
    # TODO: Change to loading from environment profile database with tags
    uuid_dict_list = get_env_pks_by_tag(args.env_tag)
    uuid_list = [uuid_dict['uuid'] for uuid_dict in uuid_dict_list][args.split_begin: args.split_end]

    # The uuid here is with plausible
#    uuid_list = []
    
    print(f"Length of uuid: {len(uuid_list)}")
    # agents_list = [get_agents(args,uuid_dict['countries']) for uuid_dict in uuid_dict_list][args.split_begin: args.split_end]
    uuid_list = uuid_list[args.split_begin: args.split_end]
    agents_list = [get_agents(args,get_env_countries(uuid)) for uuid in uuid_list]

    # uuid_list = [get_env_pk(i) for i in game_phases]

    # uuid_list = [pk for pk in get_env_pks(game_phases)]
    # if len(uuid_list) > args.sample_size:
    #     indices = list(range(len(uuid_list)))
    #     sampled_indices = random.sample(indices, args.sample_size)
    #     sample_uuid = [uuid_list[i] for i in sampled_indices]
    #     sample_agents_list = [agents_list[i] for i in sampled_indices]
    # else:
    #     sample_uuid = uuid_list
    
    env_agent_combo_list = []

    # "llama3_70b"
    model = args.model    
    for i in range(len(uuid_list)):
        # TODO: Later can change to different models to see the comparision between different models
        env_agent_combo_list.append(create_env_agent_combo(model, model, uuid_list[i], agents_list[i]))

    await episode_generation(args.epi_tag, env_agent_combo_list)

if __name__ == "__main__":
    asyncio.run(main())
    # main()