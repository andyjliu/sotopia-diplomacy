import asyncio
import sys
sys.path.append("../")
from concurrent.futures import ThreadPoolExecutor
from typing import cast, Literal, List, Tuple
from sotopia.agents import Agents, LLMAgent, HumanAgent, RedisAgent, ScriptWritingAgent, BaseAgent
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
from sotopia.envs import ParallelSotopiaEnv
from sotopia.messages import AgentAction, Observation
from sotopia.messages.message_classes import ScriptBackground
from sotopia.generation_utils.generate import LLM_Name, agenerate_action, agenerate_goal, agenerate_script
from sotopia.samplers import EnvAgentCombo
from profile_utils import adjacency_selection, get_env_pks, random_country_adjacency_selection, get_env_pk
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
import pandas as pd
import os
import pdb
import random

def get_country(c):
    country_maps = {'A': 'AUSTRIA', 'E': 'ENGLAND', 'F': 'FRANCE', 'G': 'GERMANY', 'I': 'ITALY', 'R': 'RUSSIA', 'T': 'TURKEY'}
    return country_maps[c]

def get_agents(args,countries):
    countries = [country.capitalize() for country in countries]
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
    # pdb.set_trace()
    if len(agents_list) != 2:
        raise ValueError("Two agents are required.")
    return agents_list

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

def get_spec_env_uuid(game_id, phase_name):
    all_task_pks = list(EnvironmentProfile.all_pks())
    for pk in all_task_pks:
        envp = EnvironmentProfile.get(pk)
        if envp.game_id == game_id and envp.phase_name == phase_name:
            return pk

async def main():

    
    logging.info("Starting the program")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--games_dir", default="/data/user_data/wenkail/sotopia_diplomacy/games", type=str, required=False, help="Choose the evaluate model, the name can be seen in config")
    parser.add_argument("--model", default="llama3_70b", type=str, required=False, help="Choose the env model")
    # parser.add_argument("--env_model", default="llama3_70b", type=str, required=False, help="Choose the env model")
    # parser.add_argument("--agent_model", default="llama3_70b", type=str, required=False, help="Choose the agent model")
    parser.add_argument("--tag", type=str, required=True, help = "Choose a tag for access the database")
    parser.add_argument("--human_annotation", default="/home/wenkail/diplomacy/sotopia-diplomacy/src/evaluate_intent/manual_annotations.csv", type=str, required=False, help="Choose the human annotation file")
    # parser.add_argument("--sub_sample", action='store_true', help="Whether Choose to Sub Sample")
    # parser.add_argument("--sample_size", type=int, default=None, required=False, help = "Choose sample_sizes")
    # parser.add_argument("--c1", type = str, choices = valid_countries, required=True, help="Choose the first country")
    # parser.add_argument("--c2", type = str, choices = valid_countries, required=True, help="Choose the second country")
    args = parser.parse_args()

    # "llama3_70b"
    model = args.model
    # adjacency = {
    #     'Austria': ['Italy', 'Germany', 'Turkey', 'Russia'],
    #     'England': ['France', 'Germany'],
    #     'France': ['England', 'Germany', 'Italy'],
    #     'Germany': ['France', 'England', 'Russia'],
    #     'Italy': ['Austria', 'France'],
    #     'Russia': ['Austria', 'Turkey', 'Germany'],
    #     'Turkey': ['Austria', 'Russia']
    # }

    # countries = [c1, c2] 
    phases = []
    ha = pd.read_csv(args.human_annotation)
    for row in tqdm(ha.itertuples()):
        phase = {}
        phase['game'] = str(row[1])
        phase['phase'] = row[2]
        phase['power1'] = row[3]
        phase['power2'] = row[4]
        phase['actual_order'] = row[5]
        phases.append(phase)

    # pdb.set_trace()
     
    agents_list = [get_agents(args, [get_country(phase["power1"]), get_country(phase["power2"])]) for phase in phases]
    uuid_list = [get_spec_env_uuid(phase['game'], phase['phase']) for phase in phases]

    # if len(uuid_list) > args.sample_size:
    #     indices = list(range(len(uuid_list)))
    #     sampled_indices = random.sample(indices, args.sample_size)
    #     sample_uuid = [uuid_list[i] for i in sampled_indices]
    #     sample_agents_list = [agents_list[i] for i in sampled_indices]
    # else:
    #     sample_uuid = uuid_list
    
    env_agent_combo_list = []

    for i in range(len(uuid_list)):
        env_agent_combo_list.append(create_env_agent_combo(model, model, uuid_list[i], agents_list[i]))
    # pdb.set_trace()
    await episode_generation(args.tag, env_agent_combo_list)

if __name__ == "__main__":
    asyncio.run(main())
    # main()