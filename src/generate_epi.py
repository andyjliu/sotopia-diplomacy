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
from profile_utils import adjacency_selection, get_env_pks
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

def setup_logging(log_file='generation.log'):
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)
    
    file_handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5)
    
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)




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

async def episode_generation(env_agent_combo_list: List[EnvAgentCombo[Observation, AgentAction]]):
    for env_agent_combo in tqdm(env_agent_combo_list, desc = "Processing Eposide Generation: "):
        try:
            logging.info("Starting episode generation")
            messages = await run_async_server(
                env_agent_combo_list=[env_agent_combo],
                omniscient=False,
                script_like=False,
                json_in_script=False,
                tag="simple",
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

    # setup_logging()
    
    logging.info("Starting the program")

    parser = argparse.ArgumentParser()
    parser.add_argument("--games_dir", default="/data/user_data/wenkail/sotopia_diplomacy/filter_games", type=str, required=False, help="Choose the evaluate model, the name can be seen in config")
    parser.add_argument("--env_model", default="together_ai/meta-llama/Llama-3-8b-chat-hf", type=str, required=False, help="Choose the env model")
    parser.add_argument("--agent_model", default="together_ai/meta-llama/Llama-3-8b-chat-hf", type=str, required=False, help="Choose the agent model")
    args = parser.parse_args()

    # TODO: After testing, remove the model here
    model = "llama3_8b"

    countries = ["England", "Germany"]
    agents_list = []
    all_character_pks = list(AgentProfile.all_pks())
    for pk in all_character_pks:
        profile = AgentProfile.get(pk)
        if profile.country in countries:
            agent = LLMAgent(
                agent_name=profile.first_name,
                agent_profile=profile,  # Pass the AgentProfile here
                # TODO: After testing, change to args.agent_model
                model_name=model
            )
            agents_list.append(agent)
            if len(agents_list) == 2:  # Ensure we only get two agents
                break

    if len(agents_list) != 2:
        raise ValueError("Two agents are required.")

    game_phases = adjacency_selection(args.games_dir, countries)
    
    # TODO: After test, add games_num argument, this should be: uuid_list = get_env_pks(game_phases, games_num = None)
    uuid_list = get_env_pks(game_phases)

    env_agent_combo_list = [
        # create_env_agent_combo(args.env_model, args.agent_model, uuid, agents_list)
        create_env_agent_combo(model, model, uuid, agents_list)
        for uuid in uuid_list
    ]

    await episode_generation(env_agent_combo_list)

if __name__ == "__main__":
    asyncio.run(main())
    # main()