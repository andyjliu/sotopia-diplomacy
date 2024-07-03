import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import cast, Literal

from sotopia.agents import Agents, LLMAgent, HumanAgent, RedisAgent, ScriptWritingAgent, BaseAgent
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
from sotopia.messages import AgentAction, Observation
from sotopia.messages.message_classes import ScriptBackground
from sotopia.generation_utils.generate import LLM_Name, agenerate_action, agenerate_goal, agenerate_script
from profile_utils import adjacency_selection, get_env_pks
from tqdm import tqdm
import argparse

async def episode_generation(env_model, agent_model, agents_dict, env_uuid, action_order: Literal["simutaneous", "round-robin", "random"] = "round-robin"):
    model_dict = {
        "env": env_model,
        "agent1": agent_model,
        "agent2": agent_model,
    }
    env = ParallelSotopiaEnv(
        model_name=model_dict["env"],
        action_order=action_order,
        evaluators=[
            RuleBasedTerminatedEvaluator(),
        ],
        uuid_str=env_uuid,
    )

    # Reset the environment
    environment_messages = env.reset(agents=agents_dict, omniscient=False)
    agents_dict.reset()
    try:
        message = await run_async_server(
            # model_dict = model_dict,
            env_agent_combo_list,
            omniscient=False,
            script_like=False,
            json_in_script=False,
            tag="simple",
            push_to_db=True
        )
        print(message)
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games_dir", default="/data/user_data/wenkail/sotopia_diplomacy/filter_games", type=str, required=False, help="Choose the evaluate model, the name can be seen in config")
    parser.add_argument("--env_model", default="together_ai/meta-llama/Llama-3-8b-chat-hf", type=str, required=False, help="Choose the env model")
    parser.add_argument("--agent_model", default="together_ai/meta-llama/Llama-3-8b-chat-hf", type=str, required=False, help="Choose the agent model")
    args = parser.parse_args()

    agents_list = []

    countries = ["England", "Germany"]
    all_character_pks = list(AgentProfile.all_pks())
    for pk in all_character_pks:
        profile = AgentProfile.get(pk)
        if profile.country in countries:
            agents_list.append(LLMAgent(agent_name=profile.first_name, model_name=args.agent_model))
            if len(agents_list) == 2:  # Ensure we only get two agents
                break

    if len(agents_list) != 2:
        raise ValueError("Two agents are required.")

    agents_dict = Agents({agent.agent_name: agent for agent in agents_list})  # Convert list to Agents instance

    game_phases = adjacency_selection(args.games_dir, countries)
    uuid_list = get_env_pks(game_phases)
    for uuid in tqdm(uuid_list, desc="Generating Episode: "):
        await episode_generation(args.env_model, args.agent_model, agents_dict, uuid)

if __name__ == "__main__":
    asyncio.run(main())
