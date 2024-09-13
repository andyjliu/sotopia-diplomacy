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

    if EnvironmentProfile.get(env) is not None:
        return EnvironmentProfile.get(env).agent_powers
    else:
        print(f"Didn't find env uuid: {env}")

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
    # random.seed(42)
    # game_phases = None
    # while game_phases is None:
    #     sampled_countries = random.sample(valid_countries, 2)
    #     # Try to select game phases based on the sampled countries
    #     # game_phases = adjacency_selection(args.games_dir, sampled_countries)
    #     game_phases = random_country_adjacency_selection(args.games_dir, valid_countries)
    
    # TODO: Change to loading from environment profile database with tags
    # uuid_dict_list = get_env_pks_by_tag(args.env_tag)
    # uuid_list = [uuid_dict['uuid'] for uuid_dict in uuid_dict_list][args.split_begin: args.split_end]
    uuid_list = ['01J7FDPS2PYERZDER3ZF59PTJM',
 '01J7FDPPFFRKE7ZWPTA42TPV8M',
 '01J7FDPT014YS7712NZ70R145X',
 '01J7FDPS22ECDCZTRA24C9QXZ4',
 '01J7FDPQFWW2D5E3V4ZW8FC4D9',
 '01J7FDPS4JF65SVY1A0N1FEX8J',
 '01J7FDPQ55FZC5QNCVPPG6ADC6',
 '01J7FDPPRMK2N6JWH3HGQECGWE',
 '01J7FDPS1WF1YFYRTW6QR1C757',
 '01J7FDPSFKJG89HX4BYC5H33PR',
 '01J7FDPSYEF3BAVBH9Y6ZVAMHN',
 '01J7FDPRMM1JN91SNP053E3PAK',
 '01J7FDPRHNEVMRM4MAXYX6GTP0',
 '01J7FDPTPEY314883MP21HKZSQ',
 '01J7FDPPMPJJNS4RTRG15N03M2',
 '01J7FDPPSC4YHXGFHVFNG2V7EM',
 '01J7FDPS1QX1ZVP6CWAZG3FM04',
 '01J7FDPPEJ8BG86YTVKR7PK5W3',
 '01J7FDPPKMREG7JZBEYCXXW1A6',
 '01J7FDPPGGS2Q5JJ58BJXC7W5R',
 '01J7FDPTPA8EVRXKQHX5PHDHKS',
 '01J7FDPRXE0JB8H1YM30A4KG5T',
 '01J7FDPS124G61PK6QXD2EMSTN',
 '01J7FDPTF8PG4SYSNJYZV3DRA8',
 '01J7FDPSCY9C4NXMNZJ8A202PT',
 '01J7FDPQJHKXV1FZNPPAZCG26H',
 '01J7FDPQFPEDDWB747G6X0XHZP',
 '01J7FDPTP8AQBXA62N38G2V9ZE',
 '01J7FDPRZFDH5DBR6V36AZ6T06',
 '01J7FDPSNQJM4RBQNYAZC95JMB',
 '01J7FDPR6981GRF306PZJMG13T',
 '01J7FDPS107XHF7D8VNMG5Z8KX',
 '01J7FDPSC9J66HM6HPF4QSHNCG',
 '01J7FDPQV9T272SF1VKZX0MZC1',
 '01J7FDPRYMZD9GSMAB1R2SCKH1',
 '01J7FDPT3NZFM39X5QPYRQ0GSK',
 '01J7FDPQCHZ5B5ZT2CPYFA5983',
 '01J7FDPQSWW0H1M1S0PHF9A4Q4',
 '01J7FDPT6H9YKWFXCR7KJA57M4',
 '01J7FDPQ388S6CN8KCEDHBY115',
 '01J7FDPTN7J0R9KAA42C03BP98',
 '01J7FDPQ9ST9NZ9G7ZPJXM68WW',
 '01J7FDPPDPFN18RPJ19ZTD23T1',
 '01J7FDPPJEN090Z1PG4JC26YEF',
 '01J7FDPPGB6Z6Z1S3FFXNB7B27',
 '01J7FDPPKJY6V56DTRPSN764P8',
 '01J7FDPR7RGDEVREGA3Q0ZBV5B',
 '01J7FDPS2GJGXZ8VKYV5N3YTWY',
 '01J7FDPQ50B05KH5FEATWKJT18',
 '01J7FDPS0YPFX59E9C2N9TA9QW',
 '01J7FDPRVKQ48154RZERJG5T38',
 '01J7FDPPQV3MQVPYTBT31SR9KK',
 '01J7FDPPNBVF61890FRSY76ZVH',
 '01J7FDPT0A1AWQ8F6C0KPSHKQZ',
 '01J7FDPPJKN0THWKFHZXJT61JE',
 '01J7FDPQ1N8HKX4ZBDME506PT0',
 '01J7FDPSBXEZ3GXGK82DNAYP5S',
 '01J7FDPPT5GRYA76SEJNECFXY1',
 '01J7FDPRQ7TRRH74A62VCY370F',
 '01J7FDPS1HNBRQQYP2RAB6SZZ8',
 '01J7FDPPDZY89M383W8Q9XAZ10',
 '01J7FDPPT3P0AHYT61BYR3W0HB',
 '01J7FDPS1YKN4N93V1SWW1PPMA',
 '01J7FDPPDVYR4R0DS2PA757ETA',
 '01J7FDPPN966X0KZJQ4M7SBGZR',
 '01J7FDPQX4RP0Q4N370EENP4NW',
 '01J7FDPPJH1RKBQV6NHWD7CB0Z',
 '01J7FDPSNT0V421G31N1B2819D',
 '01J7FDPT6EVAQGAXAQNS6QMJZJ',
 '01J7FDPS2RD2WH19PX8PV6F69R',
 '01J7FDPS1DX65ZDSWAN695J3NC',
 '01J7FDPQEB5VM1EFYGZ27XA862',
 '01J7FDPQRHRFSF89BYR83CA5MX',
 '01J7FDPSBPGHMZYYKP42WZ2S6V',
 '01J7FDPRGVT4P8EASDJASEDDKQ',
 '01J7FDPSYK2DF205XW90JKH7PY',
 '01J7FDPT6B0HXX7VV8BJCMV2BJ',
 '01J7FDPRRHKS0DXSTQWMR5CDH2',
 '01J7FDPRRNDWG7GZ3D8JQXB2YX',
 '01J7FDPSNY3TWKSPX4BKVVXY7C',
 '01J7FDPQ0TRPF45GDQZ9RJV122',
 '01J7FDPSXVCDVYBT2V1FSRDNPE',
 '01J7FDPRXX9ANAJAK69XRBKBCR',
 '01J7FDPRXNHFJZANFR4M55G6N2',
 '01J7FDPS39JJ872FS7HBBYA8EA',
 '01J7FDPQ62FJAX8FZYDCWXEX31',
 '01J7FDPTSZSW3R8V611W1G3Y86',
 '01J7FDPSCPX4H6GEKBPBCKQKDX',
 '01J7FDPQX7C0ZAFW79Z5KG1WD6',
 '01J7FDPQ0WSD19TPA6SV6FVCE9',
 '01J7FDPQ4Y9CQAHSR4AT4BV5TA',
 '01J7FDPPWRQ7D9R7KWY9V12SM9',
 '01J7FDPQRXSJFQXAQ43PXJ700C',
 '01J7FDPR1N16BCWR56E1CHE7CB',
 '01J7FDPQPDP418FYPK3B5KQMNW',
 '01J7FDPSHPMY5W04RR2EKEBEHH',
 '01J7FDPR3FV2DJ5SP09XRCHD34',
 '01J7FDPSJV40APMWX4D00YBJ0J',
 '01J7FDPPGJXZQZ9002BWGKYGBC',
 '01J7FDPTPZ7Q6H2F9S7JKSFX75',
 '01J7FDPQGFYEN4N2EN7JYTZTKC',
 '01J7FDPTFFV3X3R20FVK8EEMR6',
 '01J7FDPSRFBW6MC05ST5SVD4BJ',
 '01J7FDPPSSB7Z00ZRSZ55G6FQC',
 '01J7FDPPKB7XACP2E946X7PTT5',
 '01J7FDPSB0X6MQAQWXQ0JJRFXT',
 '01J7FDPQPH5M8K39KB2K0QMN7J',
 '01J7FDPTTJX5ETNP1W4C4G5BM4',
 '01J7FDPR261QTBBVEHXJE9F705',
 '01J7FDPRQ01G2QFSTZEC0B71D1',
 '01J7FDPQN7V7N795YAFSD8K957',
 '01J7FDPQPFXK32HR16PXP7FKA5',
 '01J7FDPSR0YWVJNX60M6VVR7GT',
 '01J7FDPQPRGKZZDRN1FHH6DGJJ',
 '01J7FDPTTMYPBVTZA48CCM9CD9',
 '01J7FDPPMM2J07FB1EQ54NZV6X',
 '01J7FDPSBVQM1HGMXH3PATD2VE',
 '01J7FDPPS6V4JY8EFY8XSF6654',
 '01J7FDPR7NHRZ5RXP3FC3QXC1T',
 '01J7FDPTFCYNXT2R0FJEG1BSXY',
 '01J7FDPTNX08RQ5KRAPDK3HSD3',
 '01J7FDPSZHV5WXHFQSZAK1VPRR',
 '01J7FDPSY0MBX42N7BJ8RP0FC8',
 '01J7FDPS200Z984DQKMYF77GC7',
 '01J7FDPRRZ1ETGA7HJYR9NP1AD',
 '01J7FDPS1V0NNDZ2E484X4S4VX',
 '01J7FDPPSMTHBXJ4G25SWAAAEE',
 '01J7FDPT347R879EYW7C8FPN5R',
 '01J7FDPPDDHKWDC0EB53B4AQJG',
 '01J7FDPRGXDPTJ5YCPXCX7B60C',
 '01J7FDPS4SRFQN4NCMPTDBTPEV',
 '01J7FDPRK8E9BCZQ363TA95HWD',
 '01J7FDPS66AFZFGT71XGTSZKY1',
 '01J7FDPQ20KGT1HFNM2ZKQVPXP',
 '01J7FDPSZKFZZQHR3P3Z24F2SK',
 '01J7FDPQP3GZRNAVY0HE0CQEGQ',
 '01J7FDPSAJN6ESBHET1FGN8SM7',
 '01J7FDPRJZXP49BSQWX089EW6B',
 '01J7FDPSBQWKRAQXV381NZHHFH',
 '01J7FDPPSPVR6XGYYJZ84PS4R3',
 '01J7FDPSJPEJEQRYAAMP602B8J',
 '01J7FDPPFC7TEV38V5AN5DBK02',
 '01J7FDPPVKJT1NMCYSN9R46VZK',
 '01J7FDPQQ6RMDZMPZWGRH7W0JJ',
 '01J7FDPPTTXSWM7F7XMX77W4J3',
 '01J7FDPTGPDP2VHXW6YD1AKHJ7',
 '01J7FDPTTSXH2YQTH7VN3V2N59',
 '01J7FDPSFBSV888EXCVEM75XRV',
 '01J7FDPSS9CXNAH0Q1WABEAC4R',
 '01J7FDPSVGSGX06J9Z6QX15TWM',
 '01J7FDPQZZP6PGBQ17A65TA052',
 '01J7FDPPPRC0ZZBPC1CM6WGXNT',
 '01J7FDPSXXY1DEWAZSKSYVDQ1B']
    # agents_list = [get_agents(args,uuid_dict['countries']) for uuid_dict in uuid_dict_list][args.split_begin: args.split_end]
    # uuid_list = uuid_list[args.split_begin: args.split_end]
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