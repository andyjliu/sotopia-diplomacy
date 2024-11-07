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
    # uuid_dict_list = get_env_pks_by_tag(args.env_tag)
    # uuid_list = [uuid_dict['uuid'] for uuid_dict in uuid_dict_list][args.split_begin: args.split_end]

    # The uuid here is with plausible
    uuid_list = ['01JBDMK14PC8R7S2R25NDS9AXA',
 '01JBDMK17SZQQ3FG5CACVDSYE5',
 '01JBDMK1MD46XHWVZY3EBM5HHD',
 '01JBDMK1W9J9ZSAYRWBRFGXPTX',
 '01JBDMK1HZXNMAQ85KVJ07T3BR',
 '01JBDMK0XE8R0QFNF4MS42SVMP',
 '01JBDMK1NY48W962CB1HKH8W1H',
 '01JBDMK27T1Y9303XVZDB27R4W',
 '01JBDMK1WS6GMGX5P6VXCP8VKZ',
 '01JBDMK12Z1TKH82RRDP4143WM',
 '01JBDMK138SNT7S3AK9C0B4WF6',
 '01JBDMK1RZDRJY15W2EK2NEV1K',
 '01JBDMK20QKSS90M7352BFS3D6',
 '01JBDMK241GJB232J9HZC7R8DH',
 '01JBDMK1XSF38TSJ813R182E1S',
 '01JBDMK1CBHXYFSSE094HQNGD0',
 '01JBDMK1R1YQGHXHWX1F1HQW4F',
 '01JBDMK2230JKQMW0GAXJADGG0',
 '01JBDMK1JQD1F0TD01XFG4J4B2',
 '01JBDMK1W3J2BNCZXFMH37NWNJ',
 '01JBDMK1XWSQ9N0YM181SWYF77',
 '01JBDMK1S7A7Q0R9TH7945TYR8',
 '01JBDMK1JN2PARJKMQD8Y6RQW5',
 '01JBDMK1ZVFDTAWWR4GBJ7ZYDV',
 '01JBDMK23A7FVJQ5J65QRF81DM',
 '01JBDMK2C9K275FEMPVBWWRDV7',
 '01JBDMK1CFSFC47GD2WXKH6VX5',
 '01JBDMK296QF4VPEFQH4PNG3PD',
 '01JBDMK23JK78M81FJ6FCJ39FE',
 '01JBDMK1WHCSNFV2QE272WASZR',
 '01JBDMK1TBHKS824WG61M27YQJ',
 '01JBDMK1V9KWN86JH1KZV2SFPA',
 '01JBDMK1WNHFN46ACWJBXNDPPR',
 '01JBDMK1SG8746GNMKW7BD2NGP',
 '01JBDMK1YZ3E9TM51NR7RTABCX',
 '01JBDMK1CRPQF0FCB98FCFJD3N',
 '01JBDMK22RZH4TSEBRDTFWP5QM',
 '01JBDMK27WJVFGK8SSWGMG93CM',
 '01JBDMK2C8VP4KN5RYK4N82V4N',
 '01JBDMK22JKMENZ69E8X98C6WH',
 '01JBDMK23YX0P4D2T436DPJPK9',
 '01JBDMK1VSS4J08C7N4QTQCDRG',
 '01JBDMK0SC4JJHEANZH93V6C8D',
 '01JBDMK0ZBAPQQ4JW53XS82TSV',
 '01JBDMK20Z3R6CFWZJSKR5TQXS',
 '01JBDMK1DEYTG4A5DV3A9NAEZH',
 '01JBDMK293Y74RD2FMR0JJ7PP1',
 '01JBDMK1YE0G0KDK04X2ZWRYEJ',
 '01JBDMK1KFJMZNHHZD37RJEEQQ',
 '01JBDMK1T9M4HBA2XRAR4PVK5T',
 '01JBDMK1T6Q57R9PSYP837X09G',
 '01JBDMK291RBN96Q3DZYDAX8G7',
 '01JBDMK28CD2NNXYNJ9QPW2JVE',
 '01JBDMK0X5A3QE57DWGM7HNMPH',
 '01JBDMK29ERB2QJ9XJMZY0AWMJ',
 '01JBDMK2C2TDXS91MQ5Y13QMHV',
 '01JBDMK17CX6YR9QKRCQ3XQ4ZW',
 '01JBDMK28EYCYH3A0XAHAM367Z',
 '01JBDMK2252B5V97GMWPT0TNWS',
 '01JBDMK0RW48437F46QCY90E3C',
 '01JBDMK2A6ATM7FB1BT8AFT39T',
 '01JBDMK2CMQ5NNDY6D01K1BQ1G',
 '01JBDMK21FWVXCD9C8Q1CWEV4A',
 '01JBDMK1X5BFQX1SFK79NFT1BY',
 '01JBDMK1E1VEFAKJFXHADZMK7B',
 '01JBDMK1NK1E9NPRNJEX7N3KMB',
 '01JBDMK1JJCYPTDNGTJZKWDTBS',
 '01JBDMK2ANXQTADGR0MZXPQMC2',
 '01JBDMK0ZDEC7P03HD7YCNKWH9',
 '01JBDMK1S96HFMH6RECQ93ZBBQ',
 '01JBDMK1ERRNNRMFCDWK77GFAS',
 '01JBDMK1M4XVZ73WX9SH1H5N5V',
 '01JBDMK22GBGYHCGV2QAGKH9BM',
 '01JBDMK2BNF2BYTMGSEZAMX70H',
 '01JBDMK236BNMMYKJW3JPVY7TE',
 '01JBDMK2951TJY9XC3KBF09C34',
 '01JBDMK231BJ7WA1JMGWGZGPSF',
 '01JBDMK1TVTZ3TP9HJRC0QBH35',
 '01JBDMK1C7CRY0CJC468VM3358',
 '01JBDMK1Q0JWKZM1EF16JY4P1D',
 '01JBDMK11VYAT1W364TMFVVBWN',
 '01JBDMK23SRF0E83ZMBZTN5NEP',
 '01JBDMK1WB4S7T6TA84XHHQWPE',
 '01JBDMK0SS526CVGNS8DTXD6K6',
 '01JBDMK2ASM14TW0CH75N0SJHY',
 '01JBDMK1T7RN8C5198746ZG5YA',
 '01JBDMK17YB39HV72VRDCFNM7X',
 '01JBDMK22VCBXK9F958F8T5GRE',
 '01JBDMK0R6XRHNPBGSCAWHW04C',
 '01JBDMK18PVSVQWJ6BGPANTNJE',
 '01JBDMK1XAKP98KH8PD2HNFSZA',
 '01JBDMK1AVCHC7GPGHVY54CWPY',
 '01JBDMK2DHZ67BV0EVJGK13E8Z',
 '01JBDMK1ZWHM1C68FJYSYYFQBQ',
 '01JBDMK1A1TZRY97FP9TZ5YY8Q',
 '01JBDMK0Z8TQB0FCRVBQMVXQXB',
 '01JBDMK2AD0H9G6WFNSH5EPQ4W',
 '01JBDMK0NDYH8XYW9HDRSRMNB2',
 '01JBDMK1EMT8DG6VRZAY5TEPXK',
 '01JBDMK1JKY78Z5DNWCQS8EGEP',
 '01JBDMK1D05AT3T6J7ZM4GGG02',
 '01JBDMK1XPSB0X1VZ9ZMGVTBC7',
 '01JBDMK1E6D6JFDAX85KGPPCMW',
 '01JBDMK28S1Y7KK90NSPJXR5KE',
 '01JBDMK1E4XYWRYZ5Y3E574DSC',
 '01JBDMK2004JEPZDJC9DXC2R0G',
 '01JBDMK1508G5S6JST0660PAXZ',
 '01JBDMK1CWSF812F9T0WSXC04W',
 '01JBDMK0QSFXSTCHT4BQHZ56C7',
 '01JBDMK1ECXNA87XZXF3VA55K5',
 '01JBDMK1Q5NY0E1YPGPF5G0MF4',
 '01JBDMK0X04JVXEAS747B4AQHM',
 '01JBDMK2336X5YF6R1VAEP64MC',
 '01JBDMK1CXZFQ7HE67FNWHFPHT',
 '01JBDMK1TJMW89Z9CM8AP8N3D8',
 '01JBDMK0PACWCS0V4Q9737TXGC',
 '01JBDMK1JD9AM894ZAKEA5EE2V',
 '01JBDMK1PKW54JXEX28MC25NMM',
 '01JBDMK24BE5DWQKFNTS8Q02XT',
 '01JBDMK1VXH11P04TM8V4ZY54Z',
 '01JBDMK0PBKTZ5JA14QBGWENN2',
 '01JBDMK2CBH0GJ3K6YX0D847FN',
 '01JBDMK1E8JSQDKXRR5JK06ADS',
 '01JBDMK1Z9E3R886YFG685B5YA',
 '01JBDMK1BAS84ZMMTETG57RHD2',
 '01JBDMK2B4FCD2DXR7SBGFWTSN',
 '01JBDMK1MA449RMVDQRTS0CARP',
 '01JBDMK2BFBJYVBA152V2KGM7B',
 '01JBDMK1W68E5XBGVNZ2QBGV6X',
 '01JBDMK1SH3X200292EQJ56873',
 '01JBDMK1Q3FGY7RYQZZSXARF24',
 '01JBDMK12CAB7SVDTEB55MAR4V',
 '01JBDMK20W41E7F6P0Y303GNZS',
 '01JBDMK2CVW4X93D44F573G83X',
 '01JBDMK28AYZZ049WEYT1HZH1D',
 '01JBDMK1CD3ZAV4SQV5N2NXX8F',
 '01JBDMK1C9Z657V5RB5HFHET5H',
 '01JBDMK2BZGZSRGH71ZTKP290Q',
 '01JBDMK1EPSMPGKSABFD7WVK2S']
    
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