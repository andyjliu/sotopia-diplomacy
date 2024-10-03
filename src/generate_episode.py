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
    uuid_list = ['01J92MBNG61S5TN7559PFBASFS',
'01J92MBSBKQ5GVPJQMJ4YV5NNF',
'01J92MBTR1AADEQ5SHFR1YXDFG',
'01J92MBS3R38CRNTHYEZDQXC7X',
'01J92MBNJR4EYW8QN2NGCDK3NE',
'01J92MBP9NMC0MW5XB77N431PB',
'01J92MBT7F5TGDM0E9BA3ABPN2',
'01J92MBNYFP1ANG3ZWQS2X0PNA',
'01J92MBV2MG40R8C53CK60YE1E',
'01J92MBSN6P9XTMGRYVVM7QW45',
'01J92MBSAJP1GQ18953SGMB88Y',
'01J92MBVD273R87ZX98RCM4CJ3',
'01J92MBN1F4YFVGSQ6TN9NVF03',
'01J92MBMWHS0XY95WXV972DDAG',
'01J92MBN4RR9A1W61G2KSEXG8P',
'01J92MBP1MC7T91HTKE0FF2KM9',
'01J92MBR63P8XC998RQD39Z0E2',
'01J92MBSXYTDFY6VVRQYSR3N5W',
'01J92MBS9ZPAHFXM7Q4STBZAN5',
'01J92MBTFCX1NTVMZ0Q0FE3CEV',
'01J92MBRNYH9Z3WM91WVKB83C1',
'01J92MBSBTB83ZXB7348TW93GS',
'01J92MBPKQSFY6RGRQY78Q19MF',
'01J92MBSPH506JWNJA68NXDFAE',
'01J92MBQHFFJP2PS8HE1R2JRJ5',
'01J92MBVEHP4M3H7JG96CFBYD2',
'01J92MBSBPF7C6W0WCC0BBKVVA',
'01J92MBSBF64MQCC1HT8QRR6CT',
'01J92MBTHRRPDRSA4XS20NJG19',
'01J92MBMVWXBJM0HKFZB4Q32YV',
'01J92MBTCGAMCHFG55G9ACVFW8',
'01J92MBSB5JZ1HCQGD95KVEPQS',
'01J92MBP3HMENP40KCA6RSQ9D0',
'01J92MBP3BFKHFZD763MCK9PKD',
'01J92MBSZSQRHNDVFFY42M554Q',
'01J92MBRBX72VJCKXYB01FR994',
'01J92MBSV4AVKDP9VXNZ2AEYEC',
'01J92MBPMKM3N00PZQS8HNJMY9',
'01J92MBRHEMXM3EPEZH3KVQTTK',
'01J92MBMW776R96TX8RT2WGZWG',
'01J92MBS6MY3EY73FG0M0J602Q',
'01J92MBSAF3682RQK5V515DD5X',
'01J92MBTN7CS8V8Y3V0W1W1P83',
'01J92MBN7PRJVCDNQBXC3HX4D7',
'01J92MBNDMKGS4XTCY1CXMW9RX',
'01J92MBQ74WW1XGJKWA65CM1BD',
'01J92MBP47A0AJC8NAGAY0C39T',
'01J92MBT66FWE5RHVYXCX5Q37T',
'01J92MBQ76VJYJJK6N7ZAED9ET',
'01J92MBT2CQSKPXG3J7DW48Z3J',
'01J92MBS8HJPQ6KMTAJ4DGMN89',
'01J92MBPH809255CB9F357N8Z2',
'01J92MBVC5FEC81GCJGY7FJ1RE',
'01J92MBSPSXXTZ8PDW8V6C8YBR',
'01J92MBSCQX0GGME93ZYJ32R8M',
'01J92MBNAGNTFMYFG0DGQHH47B',
'01J92MBRMBSDAYJHFGQDVNMA7N',
'01J92MBNTADYWJR018H48E128Q',
'01J92MBTF981WFXQCM3YR47JNE',
'01J92MBSBRTVWR7YGSX8A5V480',
'01J92MBVDPNJGF2P4Z1MWXQ9XK',
'01J92MBRQ562F09HH8R7K88XN5',
'01J92MBRJBY8D87M769R4K9ENF',
'01J92MBP20ZEKAEN9C0AN7TRZC',
'01J92MBPD5094087YJQDTD6WBE',
'01J92MBSQ6KWHNG52YSW6XT8YB',
'01J92MBRJDJSS53HVH2YTAG1T0',
'01J92MBQ6EMK0R6DKV0DM785MC',
'01J92MBPH56S83N9KD4G8GN8FT',
'01J92MBRVSJX55G4KQSMSQ1RN8',
'01J92MBQX8E9GV40ZB7HGCCTP1',
'01J92MBRZMCS6FV69WKPR6VV2T',
'01J92MBSM76RYZ4VBFQYSDYQQK',
'01J92MBQM6V3Q4M2JH03BRDD38',
'01J92MBVDJART7RQAQ74BNQ0BF',
'01J92MBQGH2KHH5ZE647VGHJK0',
'01J92MBT0QZD1ZA2B6MM9XHZXA',
'01J92MBV2E3W1SVEM530XPWW1H',
'01J92MBRHCABJKXFBJ0G2MDZ03',
'01J92MBND515BD9GBHY9Q60427',
'01J92MBP37BZE8VDXZR52TCTS0',
'01J92MBRVBA51WKTJT3N7A7EFS',
'01J92MBS6CFQJMPAVEMT7FG3DM',
'01J92MBSCEQ9XPHJP17NXEM0YX',
'01J92MBPJMY6RJ775BPN2PZJ9B',
'01J92MBS08AT32F5JZ6T91WRFJ',
'01J92MBPRG7GTQSYZV07726R9G',
'01J92MBPQ6HXZFR3R4S33MC3FC',
'01J92MBTVSMG3HHZ0DJM3XC68W',
'01J92MBNASA1HGJ25SGJ968216',
'01J92MBVK6GRZC2PY2YKWR75D1',
'01J92MBQTR2Y66SNKTBP752P6Y',
'01J92MBTVWZF7YE45JHEXZBAQD',
'01J92MBPK1M0HJ0V1332EVVTR7',
'01J92MBPQ8JHYBG5CRDVXXRXQ4',
'01J92MBQN8WEKMC13ST0RFC2SK',
'01J92MBQ1ENV9SCZV4WC1NCNZY',
'01J92MBSPKXBTA1W2PP2HKY4PH',
'01J92MBTG04PVK180SXXWAE777',
'01J92MBSR3F2YNXRXSX7XWH9GF',
'01J92MBRPYXT6NAQ9S9V7QG7CS',
'01J92MBTKCKSWK9NV320B4WEB4',
'01J92MBTQD916EAF9KD2H5R32M',
'01J92MBN1NZ07G3GCW8DHEK493',
'01J92MBSCTXCTS1CKQZ6MNNA3W',
'01J92MBT2Q3Y73B12WG4MNJENX',
'01J92MBV2CFAPBTZJG68BWHZ0J',
'01J92MBSZ2MV1H8RBQS7ZPXDVR',
'01J92MBMZGSFRTGWK14PR3D71P',
'01J92MBPB9YSW8N5CB77PFTTSZ',
'01J92MBREYDKPNWYM5HQXD25YC',
'01J92MBVJ5PJV5EXSM46Y4Z6Z8',
'01J92MBRC161F0B5P3R0QVRYPC',
'01J92MBTJBN5XS6PQBSEZ2TB8Z',
'01J92MBNG9BFMWEKABPDXQ5QA4',
'01J92MBTFV7ADQ8AETK6BKDYNM',
'01J92MBSZ8KM5KRV4PSTP2E8Y6',
'01J92MBR40D4TMJX8DBR6F4D1M',
'01J92MBP5GPGEE347CD1XPAN9J',
'01J92MBVJYBC6GMPWWGJR9TJAF',
'01J92MBQP53SAQ4YHQC6XZX5ZN',
'01J92MBQF44K0E8EA31C9YR206',
'01J92MBQEHTVCS7RFXBXCZ8S81',
'01J92MBSB1186RT04PPZB3WWQ8',
'01J92MBN5GC8VF639H8DK41FCG',
'01J92MBQ5KKC6NXN81M1NEP4SB',
'01J92MBQPE1F52H2DWR1VX0QMB',
'01J92MBQ98DED9ESAHT9FT8S92',
'01J92MBTHVVQ4CW2BTNWGDEA2Z',
'01J92MBP0BAV3J2F93TN43SEZP',
'01J92MBRKY7Q4YZM78JGJZM618',
'01J92MBVK02ZSX411XHH689WB9',
'01J92MBSNPSGK1874ZPPYK4NCE',
'01J92MBPMAWSMZMNHCAYEQWZYX',
'01J92MBV27YW9ABTR80BV6X9Z6',
'01J92MBSPQVZ4RNYSTNBXX0EPF',
'01J92MBQGZCJ8T8EX029R41FRS',
'01J92MBSVNRV16SSSHHHF60NTD',
'01J92MBQGWNQK6JTZ4V7JZZXG1',
'01J92MBR99DXZBX1CQWZ450NM7',
'01J92MBP42KX4MZZNHTR3534EV',
'01J92MBQX4J204KTZ59H0SW601',
'01J92MBQH965094C5BP1TMCCKA',
'01J92MBPRNQRWRQ725205RA291',
'01J92MBSF85CYV5Z33KW9VSQFX',
'01J92MBNDREGPTV06QAWJC80XF',
'01J92MBT5GDC09JK4KHK0H91DH',
'01J92MBS63HVE32J2VNQ8109AC',
'01J92MBS7FNJ3N27MNDABSS22V',
'01J92MBQHSETHFMREBBZXH1VH6',
'01J92MBSVCWBBVGVADNSJWE9X4',
'01J92MBR8AYVEEXN0R2ENG7EA0',
'01J92MBQH17PBWZV207RPERN31',
'01J92MBTFF8F90MH2RX9B01Y9B',
'01J92MBT2VGG4F3WRCYJ3S46B1',
'01J92MBT2Z5R82BTD4Q07HZZT5',
'01J92MBVDFQ6AHXFQ42GJ146WN',
'01J92MBN527T441P8YDMXAQ0XR',
'01J92MBSQ8338TGAA9SND18389',
'01J92MBQANTK0W5P2J6YQS170V',
'01J92MBV48R0Q70H01J654RWAP',
'01J92MBV3WVYNC0XX86V77KRQ9',
'01J92MBSEZG5638T25C8PPRQZ6',
'01J92MBSBWMEPP85TRHA4TJ6TC',]
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