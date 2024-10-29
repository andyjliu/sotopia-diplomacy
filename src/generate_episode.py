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

    # The uuid here is with plausible
    uuid_list = ['01J9BTDT5GK912JBM5S0RV1ZNV',
'01J9BTDTGXBPGR57BWCFJ97S1B',
'01J9BTDTDV56MSD9MHFAR6Q73J',
'01J9BTDTT9PRXK5P42G503ECAS',
'01J9BTDT6V7NX7H6P46FZZEAC8',
'01J9BTDT7KA05TAFMH16D85XZF',
'01J9BTDSTRZVNBK4EZAT1SEMBH',
'01J9BTDTSTP9SEJ185RTTX40QZ',
'01J9BTDT7JCRHMF8AS9XK04SW4',
'01J9BTDTN901X2S3F0AD9AHY0G',
'01J9BTDSP36CHY32M2CZ5DRF5E',
'01J9BTDT4GDR5B6ENTFZNPWTWA',
'01J9BTDSN2NXABYEAXH1Z6AB85',
'01J9BTDSQDYF8EQXZQTB07SKA4',
'01J9BTDSRDYJ1BMSFEZ8M6DAW6',
'01J9BTDSMYJA83CMQ4B7ASB3S2',
'01J9BTDTRKJMJMSEQ89ANQQR68',
'01J9BTDTJ5A30HGZVJYFZ7MPJ3',
'01J9BTDTPXAB6DA8TT1145JDJV',
'01J9BTDT22680DTH0F1S250BY2',
'01J9BTDT8BQF6HCESCKBD17RT8',
'01J9BTDTA0WKY1TFVRY4H8NJMS',
'01J9BTDSXYDEYXEEKV4RGP6MQB',
'01J9BTDT5T8N2WWXD7RW2WZA1Y',
'01J9BTDSWA59Y0BMZKRB6D0F43',
'01J9BTDT3JA67DDBQS62J8WE8B',
'01J9BTDSNQWZSRNCSS48RM9DVJ',
'01J9BTDT8W6WQWXFR61PQCRTH8',
'01J9BTDSSQJS2JEDJNCFKGV25T',
'01J9BTDTB99XPMM071SXMSHC3X',
'01J9BTDTSNJQZ8WTS8K7C7CRVE',
'01J9BTDTPBDPEV1XZ3APKVTQM7',
'01J9BTDSRQ7M220VVC3VAMA3WG',
'01J9BTDTDWZ2C3XR86930DW0JQ',
'01J9BTDSN7NEQPR7NVF7N9E04Z',
'01J9BTDSXBH0EWFMJGZBZ3AVZR',
'01J9BTDTHSQR7R60DV6VRDAC1R',
'01J9BTDSPYGMZA6FSR98P32MAX',
'01J9BTDT6D8HY5G3M7F790KMHA',
'01J9BTDTD7HK86MBY0BRMJ210T',
'01J9BTDTH9S7VNPWPANM177WHZ',
'01J9BTDSWMTDMV22TB1FR8XG58',
'01J9BTDSM52WT1CQH0HM0XK57Q',
'01J9BTDT71ACCB91CB6N66A03C',
'01J9BTDTA22A2RRXPD98V65YP8',
'01J9BTDTD56YHYCD65R8PSA8QG',
'01J9BTDT96D9H2NPAWGQ6SFCYR',
'01J9BTDTAHV0S0H94ZQKVW5WPP',
'01J9BTDTB48HXXXNCVMWRAEKXV',
'01J9BTDSR2VH72M3SDZ1TVWW6T',
'01J9BTDTAWXG0ZAW4KAYRV7PAT',
'01J9BTDT53XNZMEG30HZWYJHVW',
'01J9BTDT59CGR4Y6FRCA4DJ6Z8',
'01J9BTDTH7NSHB807KP54D24FC',
'01J9BTDTATJMDQ01NW668495FB',
'01J9BTDTR8SZRP1X5KVZYB8S3C',
'01J9BTDTQS6H2AW8Y753QM8PJB',
'01J9BTDSNMWQJ7CJEMMA3SR219',
'01J9BTDSWNPGHSVA9BTJHAFFQ1',
'01J9BTDTD1SZ2WBY2PPHEBDM82',
'01J9BTDT72ZWD9V63FR76N2KN0',
'01J9BTDSSS3ESENE2610C1CDHX',
'01J9BTDTP4YQ37838CK245WYQN',
'01J9BTDTH19BMYYZDT72Z8FK3W',
'01J9BTDTC6RBF7GBFHPAAE9SW9',
'01J9BTDT5YY36QDH9QG61TFF5E',
'01J9BTDTTM64866S99X9GJ9HAA',
'01J9BTDSWEXWNEH4R0GWBR10K6',
'01J9BTDTPDZ8F3T32239B96ARW',
'01J9BTDT4TK7F2SJMF1Q9BD5VG',
'01J9BTDT6Q31FB4XTCW1R47Z1Q',
'01J9BTDTQHH6ZEZEG7HWR0C3FW',
'01J9BTDSMS71VQNN4PQAQP6XR3',
'01J9BTDTSBANN1KQQ23QDTEPHP',
'01J9BTDTJ19FEZVMTQRSVJ1NF1',
'01J9BTDT2VSCP571PH17YG3M2N',
'01J9BTDSMXH316DKA29DXR8BTC',
'01J9BTDTP74RKF8GRVY0KHQBKR',
'01J9BTDSQ0ATJVTFD7QB2HMWP8',
'01J9BTDSQT7X25671B8AW77APQ',
'01J9BTDT85MKS9GH1P34FCVNM3',
'01J9BTDSR4YWJ7QT1E3AG3H0EB',
'01J9BTDST911A6F81C8CJTNMA2',
'01J9BTDSR0MSJCJCKH741ZBS1V',
'01J9BTDT8EDCMY1QSR8DK99AMF',
'01J9BTDSQ6JZN6VQ9G2GX5JJ2R',
'01J9BTDSXE269Y69C8S27R0475',
'01J9BTDTPTM9EPMZ1H9T4PB5VW',
'01J9BTDTT03YT1RQ894WCA8H2S',
'01J9BTDT6698Q7A2TSXAJ8S3CY',
'01J9BTDTCE0KTFA96S4TXCA87N',
'01J9BTDT79AMAJTVRRQWQHBHB4',
'01J9BTDT9N6JJ9E8BPM3GEMWTR',
'01J9BTDSY4B6Y4RFC1K3M81C7V',
'01J9BTDTRRP22KXD2DZPKSMJA3',
'01J9BTDSQYQXR3BNR9TPC2Y88M',
'01J9BTDT7BPRS36SC0E4PJ4QST',
'01J9BTDT27D2GF9VP3QENKCNF1',
'01J9BTDTDRKWXCREK7W1QN01CY',
'01J9BTDSXGH47AF47HMJT4QKDM',
'01J9BTDT2RJ5ZV7R7B9KK426BJ',
'01J9BTDSWQYZAN8YECHKZNETBB',
'01J9BTDT8AXH28AVDKV4YT7MP8',
'01J9BTDT38Z0C4GJF472VH9SPB',
'01J9BTDSN02VFNXB3DX6G5PGNX',
'01J9BTDTFWZVDMAWQ0612T7AV6',
'01J9BTDTCKCGFFE70DYB9FEGXY',
'01J9BTDTS9HT9RDPTK5JGV23QK',
'01J9BTDTGT19V0CHK21GJZJ7MH',
'01J9BTDTTABYNXAA9FPJJFA2Q0',
'01J9BTDT88REHMPA1PG0N3J11K',
'01J9BTDSR59H3X9VYMEM0CK56W',
'01J9BTDSPARTEPS43HPYP8Z0FW',
'01J9BTDT7EHS7BV0XZJQ13W6FC',
'01J9BTDTABJD0667Y1JN0YTQAN',
'01J9BTDTNJHKJX55PNNMVZVYAQ',
'01J9BTDSXQGNPYRQBBP4H2PXRY',
'01J9BTDTEP5HY5KV2DPZ8DYT54',
'01J9BTDSP8DJGPR44YRRXVT8GH',
'01J9BTDTGA6F2TCGZFWZKXY244',
'01J9BTDTNTPS06KTEYE0KF9EQ7',
'01J9BTDSXTMQ1VYY084ZJRB4Q7',
'01J9BTDT6C9ZQ5THZ38D88FSJR',
'01J9BTDTEGA0EJTWHMP0XT5EKA',
'01J9BTDSWCYERQP48DFNEVBXCJ',
'01J9BTDTAK42N8Q7K2CJYKKAGW',
'01J9BTDTGRZF95KBAVKWZJR92D',
'01J9BTDSS84WQWGFYTQD9PVP8B',
'01J9BTDT9YQH8MW0B3FMG33HRM',
'01J9BTDT9DY7NE3Q9JYASVW0CQ',
'01J9BTDSQ2F2DZJH2VKMKD2SV7',
'01J9BTDTNM8C9JHYZSNMCJV5P6',
'01J9BTDTN3QSRECKZ8S66NXKK4',
'01J9BTDTCVM0KZ8ZYD2NSNZ8SQ',
'01J9BTDT87M24YG175G9ZCPEC8',
'01J9BTDTQ5PR8QFN5ZTJT42WMM',
'01J9BTDTE64H24YDCNYQS0DRG7',
'01J9BTDTFRY6KKFTG85R69765Q',
'01J9BTDTHQY9N1A8ZC98PEPSJ3',
'01J9BTDSNAKZ67A3RXZ2WQBNPJ',
'01J9BTDSPFM2R61H5G7C71BM7D',
'01J9BTDTGGDGZB6XPB8DCE27QM',
'01J9BTDTQX6WSH9HYW4NG24C66',
'01J9BTDSWSPAJE9RTER4B7FKV5',
'01J9BTDSKZZDH1K4PN0NG3KM2E',
'01J9BTDTERK8M362KR3N68PA0T',
'01J9BTDTBRSGVAE7FEYRD06WFT',
'01J9BTDT7GD0DXE90HNYMS87EY',
'01J9BTDSVJC3BK3BR5NW9VXFT7',
'01J9BTDTP9ASB8DGB2AC9XB681',
'01J9BTDSRCT8CZM4TSNYGDVFJV',
'01J9BTDSS0NQ97CP3TSGDFPX7A',
'01J9BTDTQZTV3MCC1P1CR1K3J5',
'01J9BTDSVF9R8E6570W7AC3B9Z',
'01J9BTDSW6Y21EP8E0T4X93S5M',
'01J9BTDSY00AR7K1ATPDJ5HAPD',
'01J9BTDTMVQNGP886PM6SAW2CH',
'01J9BTDT99D7SCHFXCXEXZDB5H',
'01J9BTDT2AQY04HSFMA85JK1EQ',
'01J9BTDSSGGM1H6CCBVG22TTM6',
'01J9BTDSW58B3TYF5C363XCPE6',
'01J9BTDSRYRG3S83SVGK4KAQ5M',
'01J9BTDSM4K0WAK3Y6HZ0FESZ4',
'01J9BTDSMHAQS4NXD6XCK9PV7P',
'01J9BTDTHFZQ7EEDJDDZWH4GNE',
'01J9BTDSQG2BSGSTRN9XM36HQC',
'01J9BTDTDYS7AWEK6QMGPVPM42',
'01J9BTDSPH2TSKJ8X10CZ63ZK5',
'01J9BTDTT5QXGS9G6CHPP3J786',
'01J9BTDT24C0PJ34V3BDQS2BYE',
'01J9BTDSSM0Y9H3A71T1J1W4R3',
'01J9BTDST4F02PR5V2NHPY16WV',
'01J9BTDT2TS8WQJ61HMHZG5VC2',
'01J9BTDTCAS0QM25R4E5SJRR20',
'01J9BTDTF6RZBMFE7VRMRTEQ1S',
'01J9BTDTBGC9T664C2Q4XWW8PM',
'01J9BTDSS5B0A4TPC4ARADSFAT',
'01J9BTDT9RW4FZ9XXJVKHV3JT5',
'01J9BTDSYCSVYB8WHRSY6CJQKJ',
'01J9BTDTF8K4CG8346EG71H0AJ',
'01J9BTDSMCTW0RBCD0APVQDYFE',
'01J9BTDTN0MVYVQB3N319XX4DK',
'01J9BTDT5SRY8MG1S9M09Q7Y8A',
'01J9BTDSW0Z30MGSM9PGSA9ETD',
'01J9BTDT5WSHJRAMKGKE28G4V1',
'01J9BTDTRXN9DS4J603W0JMSJF',
'01J9BTDTS1FVBZSQN7GQMD03JV',
'01J9BTDT505W0106BC653P0HC0',
'01J9BTDSTJCWDDG5JVZYV2KXH2',
'01J9BTDTPKBRKE5AWHN2V4SRJ7',
'01J9BTDSQX725RCZMQ6R71S0NT',
'01J9BTDTEB5FYWMA14TW5NFVBK',
'01J9BTDTGKW4899MB1FEVHNKT3',
'01J9BTDSRSDCEG0WANKBRNRQEK',
'01J9BTDT9MNS8TBN0DTSPQ5284',
'01J9BTDT8YVN4ZKV6FD7SH8QAX',
'01J9BTDT30YMEJ5Y6QQD4TKV3N',
'01J9BTDTG77X8Y0ZK8B3KMZ6SC',
'01J9BTDSP1XCGFBYFTPZZ0FQFT',
'01J9BTDT699VH76AJ1WHA4140T',
'01J9BTDTNZHQSC4NMZB8AHGA1P',
'01J9BTDSVVC692SEFFYX920VX9',
'01J9BTDSQ4GA7MR4P07VPYKE9B',
'01J9BTDTA7P7XS7CZWFNBC4R0K',
'01J9BTDT2GJT2CG0H21CSPF52W',
'01J9BTDSXN8MQ7EKGC10PGDS9A',
'01J9BTDTHJRBA0AT46K25WCJH9',
'01J9BTDSWYYBVVJV7PAD18S2Q7',
'01J9BTDSQS6KB7A4BGQ922J1XY',
'01J9BTDSY61WF0T5RG546S9HCF',
'01J9BTDTCC21S7AK6HYY0G4CDW',
'01J9BTDSYGK84YVPJR5PQ5MTA8',
'01J9BTDSNJJ86BRTHG8QRV0FR9',
'01J9BTDSRHJT2X2H54DQMPZDDC',
'01J9BTDSYEPVYRTGMAHAXDF6DK',
'01J9BTDSWVRRDPZ5BQ0P6AD5F7',
'01J9BTDT3QB1X03HR4DTW7HWGY',
'01J9BTDTBN4QCGF8HDPRCQGPW5',
'01J9BTDTTJ76DEFBAJN9VAFEDX',
'01J9BTDTFB86FA1EG76TT5QDVH',
'01J9BTDTSCMW1BMWMKZXMKCYYH',
'01J9BTDTFYH8WGYH3QERZZW4W7',
'01J9BTDTBVQ41ZJ7XKJXY6Z85M',
'01J9BTDTDPGWQEJ9SCATSATFP1',
'01J9BTDTRD4AR9W2FTY3FEFH35',
'01J9BTDTBXA39XQ4MCP1Q0T14H',
'01J9BTDSSJGWFJXKNYBE9PX886',
'01J9BTDSS318X6KYC54VW10658',
'01J9BTDT9FC6X1M1NB500NST63',
'01J9BTDTAN876GF7G21YD0F60Z',
'01J9BTDT9CCBG3PFJ4Z9PPYHJK',
'01J9BTDTA48Q89K8Q3GPY4EJHX',
'01J9BTDSVCG9X0X7W35HYK9J4D',
'01J9BTDT7PG1572QMPB8AQ65X1',
'01J9BTDTDJ3B685ZW1TKG5A0QA',
'01J9BTDT5PEDKHDH739GW5A7T4',
'01J9BTDT2D88ZT7RFS8G964D0Q',
'01J9BTDT6SXDS0EPM2XVYP6B95',
'01J9BTDTHYKQ2BSQQ4YM5E9SB0',
'01J9BTDTD9PK06M3MXFV723BZF',
'01J9BTDT31GF3NF66SYN9MFYWN',
'01J9BTDSNC90WQJ8HCNCM80E3M',
'01J9BTDT2X4TTTC5SRY04VBSN0',
'01J9BTDTCG6AQ1J942GGXZ8WFF',
'01J9BTDT1ZQ8N9G6EWZ6AEWJBQ',
'01J9BTDTSWWRPNFRBJ5RB98N56',]
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