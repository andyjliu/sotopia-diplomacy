import redis
import os
from redis_om import get_redis_connection
from redis import BusyLoadingError
import time
from diplomacy.engine.map import Map
import sys
sys.path.append('../')
from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from typing import Any
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from sotopia.server import LLM_Name
from template.scenario_template import Template
import json
from tqdm import tqdm
import random
import pdb

def add_agent_to_database(**kwargs: dict[str, Any]) -> None:
    agent = AgentProfile(**kwargs)
    agent.save()

def add_env_profile(**kwargs: dict[str, Any]) -> None:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()

def store_env_profile(game_id, game_phase, countries, tag):
    # countries = list(game_phase['state']['units'].keys())
    scenario, agent_goals = Template.get_format_scenario_template_goals(game_phase, countries, game_id)
    add_env_profile(
        game_id = game_id,
        phase_name = game_phase['name'],
        scenario=scenario,
        agent_goals = [social_goal for social_goal in agent_goals],
        agent_powers = countries,
        env_tag = tag
    )

def store_env_profile_with_previous(game_id, game_phase, countries, tag, game_dir):
    scenario, agent_goals = Template.get_previous_scenario_fewshot(game_phase, countries, game_id, game_dir)
    add_env_profile(
        game_id = game_id,
        phase_name = game_phase['name'],
        scenario=scenario,
        agent_goals = [social_goal for social_goal in agent_goals],
        agent_powers = countries,
        env_tag = tag
    )

def store_env_profile_with_previous_plausible(game_id, game_phase, countries, tag, game_dir, c1_plausible_move, c2_plausible_move):
    # pdb.set_trace()
    scenario, agent_goals = Template.get_previous_scenario_fewshot_plausible(game_phase, countries, game_id, game_dir, c1_plausible_move, c2_plausible_move)
    add_env_profile(
        game_id = game_id,
        phase_name = game_phase['name'],
        scenario=scenario,
        agent_goals = [social_goal for social_goal in agent_goals],
        agent_powers = countries,
        env_tag = tag
    )
    
def store_env_profile_with_actual_moves(game_id, game_phase, countries, tag, game_dir):
    # pdb.set_trace()
    scenario, agent_goals = Template.get_previous_scenario_fewshot_actual_moves(game_phase, countries, game_id, game_dir)
    add_env_profile(
        game_id = game_id,
        phase_name = game_phase['name'],
        scenario=scenario,
        agent_goals = [social_goal for social_goal in agent_goals],
        agent_powers = countries,
        env_tag = tag
    )

def get_actual_moves(phase, country):
    # Upper country
    uc = country.upper()
    return phase['orders'][uc]
    
def read_games_from_folder(game_folder):
    games = []
    for root, dirs, files in os.walk(game_folder):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                games.append(json.load(f))
    return games


def has_neighboring_units(country1, country2, unit_positions):
    def get_location(unit):
        return unit.split()[-1]
    map = Map(name='standard', use_cache=True)
    neighboring_units = []

    for unit1 in unit_positions[country1]:
        loc1 = get_location(unit1)
        for unit2 in unit_positions[country2]:
            loc2 = get_location(unit2)
            if loc2 in map.abut_list(loc1, incl_no_coast=True):
                neighboring_units.append((unit1, unit2))

    if neighboring_units:
        return neighboring_units
    else:
        return None

# Find adjunction games and phases, also remove phase ended without M
def adjacency_selection(games_dir, countries):
    games = read_games_from_folder(games_dir)
    game_phases = {}
    country1, country2 = countries
    country1 = country1.upper()
    country2 = country2.upper()
    
    for game in games:
        game_phases[game["id"]] = []
        for i in range(len(game['phases'])):
            result = has_neighboring_units(country1, country2, game['phases'][i]['state']['units'])
            if result:
                game_phases[game["id"]].append(game['phases'][i]['name'])
                game_phases[game["id"]]
    game_phases = {k: v for k, v in game_phases.items() if v}
    return game_phases

def random_country_adjacency_selection(games_dir, valid_countries):
    games = read_games_from_folder(games_dir)
    game_phases = []
    # pdb.set_trace()
    for game in games:
        for i in range(len(game['phases'])):
            result = None
            while result is None:
                sampled_countries = random.sample(valid_countries, 2)
                country1, country2 = sampled_countries
                country1 = country1.upper()
                country2 = country2.upper()
                result = has_neighboring_units(country1, country2, game['phases'][i]['state']['units'])
            game_phase_pair = {}
            game_phase_pair['game_id'] = game['id']
            game_phase_pair['countries'] = sampled_countries
            game_phase_pair['phase'] = game['phases'][i]['name']
            # game_phases[game["id"]]['country'] = sampled_countries
            # game_phases[game["id"]]['phase'] = game['phases'][i]['name']
            game_phases.append(game_phase_pair)
    # game_phases = {k: v for k, v in game_phases.items() if v}
    # pdb.set_trace()
    return game_phases

def get_env_pks(game_phases: dict[str, Any], games_num = None):
    selected_pks = []
    all_task_pks = list(EnvironmentProfile.all_pks())
    
    # Create a dictionary to map game_id and phase_name to pks
    env_dict = {}
    for pk in tqdm(all_task_pks, desc = "Processing env profile dictionary"):
        env = EnvironmentProfile.get(pk)
        if env.game_id not in env_dict:
            env_dict[env.game_id] = {}
        if env.phase_name not in env_dict[env.game_id]:
            env_dict[env.game_id][env.phase_name] = []
        env_dict[env.game_id][env.phase_name].append(pk)
    
    # Search using the pre-built dictionary
    for game_id, phases in tqdm(list(game_phases.items())[:games_num], desc = "Processing game phases"):
        if game_id in env_dict:
            for phase in phases:
                if phase in env_dict[game_id]:
                    selected_pks.extend(env_dict[game_id][phase])
    return selected_pks


def get_env_pk(game_phase):
    all_task_pks = list(EnvironmentProfile.all_pks())
    
    # Create a dictionary to map game_id and phase_name to pks
    env_dict = {}
    # pdb.set_trace()
    for pk in tqdm(all_task_pks, desc = "Processing env profile dictionary"):
        env = EnvironmentProfile.get(pk)
        # pdb.set_trace()
        if env.game_id == game_id and env.phase_name == pahse_id:
            return pk

def get_env_pks_by_tag(tag):
    all_task_pks = list(EnvironmentProfile.all_pks())
    pks_list = []
    for pk in tqdm(all_task_pks, desc = "Processing env profile dictionary"):
        env = EnvironmentProfile.get(pk)
        if env.env_tag == tag:
            pk_with_countries = {}
            pk_with_countries['uuid'] = pk
            pk_with_countries['countries'] = env.agent_powers
            pks_list.append(pk_with_countries)
    return pks_list


def get_previous_dialogue_unit(game_dir, game_id, phase_name, countries):

    upper_countries = [c.upper() for c in countries]
    with open(game_dir + game_id + ".json") as f:
        game = json.load(f)
    
    previous_phase = []

    for phase in game['phases']:
        if phase['name'] == phase_name:
            print(phase)
        else:
            previous_phase.append(phase)
    if len(previous_phase) > 2:
        previous_phase = previous_phase[-2:]
    dialogue_unit = ""
    for phase in previous_phase:
        dialogue_unit += f"{phase['name']}: \n"
        dialogue_unit += f"Dialogue Between Two Countries: \n"
        for message in phase['messages']:
            if message['sender'] in upper_countries and message['recipient'] in upper_countries:
                clean_message = message['message'].replace("\n", " ")
                dialogue_unit += f"{message['sender']} to {message['recipient']}: {clean_message}\n"
        dialogue_unit += f"Countries' Center in This Phase: \n"
        dialogue_unit += str(phase['state']['centers']) + "\n"
        dialogue_unit += f"Countries' Units in This Phase: \n"
        dialogue_unit += str(phase['state']['units']) + "\n"
        dialogue_unit += f"Countires' Order in This Phase: \n"
        dialogue_unit += str(phase['orders']) + '\n'

    return dialogue_unit
            
