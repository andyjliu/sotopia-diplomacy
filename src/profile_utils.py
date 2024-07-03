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


def add_agent_to_database(**kwargs: dict[str, Any]) -> None:
    agent = AgentProfile(**kwargs)
    agent.save()

def add_env_profile(**kwargs: dict[str, Any]) -> None:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()

def store_env_profile(game_phase, game_id, countries):
    # countries = list(game_phase['state']['units'].keys())
    scenario, agent_goals = Template.get_format_scenario_template_goals(game_phase, countries, game_id)
    add_env_profile(
        game_id = game_id,
        phase_name = game_phase['name'],
        scenario=scenario,
        agent_goals = [social_goal for social_goal in agent_goals]
    )

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

# Find adjunction games and phases
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
    game_phases = {k: v for k, v in game_phases.items() if v}
    return game_phases

def get_env_pks(game_phases: dict[str, Any], games_num = 2):
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
    for game_id, phases in tqdm(list(game_phases.items())[games_num:games_num+4], desc = "Processing game phases"):
        if game_id in env_dict:
            for phase in phases:
                if phase in env_dict[game_id]:
                    selected_pks.extend(env_dict[game_id][phase])
    return selected_pks