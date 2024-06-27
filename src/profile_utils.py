import redis
import os
from redis_om import get_redis_connection
from redis import BusyLoadingError
import time
import sys
sys.path.append('../')
from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from typing import Any
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from sotopia.server import LLM_Name
from template.scenario_template import Template

def add_agent_to_database(**kwargs: dict[str, Any]) -> None:
    agent = AgentProfile(**kwargs)
    agent.save()

def add_env_profile(**kwargs: dict[str, Any]) -> None:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()

def store_env_profile(game_phase, game_id):
    countries = list(game_phase['state']['units'].keys())
    scenario, agent_goals = Template.get_format_scenario_template_goals(game_phase, countries, game_id)
    add_env_profile(
        game_id = game_id,
        phase_name = game_phase['name'],
        scenario=scenario,
        agent_goals = [social_goal for social_goal in agent_goals]
    )