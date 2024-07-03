import sys
import rich
import os
from parlai.core.opt import Opt
from parlai.core.agents import create_agent
import re
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
# paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "../"]
# sys.path.extend(paths)
sys.path.append("/home/wenkail/diplomacy")
# rich.print(sys.path)
from diplomacy_cicero.fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
# from diplomacy_cicero.fairdiplomacy import pydipcc
import json
import inspect

# dir_path = "/data/user_data/wenkail/"
# intent_model_path = "/data/user_data/wenkail/models/imitation_intent.opt"
# game_dir = "/data/user_data/wenkail/sotopia_diplomacy/filter_games/"




def imitation_order(intent_agent, countries, prompt):
    intent_agent.skip_generation = False
    intent_agent.observe(dict(text=prompt, episode_done=False))
    intent_response = intent_agent.act()
    return intent_response['text']

def setup_value_model(dir_path):
    value_model = BaseStrategyModelWrapper(dir_path + "models/rl_value_function.ckpt")
    return value_model
    
def get_phase_value(value_model, countries, game_file_path, phase, c1_predicted_moves, c2_actual_moves):
    # you can use pydipcc_game.rolled_back_to_phase_end(PHASE_NAME) to roll game state back to a specific phase, and you can also input moves here
    assert len(countries) == 2, "Function requires exactly two countries"
    # Order of the value model outputs are in alphabetical order
    country_list = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    pydipcc_game = pydipcc.Game.from_json(open(game_file_path, 'r').read())
    c1 = countries[0]
    c2 = countries[1]
    c1_predicted_cf = pydipcc_game.rolled_back_to_phase_end(phase['name'])
    c1_predicted_cf.set_orders(c1, c1_predicted_moves)
    c1_predicted_cf.set_orders(c2, c2_actual_moves)
    for power in country_list:
        if power not in [c1, c2]:
            c1_predicted_cf.set_orders(power, phase['orders'][power])
        
    c1_predicted_cf.process()
    actual = pydipcc_game.rolled_back_to_phase_end(phase['name'])

    values = value_model.forward_values([c1_predicted_cf], has_press=True, agent_power='ENGLAND')
    actual_values = value_model.forward_values([actual], has_press=True, agent_power='ENGLAND')

    countries_list = pydipcc_game.get_alive_powers()
    countries_value = []
    for country in countries:
        countries_value.append(values.tolist()[0][countries_list.index(country)])

    return countries_value

#  Add utils

