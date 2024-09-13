import sys
import rich
import os
sys.path.append("/home/wenkail/diplomacy/diplomacy_cicero")
from parlai.core.opt import Opt
from parlai.core.agents import create_agent
import re

from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy import pydipcc
from diplomacy.engine.renderer import Renderer
import json
import inspect
import rich
import pdb



def get_intent_agent(dir_path, intent_model_path):
    intent_model_opt = Opt.load_init(intent_model_path)
    intent_model_opt['model_file'] = dir_path + intent_model_opt['model_file']
    intent_model_opt['dict_file'] = dir_path + intent_model_opt['dict_file']
    intent_agent = create_agent(intent_model_opt)
    return intent_agent

def imitation_order(intent_agent, countries, prompt):
    intent_agent.skip_generation = False
    intent_agent.observe(dict(text=prompt, episode_done=False))
    intent_response = intent_agent.act()
    return intent_response['text']

def setup_value_model(dir_path):
    value_model = BaseStrategyModelWrapper(dir_path + "models/rl_value_function.ckpt")
    return value_model

def get_orders(data, phase_name, country_name):
    country_name = country_name.upper()
    lines = data.split('\n')
    current_phase = ''
    current_country = ''
    result = ''

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if ':' not in line:
            current_phase = line
            continue
        
        country, orders = line.split(':', 1)
        current_country = country.strip()
        
        if current_phase == phase_name and current_country == country_name:
            return orders.strip()
    
    return ''

def parse_predicted_moves(pred, act):
    units = [' '.join(x.split()[:2]) for x in act]
    pred = pred.split('; ')
    pred_dict = {}
    for move in pred:
        unit = ' '.join(move.split(' ')[:2])
        pred_dict[unit] = move.split(' ')[2:]
    pred_list = []
    for unit in units:
        if unit in pred_dict:
            pred_move = pred_dict[unit]

            if pred_move == ['H']: # hold
                pred_list.append(f"{unit} H")

            elif pred_move[0] == 'S':
                if len(pred_move) == 3: # support hold
                    supported = " ".join(pred_move[1:])
                    pred_list.append(f"{unit} S {supported}")
                elif len(pred_move) == 4: # support move
                    supported = " ".join(pred_move[1:-1])
                    destination = pred_move[-1]
                    pred_list.append(f"{unit} S {supported} - {destination}")

            elif pred_move[0] == 'C': # convoy
                convoyed = " ".join(pred_move[1:-1])
                destination = pred_move[-1]
                pred_list.append(f"{unit} C {convoyed} - {destination}")

            elif len(pred_move) == 1 or pred_move[-1] == 'VIA': # move
                destination = " ".join(pred_move)
                pred_list.append(f"{unit} - {destination}")

            else:
                pass # couldn't find move, fill with actual move

        else:
            # find equivalent move in act
            for move in act:
                if unit == ' '.join(move.split(' ')[:2]):
                    pred_list.append(move)
                    break
    return(pred_list)

def get_pydipcc_game(game_file_path):
    try:
        game = pydipcc.Game.from_json(open(game_file_path, 'r').read()) 
        return game
    except Exception as e:
        print(f"Failed to load game from {game_file_path}: {e}")
        return None

def get_value_model(model_path="/data/user_data/wenkail/models/rl_value_function.ckpt"):
    return BaseStrategyModelWrapper(model_path)

def get_phase(game_file_path, phase_name):
    try:
        with open(game_file_path, 'r') as f:
                game = json.load(f)  
        for phase in game['phases']:
            if phase['name'] == phase_name:
                return phase
    except FileNotFoundError:
        print("Error in Loading File")

def get_country_value(values, country, countries_list, with_index = False):
    if with_index == False:
        return (country, values[countries_list.index(country)])
    else:
        pdb.set_trace()
        return (country, values.tolist()[countries_list.index(country)])

def get_predict_value(game_file_path, phase_name, c1, c2, c1_predicted_moves, move):
    c1 = c1.upper()
    c2 = c2.upper()
    # pdb.set_trace()

    # Here this game should be pydipcc
    country_list = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    phase = get_phase(game_file_path, phase_name)
    current_phase = phase['name']
    if get_pydipcc_game(game_file_path) is not None:
        pydipcc_game = get_pydipcc_game(game_file_path)
        c1_predicted_cf = pydipcc_game.rolled_back_to_phase_start(current_phase)
        new_c1_predicted_moves = parse_predicted_moves(pydipcc_game, c1, c1_predicted_moves, phase['orders'][c1])
        if move:
            c1_predicted_cf.set_orders(c1, new_c1_predicted_moves)
        else:
            prev_phase_name = c1_predicted_cf.get_prev_phase(current_phase)
            # Get the previous phase here is move is equal to false
            prev_phase = get_phase(game_file_path, prev_phase_name)
            # c1_predicted_cf.set_orders(c1, prev_phase['state']['units'][c1])

        for power in country_list:
            if power not in [c1]:
                c1_predicted_cf.set_orders(power, phase['orders'][power])

        c1_predicted_cf.process()
        actual = pydipcc_game.rolled_back_to_phase_end(current_phase)
        value_model = get_value_model()
        if move:
            pred_values = value_model.forward_values([c1_predicted_cf], has_press=True, agent_power=c1)
        else:
            prev_pydipcc_game = get_pydipcc_game(game_file_path).rolled_back_to_phase_end(prev_phase_name)
            # This is using the previous values from the previous phase
            pred_values = value_model.get_values(prev_pydipcc_game, has_press=True, agent_power=c1)
        actual_values = value_model.forward_values([actual], has_press=True, agent_power=c1)
        # # Here compare with country_list, this is the alive powers
        # alive_countries_list = pydipcc_game.get_alive_powers()
        # Init
        countries_value = {}
        countries_value['whole_predict'] = []
        countries_value['whole_actual'] = []
        countries_value['predict'] = []
        countries_value['actual'] = []
        countries_value['whole_predict'].append(pred_values.tolist())
        countries_value['whole_actual'].append(actual_values.tolist())
        for country in [c1,c2]:
            countries_value['predict'].append(get_country_value(pred_values, country, country_list))
            countries_value['actual'].append(get_country_value(actual_values, country, country_list))
        return countries_value
    return None

def get_one_predict_value(game_file_path, phase_name, c1, c2, c1_predicted_moves, move):
    c1 = c1.upper()
    c2 = c2.upper()
    # Here this game should be pydipcc
    country_list = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    phase = get_phase(game_file_path, phase_name)
    current_phase = phase['name']
    if get_pydipcc_game(game_file_path) is not None:
        pydipcc_game = get_pydipcc_game(game_file_path)
        c1_predicted_cf = pydipcc_game.rolled_back_to_phase_start(current_phase)
        new_c1_predicted_moves = parse_predicted_moves(pydipcc_game, c1, c1_predicted_moves, phase['orders'][c1])
        if move:
            c1_predicted_cf.set_orders(c1, new_c1_predicted_moves)
        else:
            prev_phase_name = c1_predicted_cf.get_prev_phase(current_phase)
            # Get the previous phase here is move is equal to false
            prev_phase = get_phase(game_file_path, prev_phase_name)

        for power in country_list:
            if power not in [c1]:
                c1_predicted_cf.set_orders(power, phase['orders'][power])

        c1_predicted_cf.process()
        
        value_model = get_value_model()
        if move:
            pred_values = value_model.forward_values([c1_predicted_cf], has_press=True, agent_power=c1)
            pred_values = pred_values.tolist()[0]
        else:
            prev_pydipcc_game = get_pydipcc_game(game_file_path).rolled_back_to_phase_end(prev_phase_name)
            # This is using the previous values from the previous phase
            pred_values = value_model.get_values(prev_pydipcc_game, has_press=True, agent_power=c1)
            pred_values = pred_values.tolist()
        # Here compare with country_list, this is the alive powers
        countries_value = {}
        countries_value['whole_predict'] = []
        # countries_value['predict'] = []
        countries_value['whole_predict'].append(pred_values)
        
        for country in [c1,c2]:
            countries_value[country] = get_country_value(pred_values, country, country_list, with_index=False)
        return new_c1_predicted_moves, countries_value
    return None, None

def get_prev_state_value(game_file_path, phase_name, c1, c2, c1_predicted_moves):
    c1 = c1.upper()
    c2 = c2.upper()
    # Here this game should be pydipcc

    country_list = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    phase = get_phase(game_file_path, phase_name)
    current_phase = phase['name']

    if get_pydipcc_game(game_file_path) is not None:
        pydipcc_game = get_pydipcc_game(game_file_path)
        c1_predicted_cf = pydipcc_game.rolled_back_to_phase_start(current_phase)
        new_c1_predicted_moves = parse_predicted_moves(pydipcc_game, c1, c1_predicted_moves, phase['orders'][c1])
        c1_predicted_cf.set_orders(c1, new_c1_predicted_moves)
        prev_phase_name = c1_predicted_cf.get_prev_phase(current_phase)
        # Get the previous phase here is move is equal to false
        prev_phase = get_phase(game_file_path, prev_phase_name)

        # for power in country_list:
        #     if power not in [c1]:
        #         c1_predicted_cf.set_orders(power, phase['orders'][power])

        # c1_predicted_cf.process()

        value_model = get_value_model()
        # pred_values = value_model.forward_values([c1_predicted_cf], has_press=True, agent_power=c1)
        prev_pydipcc_game = get_pydipcc_game(game_file_path).rolled_back_to_phase_end(prev_phase_name)
        # This is using the previous values from the previous phase
        pred_values = value_model.get_values(prev_pydipcc_game, has_press=True, agent_power=c1)

        # Here compare with country_list, this is the alive powers
        countries_value = {}
        countries_value['whole_predict'] = []
        # countries_value['predict'] = []
        countries_value['whole_predict'].append(pred_values.tolist())

        for country in [c1,c2]:
            countries_value[country] = get_country_value(pred_values, country, country_list, with_index=True)
        return countries_value
    return None

def get_actual_value(game_file_path, phase_name, c1, c2):
    c1 = c1.upper()
    c2 = c2.upper()
    # Here this game should be pydipcc
    country_list = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    phase = get_phase(game_file_path, phase_name)
    current_phase = phase['name']
    if get_pydipcc_game(game_file_path) is not None:
        pydipcc_game = get_pydipcc_game(game_file_path)
        # c1_predicted_cf = pydipcc_game.extract_first_stage_results(current_phase)
        c1_predicted_cf = pydipcc_game.rolled_back_to_phase_start(current_phase)
        for power in country_list:
            c1_predicted_cf.set_orders(power, phase['orders'][power])

        c1_predicted_cf.process()
        value_model = get_value_model()
        pred_values = value_model.forward_values([c1_predicted_cf], has_press=True, agent_power=c1)
        # Here compare with country_list, this is the alive powers
        countries_value = {}
        countries_value['whole_predict'] = []
        countries_value['whole_predict'].append(pred_values.tolist())
        pred_values = pred_values.tolist()[0]
        for country in [c1,c2]:
            countries_value[country] = get_country_value(pred_values, country, country_list, with_index=False)
        return countries_value
    return None


# TODO: Finished Task Eval by tonight
def get_task_eval(game_file_path, phase_name, c1, c2, c1_predicted_moves, move):
    c1 = c1.upper()
    c2 = c2.upper()
    country_list = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    phase = get_phase(game_file_path, phase_name)
    current_phase_name = phase['name']
    if get_pydipcc_game(game_file_path) is not None:
        value_model = get_value_model()
        pydipcc_game = get_pydipcc_game(game_file_path)
        c1_predicted_cf = pydipcc_game.rolled_back_to_phase_start(current_phase_name)
        actual_movement = phase['orders'][c1]
        new_c1_predicted_moves = parse_predicted_moves(c1_predicted_moves, actual_movement)
        prev_values = value_model.get_values(c1_predicted_cf, has_press=True, agent_power=c1)
        if move:
            c1_predicted_cf.set_orders(c1, new_c1_predicted_moves)
            for power in country_list:
                if power not in [c1]:
                    c1_predicted_cf.set_orders(power, phase['orders'][power])
            c1_predicted_cf.process()
            # render = Renderer(c1_predicted_cf)
            # render.render(output_path = '/home/wenkail/diplomacy/sotopia-diplomacy/src/map.xml')
            pred_values = value_model.forward_values([c1_predicted_cf], has_press=True, agent_power=c1)
            prev_values = prev_values.tolist()
            pred_values = pred_values.tolist()[0]
        else:
            c1_predicted_cf.set_orders(c1, new_c1_predicted_moves)
            for power in country_list:
                c1_predicted_cf.set_orders(power, phase['orders'][power])
            c1_predicted_cf.process()
            pred_values = value_model.forward_values([c1_predicted_cf], has_press=True, agent_power=c1)
            prev_values = prev_values.tolist()
            pred_values = pred_values.tolist()[0]
        task_eval_values = [pred_values[i] - prev_values[i] for i in range(len(pred_values))]
        countries_value = {}
        countries_value['whole_task_eval'] = task_eval_values
            
        for country in [c1,c2]:
            countries_value[country] = get_country_value(task_eval_values, country, country_list, with_index=False)
            
        return actual_movement, new_c1_predicted_moves, countries_value
    return None, None, None
        
        
        
        