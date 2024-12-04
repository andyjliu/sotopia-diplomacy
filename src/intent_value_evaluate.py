import argparse
import random
from rich import print
import sys
from tqdm import tqdm
import os
from evaluate_utils import (
    get_intent_agent,
    get_predict_value,
    get_orders,
    get_one_predict_value,
    get_prev_state_value,
    get_actual_value,
    get_task_eval
)
import pdb
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", default="data/intent_response/taskeval_1757_intent_prediction_gpt4.json", type=str, required=False, help="Choose the intent response")
    parser.add_argument("--tgt_path", default="data/intent_value/taskeval_1757_intent_prediction_gpt4.json", type=str, required=False, help="Choose the intent response")
    parser.add_argument("--game_dir", default="/data/user_data/wenkail/sotopia_diplomacy/whole_filter_games_100/", type=str, required=False, help="Choose the game direction")
    # parser.add_argument("--pred_move", action='store_true', help="Whether the intent model will move, give is move")
    # parser.add_argument("--prev", action='store_true', help="Whether consider the previous state")
    # parser.add_argument("--actual_move", action='store_true', help="Whether consider the actual movement")
    parser.add_argument("--move", action='store_true', help="Whether consider the previous state")
    parser.add_argument("--task_eval", action='store_true', help="Whether use the task eval evalutation")
    args = parser.parse_args()

    country_list = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']

    if args.res_path.endswith("json"):
        with open(args.res_path, 'r') as f:
            intent_response = json.load(f)
    elif args.res_path.endswith("jsonl"):
        intent_response = []
        with open(args.res_path, 'r') as f:
            for line in f:
                intent_response.append(json.loads(line))
                
    if type(intent_response) != list:
        intent_response = [intent_response]
        
    countries_value = []
    # for episode in tqdm(intent_response, desc = "Processing Intent Response: "):
    for episode in intent_response:
        countries = []
        if episode['phase_name'].endswith("M"):
            countries = episode['countries']
            country_value_dict = {}
            game_file_path = args.game_dir + episode["game_id"] + ".json"
            country_value_dict["game_id"] = episode["game_id"]
            country_value_dict["phase_name"] = episode["phase_name"]
            country_value_dict["env_uuid"] = episode["env_uuid"]
            c1 = countries[0]
            c2 = countries[1]
            predict1 = get_orders(episode[f"{c1}_response"]['text'], episode["phase_name"], c1)
            predict2 = get_orders(episode[f"{c2}_response"]['text'], episode["phase_name"], c2)
            country_value_dict[f"{c1}_original_predict"] = predict1
            country_value_dict[f"{c2}_original_predict"] = predict2
            country_value_dict[f"{c1}_units"] = episode[f"{c1}_units"]
            country_value_dict[f"{c2}_units"] = episode[f"{c2}_units"]
            country_value_dict["intent_dialogue"] = episode["intent_dialogue"]

            if args.task_eval:
                country_value_dict[f"{c1}_actual_movement"], country_value_dict[f"{c1}_parse_predict"], country_value_dict[c1] = get_task_eval(game_file_path, episode['phase_name'], c1, c2, predict1, args.move)
                country_value_dict[f"{c2}_actual_movement"], country_value_dict[f"{c2}_parse_predict"], country_value_dict[c2] = get_task_eval(game_file_path, episode['phase_name'], c2, c1, predict2, args.move)
                countries_value.append(country_value_dict)

            # if args.pred_move:
            #     country_value_dict[f"{c1}_parse_predict"], country_value_dict[c1] = get_one_predict_value(game_file_path, episode['phase_name'], c1, c2, predict1, move=args.pred_move)
            #     country_value_dict[f"{c2}_parse_predict"], country_value_dict[c2] = get_one_predict_value(game_file_path, episode['phase_name'], c2, c1, predict2, move=args.pred_move)
            #     countries_value.append(country_value_dict)

            # if args.actual_move:
            #     country_value_dict[c1] = get_actual_value(game_file_path, episode['phase_name'], c1, c2)
            #     country_value_dict[c2] = get_actual_value(game_file_path, episode['phase_name'], c2, c1)
            #     countries_value.append(country_value_dict)

            # if args.prev:
            #     country_value_dict[c1] = get_prev_state_value(game_file_path, episode['phase_name'], c1, c2, predict1)
            #     country_value_dict[c2] = get_prev_state_value(game_file_path, episode['phase_name'], c2, c1, predict2)
            #     countries_value.append(country_value_dict)
            
            
    if not os.path.exists(os.path.dirname(args.tgt_path)):
        os.makedirs(os.path.dirname(args.tgt_path))

    json.dump(countries_value, open(args.tgt_path, 'w'), indent = 4)

if __name__ == "__main__":
    main()