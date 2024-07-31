import argparse
import random
from rich import print
import sys
from tqdm import tqdm
from evaluate_utils import (
    get_intent_agent,
    get_predict_value,
    get_orders,
    get_one_predict_value,
    get_prev_state_value,
    get_real_value
)
import pdb
import json
def read_intent_response(res_path):
    return json.load(res_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", default="data/intent_response/intent_response_random_sample_100_games.json", type=str, required=False, help="Choose the intent response")
    parser.add_argument("--tgt_path", default="data/intent_value/intent_value_random_sample_100_games_real_move.json", type=str, required=False, help="Choose the intent response")
    parser.add_argument("--game_dir", default="/data/user_data/wenkail/sotopia_diplomacy/clean_global_sub_sample/", type=str, required=False, help="Choose the game direction")
    parser.add_argument("--move", action='store_true', help="Whether the intent model will move, give is move")
    parser.add_argument("--prev", action='store_true', help="Whether consider the previous state")
    parser.add_argument("--real", action='store_true', help="Whether consider the actual movement")
    args = parser.parse_args()

    country_list = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']


    # TODO: Replace Hard Code Country into Agents Match
    # countries = [args.c1, args.c2]
    with open(args.res_path, 'r') as f:
        intent_response = json.load(f)

    countries_value = []
    for episode in tqdm(intent_response, desc = "Processing Intent Response: "):
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
            country_value_dict[f"{c1}_units"] = episode[f"{c1}_units"]
            country_value_dict[f"{c2}_units"] = episode[f"{c2}_units"]
            if args.move:
                country_value_dict[c1] = get_one_predict_value(game_file_path, episode['phase_name'], c1, c2, predict1, move=args.move)
                country_value_dict[c2] = get_one_predict_value(game_file_path, episode['phase_name'], c2, c1, predict2, move=args.move)
                countries_value.append(country_value_dict)

            if args.prev:
                country_value_dict[c1] = get_prev_state_value(game_file_path, episode['phase_name'], c1, c2, predict1)
                country_value_dict[c2] = get_prev_state_value(game_file_path, episode['phase_name'], c2, c1, predict2)
                countries_value.append(country_value_dict)
            
            if args.real:
                country_value_dict[c1] = get_real_value(game_file_path, episode['phase_name'], c1, c2, predict1)
                country_value_dict[c2] = get_real_value(game_file_path, episode['phase_name'], c2, c1, predict2)
                countries_value.append(country_value_dict)

    json.dump(countries_value, open(args.tgt_path, 'w'), indent = 4)

if __name__ == "__main__":
    main()