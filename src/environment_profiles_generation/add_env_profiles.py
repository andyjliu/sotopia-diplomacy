import random
import rich
import sys
import json
from collections import Counter
import os
sys.path.append("/home/wenkail/diplomacy/sotopia-diplomacy/src")
from tqdm import tqdm
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
from profile_utils import store_env_profile_with_previous_plausible, store_env_profile_with_previous, store_env_profile_with_actual_moves
import argparse
from tqdm import tqdm
import os
import json
import pandas as pd
import pdb
import random

def parse_game_phases(plausible_move):
    parsed_moves = []
    for moves_string in plausible_move:
        # Remove the outer parentheses and split the moves
        moves = moves_string.strip("()").split(", ")
        # Remove the quotes from each move
        moves = [move.strip("'") for move in moves]
        # Join the moves with ', '
        parsed_move = ", ".join(moves)
        parsed_moves.append(parsed_move)
        
    return "".join(parsed_moves)

def add_env_profiles(games_dir, games_phases, tag, game_dir):
    for game_phases in tqdm(games_phases, desc="Processing games"):
        # TODO: Get game here
        # pdb.set_trace()
        game = json.load(open(games_dir + game_phases['game_id'] + '.json'))
        for phase in game['phases']:
            if phase['name'] == game_phases['phase']:
                # pdb.set_trace()
                store_env_profile_with_previous(game_phases['game_id'], phase, game_phases['countries'], tag, game_dir)

def add_env_profiles_with_plausible(games_dir, games_phases, tag, game_dir):
    for game_phases in tqdm(games_phases, desc="Processing games"):
        game = json.load(open(games_dir + game_phases['game_id'] + '.json'))
        for phase in game['phases']:
            if phase['name'] == game_phases['phase']:
                parse_c1_plausible_move = parse_game_phases(game_phases['c1_plausible_move'])
                parse_c2_plausible_move = parse_game_phases(game_phases['c2_plausible_move'])
                store_env_profile_with_previous_plausible(game_phases['game_id'], phase, game_phases['countries'], tag, game_dir, parse_c1_plausible_move, parse_c2_plausible_move)
                
def add_env_profiles_with_actual_moves(games_dir, games_phases, tag, game_dir):
    # TODO: add actual moves here
    # for game_phases in tqdm(games_phases, desc="Processing games"):
    for game_phases in tqdm(games_phases, desc="Processing games"):
        game = json.load(open(games_dir + game_phases['game_id'] + '.json'))
        for phase in game['phases']:
            if phase['name'] == game_phases['phase']:
                store_env_profile_with_actual_moves(game_phases['game_id'], phase, game_phases['countries'], tag, game_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--choice_file", default= "/home/wenkail/diplomacy/sotopia-diplomacy/src/environment_profiles_generation/choice_phases_list_with_cooperate_plausible_moves.json", type=str, required=False, help="The choice files")
    parser.add_argument("--tag", default="coop_with_flausible_v3", type=str, required=False, help="The tag name")
    parser.add_argument("--with_plausible_move", action="store_true", help="Whether with plausible moves")
    parser.add_argument("--with_actual_move", action="store_true", help="Whether with actual moves")
    args = parser.parse_args()

    with open(args.choice_file, 'r') as f:
        choice_phases_list = json.load(f)

    games_dir = "/data/user_data/wenkail/sotopia_diplomacy/whole_filter_games_100/"
    if args.with_plausible_move:
        add_env_profiles_with_plausible(games_dir, choice_phases_list, args.tag, games_dir)
    elif args.with_actual_move:
        add_env_profiles_with_actual_moves(games_dir, choice_phases_list, args.tag, games_dir)
    else:
        add_env_profiles(games_dir, choice_phases_list, args.tag, games_dir)


if __name__ == "__main__":
    main()
