import random
import rich
import sys
import json
from collections import Counter
import os
sys.path.append("/home/wenkail/diplomacy/sotopia-diplomacy/src")
from tqdm import tqdm
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
from profile_utils import store_env_profile_with_previous_plausible
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
                parse_c1_plausible_move = parse_game_phases(game_phases['c1_plausible_move'])
                parse_c2_plausible_move = parse_game_phases(game_phases['c2_plausible_move'])
                # pdb.set_trace()
                store_env_profile_with_previous_plausible(game_phases['game_id'], phase, game_phases['countries'], tag, game_dir, parse_c1_plausible_move, parse_c2_plausible_move)

def main():
    file = "/home/wenkail/diplomacy/sotopia-diplomacy/src/environment_profiles_generation/choice_pahse_list_with_plausible_moves.json"
    with open(file, 'r') as f:
        choice_phases_list_plausible = json.load(f)
    games_dir = "/data/user_data/wenkail/sotopia_diplomacy/whole_filter_games_100/"
    tag = "taskeval_fewshot_plausible_v2"
    add_env_profiles(games_dir, choice_phases_list_plausible, tag, games_dir)

if __name__ == "__main__":
    main()
