from profile_utils import store_env_profile, read_games_from_folder, random_country_adjacency_selection
import argparse
from tqdm import tqdm
import os
import json
import pandas as pd
import pdb
import random

def add_env_profiles(games_dir, games_phases, tag):
    for game_phases in tqdm(games_phases, desc="Processing games"):
        # TODO: Get game here
        # pdb.set_trace()
        game = json.load(open(games_dir + game_phases['game_id'] + '.json'))
        for phase in game['phases']:
            if phase['name'] == game_phases['phase']:
                store_env_profile(game_phases['game_id'], phase, game_phases['countries'], tag)


        # for game_phase in tqdm(game["phases"], desc = "Processing phases"):
        #     game_id = game['id']

def add_specific_env(game, phase, countries):
    countries = [get_country(country).capitalize() for country in countries]
    store_env_profile(game, phase, countries)

def get_country(c):
    country_maps = {'A': 'AUSTRIA', 'E': 'ENGLAND', 'F': 'FRANCE', 'G': 'GERMANY', 'I': 'ITALY', 'R': 'RUSSIA', 'T': 'TURKEY'}
    return country_maps[c]

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--games_dir", default="/data/user_data/wenkail/sotopia_diplomacy/whole_filter_games_100/", type=str, required=False, help = "Choose the evlauate model, the name can be seen in config")
    # parser.add_argument("--human_annotation", default="/home/wenkail/diplomacy/sotopia-diplomacy/src/evaluate_intent/manual_annotations.csv", type=str, required=False, help="Choose the human annotation file")
    # parser.add_argument("--country1", default="England",
    #                     type=str, required=False, help = "Choose the first country")
    # parser.add_argument("--country2", default="Germany",
    #                     type=str, required=False, help = "Choose the second country")
    parser.add_argument("--tag", default="",
                        type=str, required=True, help = "Tag for the environment profile")
    
    args = parser.parse_args()
    # Add picked and random countries
    valid_countries = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
    random.seed(42)
    games_phases = None
    while games_phases is None:
        sampled_countries = random.sample(valid_countries, 2)
        # Try to select game phases based on the sampled countries
        # game_phases = adjacency_selection(args.games_dir, sampled_countries)
        games_phases = random_country_adjacency_selection(args.games_dir, valid_countries)
        # game_phases example: {'game_id': '3516', 'country': ['Italy', 'Austria'], 'phase': 'S1901M'}
    pdb.set_trace()
    add_env_profiles(args.games_dir, games_phases, args.tag)


    # Add special picked powers and special games (Old Version)
    # game_name = "53480" 
    # with open(args.game_file_path, 'r') as f:
    #     game = json.load(f)
    # pdb.set_trace()

    # ha = pd.read_csv(args.human_annotation)
    # for row in tqdm(ha.itertuples()):
    #     phase = {}
    #     phase['game'] = row[1]
    #     phase['phase'] = row[2]
    #     for i in game['phases']:
    #         if i['name'] == phase['phase']:
    #             current_phase = i
    #     phase['power1'] = row[3]
    #     phase['power2'] = row[4]
    #     add_specific_env(phase['game'], current_phase, [phase['power1'], phase['power2']])
    # countries=[args.country1, args.country2]


if __name__ == "__main__":
    main()