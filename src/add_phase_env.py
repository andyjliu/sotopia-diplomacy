from profile_utils import store_env_profile, read_games_from_folder
import argparse
from tqdm import tqdm
import os
import json

def add_env_profiles(games, countries):
    for game in tqdm(games, desc="Processing games"):
        for game_phase in tqdm(game["phases"], desc = "Processing phases"):
            game_id = game['id']
            store_env_profile(game_phase, game_id, countries)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--games_folder", default="/data/user_data/wenkail/sotopia_diplomacy/filter_games", type=str, required=False, help = "Choose the evlauate model, the name can be seen in config")
    parser.add_argument("--country1", default="England",
                        type=str, required=False, help = "Choose the first country")
    parser.add_argument("--country2", default="Germany",
                        type=str, required=False, help = "Choose the second country")
    args = parser.parse_args()
    games = read_games_from_folder(args.games_folder)
    countries=[args.country1, args.country2]
    add_env_profiles(games, countries)

if __name__ == "__main__":
    main()