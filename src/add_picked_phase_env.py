from profile_utils import store_env_profile
import argparse
from tqdm import tqdm
import os
import json

def read_games_from_folder(game_folder):
    games = []
    for root, dirs, files in os.walk(game_folder):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                games.append(json.load(f))
    return games

def add_env_profiles(games):
    for game in tqdm(games, desc="Processing games"):
        for game_phase in tqdm(game["phases"], desc = "Processing phases"):
            game_id = game['id']
            store_env_profile(game_phase, game_id)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--games_folder", default="/data/user_data/wenkail/sotopia_diplomacy/filter_games", type=str, required=False, help = "Choose the evlauate model, the name can be seen in config")
    args = parser.parse_args()

    games = read_games_from_folder(args.games_folder)
    add_env_profiles(games)

if __name__ == "__main__":
    main()