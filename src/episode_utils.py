import sys
import rich
import os
import re
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
# paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "../"]
# sys.path.extend(paths)
# rich.print(sys.path)
# from diplomacy_cicero.fairdiplomacy import pydipcc
import json
from tqdm import tqdm
import pdb
import inspect

def get_game_phase_env_from_episode(episodelog):
    env = EnvironmentProfile.get(episodelog.environment)
    return env


def get_countries_from_agent(agents):
    countries = []
    pks = list(AgentProfile.all_pks())
    for agent in agents:
        for pk in pks:
            if agent == pk:
                countries.append(AgentProfile.get(pk).country)
    return countries

def process_conversation(data):
    # Here the data should be episodelog.messages
    formatted_messages = []
    
    for items in data:
        for item in items:
            if item[0] == 'Environment' and 'Turn #' in item[2]:
            # Extract turn number, speaker, and message from Environment tuple
                match = re.search(rf'Turn #(\d+): ({item[1]}) said: "(.*?)"', item[2])
                if match:
                    turn_number, speaker, message = match.groups()
                    formatted_message = f"Turn #{turn_number}\n{speaker} said: \"{message}\""
                    formatted_messages.append(formatted_message)
        # elif item[0] != 'Environment' and item[2].startswith('said:'):
        #     # Extract message from non-Environment tuple
        #     speaker = item[0]
        #     message = item[2][6:].strip().strip('"')  # Remove 'said: ' and surrounding quotes
        #     turn_number = len(formatted_messages) + 1
        #     formatted_message = f"Turn #{turn_number}\n{speaker} said: \"{message}\""
        #     formatted_messages.append(formatted_message)

    return "\n".join(formatted_messages)

def process_conversation_to_intent(text):
    turns = re.split(r'Turn #\d+\n', text)[1:]
    
    cities = set()
    for turn in turns:
        match = re.match(r'(\w+) said:', turn.strip())
        if match:
            cities.add(match.group(1))
    
    formatted_messages = []
    message_count = 0
    
    for turn in turns:
        match = re.match(r'(\w+) said: "(.*?)"$', turn.strip(), re.DOTALL)
        if match:
            speaker, message = match.groups()
            recipient = next(city for city in cities if city != speaker)
            
            formatted_message = f"{message_count} {speaker.upper()} -> {recipient.upper()}: {message}"
            formatted_messages.append(formatted_message)
            message_count += 1
    
    return "\n".join(formatted_messages)

import re

def get_country_from_name(name, profiles):
    for profile in profiles:
        if name in [profile.first_name, profile.last_name, f"{profile.first_name} {profile.last_name}"]:
            return profile.country
    return name

def replace_names_with_countries(text, profiles):
    def replace_name(match):
        full_name = match.group(0)
        return get_country_from_name(full_name, profiles)
    
    # Generate the regex pattern for full names to be replaced
    name_pattern = '|'.join([re.escape(f"{p.first_name} {p.last_name}") for p in profiles] + 
                            [re.escape(p.first_name) for p in profiles] +
                            [re.escape(p.last_name) for p in profiles])
    
    # Replace names with country equivalents
    replaced_text = re.sub(name_pattern, replace_name, text)

    # Correctly replace duplicate words that exactly match and are separated by space
    replaced_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', replaced_text)

    return replaced_text


# TODO: Should be modified after can gain the real data from redis
def format_diplomacy_data(scenario):
    # pdb.set_trace()
    centers_match = re.search(r"centers: (\{.*?\})", scenario)
    units_match = re.search(r"units: (\{.*?\})", scenario)
    if not centers_match or not units_match:
        return "Error: Could not find centers or units data in the scenario."
    centers = json.loads(centers_match.group(1).replace("'", '"'))
    units = json.loads(units_match.group(1).replace("'", '"'))
    def format_country_data(data, separator):
        return '; '.join([f"{country}: {separator.join(items)}" for country, items in data.items()])
    units_formatted = format_country_data(units, ', ')
    centers_formatted = format_country_data(centers, ', ')
    return f"units: {units_formatted}\ncenters: {centers_formatted}"

def get_phases_from_envs(games_dir, envs):
    file_paths = []
    games = []
    phases = []
    for root, dirs, files in os.walk(games_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            games.append(json.load(f))

    for env in tqdm(envs):
        for game in games:
            if game['id'] == env.game_id:
                for phase in game['phases']:
                    if phase['name'] == env.phase_name:
                        phases.append(phase)
    
    return phases


def get_actual_dialogue(env, phase):
    countries = [c.upper() for c in env.agent_powers]
    store_message = []
    for message in phase['messages']:
        if message['sender'] in countries and message['recipient'] in countries:
            store_message.append(message)
        
    return store_message

