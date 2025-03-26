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
    # import pdb; pdb.set_trace()
    if len(cities) != 2:
        return ""
    
    for turn in turns:
        match = re.match(r'(\w+) said: "(.*?)"$', turn.strip(), re.DOTALL)
        if match:
            speaker, message = match.groups()
            try:
                recipient = next(city for city in cities if city != speaker)
            except StopIteration:
                print(cities)
                rich.print(turns)
                print(f"Error:\nSpeaker: {speaker} \nCould not find recipient for message:\n{message}")
                print(f"cities: {cities}\nspeaker: {speaker}")
            formatted_message = f"{message_count} {speaker.upper()} -> {recipient.upper()}: {message}"
            formatted_messages.append(formatted_message)
            message_count += 1
    
    return "\n".join(formatted_messages)



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
# def format_diplomacy_data(scenario):
#     # pdb.set_trace()
#     centers_match = re.search(r"centers: (\{.*?\})", scenario)
#     units_match = re.search(r"units: (\{.*?\})", scenario)
#     import pdb; pdb.set_trace()
#     if not centers_match or not units_match:
#         return "Error: Could not find centers or units data in the scenario."
#     centers = json.loads(centers_match.group(1).replace("'", '"'))
#     units = json.loads(units_match.group(1).replace("'", '"'))
#     def format_country_data(data, separator):
#         return '; '.join([f"{country}: {separator.join(items)}" for country, items in data.items()])
#     units_formatted = format_country_data(units, ', ')
#     centers_formatted = format_country_data(centers, ', ')
#     return f"units: {units_formatted}\ncenters: {centers_formatted}"

def format_diplomacy_data(scenario_text):
    """
    从给定的 Diplomacy 场面文本中，解析出Centers和Units的信息，
    并返回形如：
        centers: XX
        units: XX
    的最终字符串。
    """

    # ------------------------------------------------------
    # 1) 分别获取 Centers 块 和 Units 块（包含换行）
    #    注意要用 re.DOTALL 可以让 '.' 匹配到换行符
    # ------------------------------------------------------
    centers_pattern = r"Centers:\s*(.*?)\n\n\s*Units:"
    units_pattern   = r"Units:\s*(.*)"

    centers_block_match = re.search(centers_pattern, scenario_text, re.DOTALL)
    units_block_match   = re.search(units_pattern, scenario_text, re.DOTALL)

    if not centers_block_match or not units_block_match:
        return "无法找到 Centers 或 Units 区块，请检查输入格式。"

    centers_block = centers_block_match.group(1).strip()
    units_block   = units_block_match.group(1).strip()

    # ------------------------------------------------------
    # 2) 分别解析 Centers 块 和 Units 块
    # ------------------------------------------------------
    # 解析后打算用两个字典来存储，比如：
    # centers_dict = {
    #     "AUSTRIA": ["BUD", "TRI", "VIE", ...],
    #     "ENGLAND": ["EDI", "LON", "LVP", ...],
    #     ...
    # }
    # units_dict = {
    #     "AUSTRIA": ["A SER", "A TYR", "A RUM", ...],
    #     "ENGLAND": ["F NWY", "F DEN", "A LON", ...],
    #     ...
    # }
    centers_dict = {}
    units_dict = {}

    # 解析 centers_block
    # 每一行格式类似： AUSTRIA: BUD, TRI, VIE, SER, ...
    for line in centers_block.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            # 先按冒号分为两部分 [国家, 剩余部分]
            country, centers_str = line.split(":", 1)
            country = country.strip()
            centers_str = centers_str.strip()
            # 再按逗号分割
            centers_list = [x.strip() for x in centers_str.split(",")]
            centers_dict[country] = centers_list
        except ValueError:
            import pdb; pdb.set_trace()
            print(f"Error parsing centers line: {line}")
            continue

    # 解析 units_block
    for line in units_block.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            # 先按冒号分为两部分 [国家, 剩余部分]
            country, units_str = line.split(":", 1)
            country = country.strip()
            units_str = units_str.strip()
            # 再按逗号分割
            units_list = [x.strip() for x in units_str.split(",")]
            units_dict[country] = units_list
        except ValueError:
            # import pdb; pdb.set_trace()
            print(f"Error parsing units line: {line}")
            continue

    # ------------------------------------------------------
    # 3) 将 centers_dict 和 units_dict 整理回你想要的格式
    #    例如：
    #    centers: AUSTRIA: BUD, TRI, VIE; ENGLAND: EDI, LON, LVP; ...
    #    units:   AUSTRIA: A BUD, A VIE, F TRI; ENGLAND: A LVP, ...
    # ------------------------------------------------------
    # 拼出 centers 的字符串
    centers_parts = []
    for country, centers_list in centers_dict.items():
        centers_str = ", ".join(centers_list)
        centers_parts.append(f"{country}: {centers_str}")
    centers_result = "; ".join(centers_parts)

    # 拼出 units 的字符串
    units_parts = []
    for country, units_list in units_dict.items():
        units_str = ", ".join(units_list)
        units_parts.append(f"{country}: {units_str}")
    units_result = "; ".join(units_parts)

    # 最终结果
    final_output = f"centers: {centers_result}\nunits: {units_result}"
    return final_output

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

