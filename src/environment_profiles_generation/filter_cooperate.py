import json
import rich 
import pdb
from tqdm import tqdm

def complete_support(phase):
    orders = phase['orders']
    complete_support_two_country = []
    for country, country_orders in orders.items():
        for order in country_orders:
            if ' S ' in order:
                parts = order.split(' S ')
                supporting_unit = parts[0]
                supported_action = parts[1]
                supported_unit = supported_action.split(' - ')[0]
                for c, u in phase['state']['units'].items():
                    if supported_unit in u:
                        supported_country = c
                # complete_support_order.append({
                #     'supporting_country': country,
                #     'supporting_unit': supporting_unit,
                #     'order': order,
                #     'supported_unit': supported_unit,
                #     'supported_country': supported_country
                # })
                if country != supported_country:
                    complete_support_two_country.append([country, supported_country])
    return complete_support_two_country

def complete_support(phase):
    orders = phase['orders']
    complete_support_two_country = []
    for country, country_orders in orders.items():
        for order in country_orders:
            if ' S ' in order:
                parts = order.split(' S ')
                supporting_unit = parts[0]
                supported_action = parts[1]
                supported_unit = supported_action.split(' - ')[0]
                for c, u in phase['state']['units'].items():
                    if supported_unit in u:
                        supported_country = c
                # complete_support_order.append({
                #     'supporting_country': country,
                #     'supporting_unit': supporting_unit,
                #     'order': order,
                #     'supported_unit': supported_unit,
                #     'supported_country': supported_country
                # })
                if country != supported_country:
                    complete_support_two_country.append([country, supported_country])
    return complete_support_two_country

def is_country_pair(country_pair, country_pairs_list):
    target_set = set(country_pair)
    for pair in country_pairs_list:
        if set(pair) == target_set:
            return True
    return False

    
def is_cooperate(game_dir, message_choice):
    game_path = f"{game_dir}{message_choice['game_id']}.json"
    with open(game_path, "r") as file:
        game = json.load(file)
    for phase in game['phases']:
        if message_choice["phase"] == phase['name']:
            current_phase = phase
    
    complte_support_list = complete_support(current_phase)
    mc_upper_countries = [c.upper() for c in message_choice['countries']]
    # pdb.set_trace()
    if is_country_pair(mc_upper_countries, complte_support_list):
        message_choice['is_cooperate'] = "yes"
    else:
        message_choice['is_cooperate'] = "no"
    
    return message_choice


def main():
    game_dir = "/data/user_data/wenkail/sotopia_diplomacy/whole_filter_games_100/"
    message_choice_phase_file = "choice_phase_list.json"
    with open(message_choice_phase_file, 'r') as f:
        mc = json.load(f)

    new_mc = []
    for m in tqdm(mc):
        new_mc.append(is_cooperate(game_dir, m))

    target_file = "choice_cooperate_phase_lis.json"
    with open(target_file, "w") as f:
        json.dump(new_mc, f, indent=2)

if __name__ == "__main__":
    main()