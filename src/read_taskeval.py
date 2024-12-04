import json
import numpy as np
import rich
import argparse
import pdb


def get_score_with_env(countries, prediction):
    scores = []
    for c in countries:
        scores.append({c:prediction[c][c.upper()][1]})
    return scores

def get_score_dict_from_dialogue(countries_list, dialogue):
    score_list = []
    for a in dialogue:
        keys = []
        for vkeys in a.keys():
            if vkeys in countries_list:
                keys.append(vkeys)
        score_list.append(get_score_with_env(keys, prediction=a))
    return score_list

def main():
    countries_list = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=True, help = "Choose a taskeval file")
    args = parser.parse_args()
    
    with open(args.file_name, 'r') as f:
        data = json.load(f)

    result = get_score_dict_from_dialogue(countries_list, data)[0]
    import pdb
    for item in result:
        # pdb.set_trace()
        for c, score in item.items():
            rich.print(f"Country: [bold purple]{c}[/bold purple] gain [bold]TaskEval[/bold] score: {score:.5f}")
        

if __name__ == "__main__":
    main()