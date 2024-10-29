import sys
sys.path.append("../../../diplomacy_cicero")
from fairdiplomacy.agents import plausible_order_sampling as sampler
import fairdiplomacy
import fairdiplomacy.agents as agents
from conf import agents_cfgs
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from types import SimpleNamespace
from omegaconf import OmegaConf
from typing import Any, Dict
import rich
import os
from parlai.core.opt import Opt
from parlai.core.agents import create_agent
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy import pydipcc
import sys
import json
import inspect
import pdb
from tqdm import tqdm

DEFAULT_CFG: Dict[str, Any] = dict(
    model_path="MOCK",
    n_rollouts=10,
    device=-1,
    use_final_iter=0,
    rollouts_cfg=dict(max_rollout_length=0,),
    plausible_orders_cfg=dict(n_plausible_orders=10, batch_size=10, req_size=10,),
)

def main():
    game_dir = "/data/user_data/wenkail/sotopia_diplomacy/whole_filter_games_100/"
    dir_path = "/data/user_data/wenkail/"
    with open("choice_cooperate_phase_list.json", 'r') as f:
        data = json.load(f)

    # Definition of Agents Configs
    cfg = agents_cfgs.SearchBotAgent(**DEFAULT_CFG)
    base_strategy = BaseStrategyModelWrapper("/data/user_data/wenkail/models/base_strategy_model_0_5_policy.ckpt")
    order_sampler = sampler.PlausibleOrderSampler(
        # cfg.base_searchbot_cfg.plausible_orders_cfg,
        cfg.plausible_orders_cfg,
        base_strategy_model=base_strategy
    )
    new_data = []
    num = 0
    for g in tqdm(data):
        if g['is_cooperate'] == 'no':
            continue
        countries = [i.upper() for i in g['countries']]
        game_file_path = f"{game_dir}{g['game_id']}.json"
        pydipcc_game = pydipcc.Game.from_json(open(game_file_path, 'r').read())
        plausible_moves_c1 = order_sampler.sample_orders(pydipcc_game, agent_power=countries[0], extra_plausible_orders=None) 
        plausible_moves_c2 = order_sampler.sample_orders(pydipcc_game, agent_power=countries[1], extra_plausible_orders=None) 
        if plausible_moves_c1[f"{countries[0]}"] == {(): 1.0} or plausible_moves_c2[f"{countries[1]}"] == {(): 1.0}:
            continue
        game = {}
        game['game_id'] = g['game_id']
        game['phase'] = g['phase']
        game['countries'] = g['countries']
        game['message_count'] = g['message_count']
        game['c1_plausible_move'] = {str(key): value for key, value in plausible_moves_c1[f"{countries[0]}"].items()}
        game['c2_plausible_move'] = {str(key): value for key, value in plausible_moves_c2[f"{countries[1]}"].items()}
        new_data.append(game)
        num += 1

    # import pdb
    # pdb.set_trace()
    print(num)
    with open('choice_pahses_list_with_cooperate_plausible_moves.json', 'w') as f:
        json.dump(new_data, f, indent=2)

if __name__ == "__main__":
    main()