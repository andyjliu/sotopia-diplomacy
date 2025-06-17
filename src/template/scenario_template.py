import rich
import json
class Template:

    previous_dialogue_instruction = '''Here is the previous dialogue, and the previous countries' centers and units between the two countries '''

    center_instruction = '''This is the information of the countries' centers, and you will playing as the given country as you act: \n centers: '''

    unit_instruction = '''This is the information of the countries' units, and you will playing as the given country as you act: \n units: '''

    finetune_instruction = '''You are playing diplomacy game, you will negotiate with the other player so that it will play moves that are beneficial to your board position, either this turn or in future turns.\n'''
    
    background_instruction = '''You are in a diplomacy game, and you will play the role of the given country. \n\n'''
    
    diplomacy_intro = '''Diplomacy is played on a detailed map representing Europe at the start of the 20th century, divided into clearly defined territories and seas.  Each territory is categorized as either a land province or a sea zone, with some land territories bordering seas along their coasts.  Certain provinces contain cities designated as supply centers, and these are essential to expanding and maintaining your forces.  Each nation begins with several units—either armies or fleets—placed in their respective home territories.

An army unit can occupy and move through land provinces only, advancing into adjacent territories connected by a direct border.  Fleet units can occupy and move through sea territories as well as coastal land provinces that have accessible coastlines.  Movement between territories happens simultaneously for all players after each negotiation period, making planning and coordination essential.  Territories are considered adjacent if they directly touch each other along borders or, for coastal provinces and sea areas, if they share an accessible coastline.  Units can move into empty adjacent provinces or challenge opposing units for control by having numerical superiority through support orders from adjacent friendly units.

To play Diplomacy effectively, each player simultaneously writes down secret orders for their armies and fleets each turn.  Possible orders include move, hold position, or provide support to another unit moving into or holding a territory.  After all orders are revealed simultaneously, conflicts are resolved by numerical superiority determined by counting supporting units.  Capturing new supply centers increases your unit count, allowing you to build more units during designated build phases.  Players negotiate extensively to create alliances, promise support, and strategically coordinate attacks or defenses, but ultimately, trust can be fleeting as betrayals and shifting alliances are central to gameplay.

'''
    
    @staticmethod
    def get_format_scenario_template_goals(phase, countries, game_id):
        c1 = countries[0]
        c2 = countries[1]
        prompt = Template.center_instruction + str(phase['state']['centers']) + Template.unit_instruction + str(phase['state']['units'])
        agent_goals_list = []
        agent_goals_list.append(f"Negotiate with {c2} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Try to make your dialogue between {c2} within 10 turns.")
        agent_goals_list.append(f"Negotiate with {c1} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Try to make your dialogue between {c1} within 10 turns.")
        return prompt, agent_goals_list

    @staticmethod
    def get_previous_scenario_fewshot(phase, countries, game_id, game_dir):
        from profile_utils import get_previous_dialogue_unit
        c1 = countries[0]
        c2 = countries[1]
        prompt = f"{Template.previous_dialogue_instruction}: {c1} and {c2}. \n"
        prompt += get_previous_dialogue_unit(game_dir, game_id, phase, countries) + "\n"
        prompt += Template.center_instruction + str(phase['state']['centers']) + '\n' + Template.unit_instruction + str(phase['state']['units'])
        agent_goals_list = []
        agent_goals_list.append(f"Negotiate with {c2} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Imitate the style and content of previous dialogues between the two countries, conducting a multi-round conversation. Make sure your own dialogue between {c2} is within 5 turns.")
        agent_goals_list.append(f"Negotiate with {c1} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Imitate the style and content of previous dialogues between the two countries, conducting a multi-round conversation. Make sure your own dialogue between {c1} is within 5 turns.")
        # import pdb
        # pdb.set_trace()
        return prompt, agent_goals_list

    @staticmethod
    def get_previous_scenario_fewshot_plausible(phase, countries, game_id, game_dir, c1_plausible_move, c2_plausible_move):
        # TODO: Add plausible into this method
        from profile_utils import get_previous_dialogue_unit

        c1 = countries[0]
        c2 = countries[1]
        prompt = f"{Template.previous_dialogue_instruction}: {c1} and {c2}: \n"
        prompt += get_previous_dialogue_unit(game_dir, game_id, phase, countries) + "\n"
        prompt += Template.center_instruction + str(phase['state']['centers']) + '\n' + Template.unit_instruction + str(phase['state']['units'])
  
        agent_goals_list = []
        agent_goals_list.append(f"Negotiate with {c2} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Here are all your plausible movements for this turn: {c1_plausible_move}. Imitate the style and content of previous dialogues between the two countries, conducting a multi-round conversation. Make sure your own dialogue between {c2} is within 5 turns.")
        agent_goals_list.append(f"Negotiate with {c1} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Here are all your plausible movements for this turn: {c2_plausible_move}. Imitate the style and content of previous dialogues between the two countries, conducting a multi-round conversation. Make sure your own dialogue between {c1} is within 5 turns.")
        # import pdb
        # pdb.set_trace()
        return prompt, agent_goals_list
    
    @staticmethod
    def get_previous_scenario_fewshot_actual_moves(phase, countries, game_id, game_dir, history_length):
        from profile_utils import get_actual_moves, get_previous_dialogue_unit
        # TODO: pdb to check whether the prompt is correct
        c1, c2 = countries[:2]
        prompt = f"{Template.previous_dialogue_instruction}: {c1} and {c2}. \n"
        if get_previous_dialogue_unit(game_dir, game_id, phase, countries, history_length) is None:
            return None, None
        prompt += get_previous_dialogue_unit(game_dir, game_id, phase, countries, history_length) + "\n"
        prompt += Template.center_instruction + str(phase['state']['centers']) + '\n' + Template.unit_instruction + str(phase['state']['units'])
        c1_actual_move = get_actual_moves(phase, c1)
        c2_actual_move = get_actual_moves(phase, c2)

        agent_goals_list = []
        agent_goals_list.append(f"Negotiate with {c2} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Here are all the movements you plan to do for this turn: {c1_actual_move}. {Template.finetune_instruction}")
        # Imitate the style and content of previous dialogues between the two countries, conducting a multi-round conversation.
        agent_goals_list.append(f"Negotiate with {c1} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Here are all the movements you plan to do for this turn: {c2_actual_move}. {Template.finetune_instruction}")
        # Imitate the style and content of previous dialogues between the two countries, conducting a multi-round conversation.
        # import pdb
        # pdb.set_trace()
        return prompt, agent_goals_list
    
    @staticmethod
    def get_previous_scenario_fewshot(phase, countries, game_id, game_dir):
        from profile_utils import get_previous_phase_finetune_format
        c1 = countries[0]
        c2 = countries[1]
        prompt = f"{Template.previous_dialogue_instruction}: {c1} and {c2}: \n"
        prompt += get_previous_phase_finetune_format(game_dir, game_id, phase, countries) + "\n"
        prompt += Template.center_instruction + str(phase['state']['centers']) + '\n' + Template.unit_instruction + str(phase['state']['units'])
        agent_goals_list = []
        agent_goals_list.append(f"Negotiate with {c2} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. {Template.finetune_instruction}")
        agent_goals_list.append(f"Negotiate with {c1} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. {Template.finetune_instruction}")
        return prompt, agent_goals_list
    
    @staticmethod
    def get_finetune_scenario(phase, countries, game_id, game_dir):
        from profile_utils import get_full_finetune_format
        c1 = countries[0]
        c2 = countries[1]
        # prompt = f"{Template.previous_dialogue_instruction}: {c1} and {c2}: \n"
        prompt = Template.background_instruction
        finetune_prompt, c1_planned_order, c2_planned_order = get_full_finetune_format(game_dir, game_id, phase, countries)
        prompt += finetune_prompt
        # prompt += Template.center_instruction + str(phase['state']['centers']) + '\n' + Template.unit_instruction + str(phase['state']['units'])
        agent_goals_list = []
        # agent_goals_list.append(f"You are play as {c1}, and you will negotiate with {c2} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific planned movements that can be made this turn for your benefit, here is your planned movements: {c1_planned_order}. Focus not only on your long-term goals but also on the specific actions you want to achieve in this current turn. Be clear about your immediate tactical objectives while maintaining your strategic position. {Template.finetune_instruction}")
        # agent_goals_list.append(f"You are play as {c2}, and you will negotiate with {c1} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific planned movements that can be made this turn for your benefit, here is your planned movements: {c2_planned_order}. Focus not only on your long-term goals but also on the specific actions you want to achieve in this current turn. Be clear about your immediate tactical objectives while maintaining your strategic position. {Template.finetune_instruction}")
        agent_goals_list.append(f"You are play as {c1}. {Template.finetune_instruction}")
        agent_goals_list.append(f"You are play as {c2}. {Template.finetune_instruction}")
        return prompt, agent_goals_list