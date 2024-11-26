import rich
import json
class Template:

    previous_dialogue_instruction = '''Here is the previous dialogue, and the previous countries' centers and units between the two countries '''

    center_instruction = '''This is the information of the countries' centers, and you will playing as the given country as you act: \n centers: '''

    unit_instruction = '''This is the information of the countries' units, and you will playing as the given country as you act: \n units: '''

    finetune_instruction = '''Engage in negotiations with other countries, emulating the dialogue style from the previous dialogues. \n'''
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
        prompt = f"{Template.previous_dialogue_instruction}: {c1} and {c2}: \n"
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
    def get_previous_scenario_fewshot_actual_moves(phase, countries, game_id, game_dir):
        from profile_utils import get_actual_moves, get_previous_dialogue_unit
        # TODO: pdb to check whether the prompt is correct
        c1, c2 = countries[:2]
        prompt = f"{Template.previous_dialogue_instruction}: {c1} and {c2}: \n"
        prompt += get_previous_dialogue_unit(game_dir, game_id, phase, countries) + "\n"
        prompt += Template.center_instruction + str(phase['state']['centers']) + '\n' + Template.unit_instruction + str(phase['state']['units'])
        c1_actual_move = get_actual_moves(phase, c1)
        c2_actual_move = get_actual_moves(phase, c2)
        import pdb
        pdb.set_trace()
        agent_goals_list = []
        agent_goals_list.append(f"Negotiate with {c2} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Here are all the movements you plan to do for this turn: {c1_actual_move}. Imitate the style and content of previous dialogues between the two countries, conducting a multi-round conversation. Make sure your own dialogue between {c2} is within 5 turns.")
        agent_goals_list.append(f"Negotiate with {c1} so that it will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist. Here are all the movements you plan to do for this turn: {c2_actual_move}. Imitate the style and content of previous dialogues between the two countries, conducting a multi-round conversation. Make sure your own dialogue between {c1} is within 5 turns.")
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