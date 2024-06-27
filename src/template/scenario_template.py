class Template:
    center_instruction = '''This is the information of the countries' centers, and you will playing as the given country as you act: \n centers: '''

    unit_instruction = '''This is the information of the countries' centers, and you will playing as the given country as you act: \n units: '''


    @staticmethod
    def get_format_scenario_template_goals(phase, countries, game_id):
        prompt = Template.center_instruction + str(phase['state']['centers']) + Template.unit_instruction + str(phase['state']['units'])
        agent_goals_list = []
        for country in countries:
            agent_goals_list.append(f"Negotiate with {country} so that they will play moves that are beneficial to your board position, either this turn or in future turns. Discuss specific army movements that can be made this turn for your benefit if any exist.")
        return prompt, agent_goals_list