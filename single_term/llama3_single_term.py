import json
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import pandas as pd
from tqdm import tqdm
from llm import call
# from utils import *
from config import get_model_config

model = 'llama3_8b'
def llm_config_func(llm):
    llm.temperature = 0
    llm.max_tokens = 4096
    return llm
config = get_model_config(model)


# Create the agents with their information
agent_1 = {
    "first_name": "Bob",
    "last_name": "John",
    "country": "ENGLAND",
    "goal": "You are playing diplomacy games, and your goal is to win this game."
}

agent_2 = {
    "first_name": "Alice",
    "last_name": "Smith",
    "country": "FRANCE",
    "goal": "You are playing diplomacy games, and your goal is to win this game."
}

# Define the scenario and social goals for the agents
scenario = '''
This is for your own strategic planning and won't be shared. Examples of things you might consider include: your relationships with other powers, what significant changes have happened recently, predictions about the other powers' orders and alliances, how much defence/offence/support/peace you plan to make, and how you might improve any of that. Do not romanticize things, be realistic.

This is the information of the countries' centers, and you will playing as the given country as you act:
'centers': {
    'AUSTRIA': ['VIE', 'TRI', 'BUD'],
    'ENGLAND': ['EDI', 'LON', 'LVP'],
    'FRANCE': ['BRE', 'PAR', 'MAR'],
    'GERMANY': ['KIE', 'MUN', 'BER'],
    'ITALY': ['NAP', 'ROM', 'VEN'],
    'RUSSIA': ['STP', 'MOS', 'WAR', 'SEV'],
    'TURKEY': ['ANK', 'SMY', 'CON']
}

This is the information of the countries' units, and you will playing as the given country as you act:
'units': {
    'AUSTRIA': ['A VIE', 'F TRI', 'A BUD'],
    'ENGLAND': ['F EDI', 'F LON', 'A LVP'],
    'FRANCE': ['F BRE', 'A PAR', 'A MAR'],
    'GERMANY': ['F KIE', 'A MUN', 'A BER'],
    'ITALY': ['F NAP', 'A ROM', 'A VEN'],
    'RUSSIA': ['F STP/SC', 'A MOS', 'A WAR', 'F SEV'],
    'TURKEY': ['F ANK', 'A SMY', 'A CON']
}
'''
example_order_with_explaination = """\n\nHere are examples showing the format for orders and with its explaination:
A LON H: Hold in London
A WAL - LVP: Move to Liverpool
F SPA/NC - MAO: Move from Spain North Coast to Mid-Atlantic Ocean
A WAL S F LON: Support Fleet in London
A WAL S F IRI - LVP: Support Fleet moving from Irish Sea to Liverpool
F NTH C A YOR - NWY: Convey Army from Yorkshire to Norway
A YOR - NWY VIA: Move to Norway via convoy
F IRI R MAO: Retreat to Mid-Atlantic Ocean
F IRI D: Disband in Irish Sea
A LON B: No standard order associated with 'B', likely an error or unknown command
"""

example_order = """\n\nHere are examples showing the format for orders:
A LON H
A WAL - LVP
F SPA/NC - MAO
A WAL S F LON
A WAL S F IRI - LVP
F NTH C A YOR - NWY
A YOR - NWY VIA
F IRI R MAO
F IRI D
A LON B
"""

social_goal_1 = f"List of strings of orders you plan to make at the end of the turn to your units in the same abbreviated format as the history. You will converse with the other powers for one rounds, then your final set of orders will be executed. Since this isn't the final message round of the phase, you aren't locked into these orders. However, you will still need to give an order list in the final term that follows this example: {example_order}"

social_goal_2 = f"List of strings of orders you plan to make at the end of the turn to your units in the same abbreviated format as the history. You will converse with the other powers for one rounds, then your final set of orders will be executed. Since this isn't the final message round of the phase, you aren't locked into these orders. However, you will still need to give an order list in the final term that follows this example: {example_order}"

# Definition of order


# Define the configuration for the language model
model_version = "llama3-8b"
# config = {
#     "temperature": 0.0,
#     "max_tokens": 100,
#     "top_p": 1,
#     "frequency_penalty": 0,
#     "presence_penalty": 0
# }

# Agents engage in a single round of conversation
agent_1_prompt = [
    "You are a helpful assistant",
    f"{scenario}\n\n{social_goal_1}\n\nAgent 1 information: {json.dumps(agent_1)}"
]
agent_1_response = call(
    agent_1_prompt,
    llm_config_func=llm_config_func,
    has_system_prompt=True,
    model_version=model,
    verbose=True,
    **config
)

agent_2_prompt = [
    "You are a helpful assistant",
    f"{scenario}\n\n{social_goal_2}\n\nAgent 2 information: {json.dumps(agent_2)}\n\nAgent 1: {agent_1_response}"
]
agent_2_response = call(
    agent_2_prompt,
    llm_config_func=llm_config_func,
    has_system_prompt=True,
    model_version=model,
    verbose=True,
    **config
)


import json

# ... (previous code remains the same)

# Save agent_1_response and agent_2_response to a file
with open("agent_responses.json", "w") as file:
    data = {
        "agent_1_response": agent_1_response,
        "agent_2_response": agent_2_response
    }
    json.dump(data, file, indent=4)
