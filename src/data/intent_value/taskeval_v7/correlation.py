import os
import sys
import json
sys.path.append("/home/wenkail/diplomacy/sotopia-diplomacy/resources/lexica")
from util import *
import pdb
import rich
from tqdm import tqdm
countries_list = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']

def get_score_with_env(countries, prediction):
    scores = 0
    for c in countries:
        scores += prediction[c][c.upper()][1]
    score_dict = {}
    score_dict['env'] = prediction['env_uuid']
    score_dict['scores'] = scores / len(countries)
    score_dict['intent_dialogue'] = prediction['intent_dialogue']
    return score_dict

def get_score_dict_from_dialogue(dialogue):
    score_list = []
    for a in dialogue:
        keys = []
        for vkeys in a.keys():
            if vkeys in countries_list:
                keys.append(vkeys)
        score_list.append(get_score_with_env(keys, a))
    return score_list

def extract_all_features(text):
    """
    Extract features from text using all available lexicons.
    
    Args:
        text: Input text to analyze
        normalize: Whether to normalize counts (default: True)
        
    Returns:
        Dictionary with lexicon names as keys and their respective feature dictionaries as values
    """
    results = {}
    
    # LIWC versions
    try:
        results['liwc_2007'] = extract(liwc.parse_liwc("2007"), text)
    except Exception as e:
        results['liwc_2007'] = f"Error: {str(e)}"
        
    try:
        results['liwc_2015'] = extract(liwc.parse_liwc("2015"), text)
    except Exception as e:
        results['liwc_2015'] = f"Error: {str(e)}"
    
    # NRC lexicons
    try:
        results['emolex'] = extract(nrc.parse_emolex(), text)
    except Exception as e:
        results['emolex'] = f"Error: {str(e)}"
        
    try:
        results['hashtag_sentiment'] = extract(nrc.parse_hashtagsent(), text)
    except Exception as e:
        results['hashtag_sentiment'] = f"Error: {str(e)}"
        
    try:
        results['vad'] = extract(nrc.parse_valence_aroursal_dominance(), text)
    except Exception as e:
        results['vad'] = f"Error: {str(e)}"
        
    try:
        results['optimism_pessimism'] = extract(nrc.parse_optpess(), text)
    except Exception as e:
        results['optimism_pessimism'] = f"Error: {str(e)}"
    
    # Connotation frames
    try:
        results['connotation_agency'] = extract(conno.parse_connotation("agency"), text)
    except Exception as e:
        results['connotation_agency'] = f"Error: {str(e)}"
        
    try:
        results['connotation_authority'] = extract(conno.parse_connotation("authority"), text)
    except Exception as e:
        results['connotation_authority'] = f"Error: {str(e)}"
    
    # Other lexicons
    try:
        results['gender_decoder'] = extract(genderJobs.parse_genderdecoder(), text)
    except Exception as e:
        results['gender_decoder'] = f"Error: {str(e)}"
        
    try:
        results['age_gender'] = extract(ageGender.parse_ageGender(), text)
    except Exception as e:
        results['age_gender'] = f"Error: {str(e)}"
        
    try:
        results['concreteness'] = extract(concr.parse(), text)
    except Exception as e:
        results['concreteness'] = f"Error: {str(e)}"
        
    try:
        results['moral_foundations'] = extract(moralF.parseMF(), text)
    except Exception as e:
        results['moral_foundations'] = f"Error: {str(e)}"
        
    try:
        results['enhanced_moral_foundations'] = extract(moralF.parseEnhancedMF(), text)
    except Exception as e:
        results['enhanced_moral_foundations'] = f"Error: {str(e)}"
        
    try:
        results['mind_perception'] = extract(mindPercept.parse_mind(), text)
    except Exception as e:
        results['mind_perception'] = f"Error: {str(e)}"
        
    try:
        results['agency_communion'] = extract(agencyComm.load_agency_communion(), text)
    except Exception as e:
        results['agency_communion'] = f"Error: {str(e)}"
        
    try:
        results['vulnerability'] = extract(vuln.parse_vulnerability(), text)
    except Exception as e:
        results['vulnerability'] = f"Error: {str(e)}"
    
    return results
    

with open("taskeval_llama_dialogue_coop_with_actual.json", 'r') as f:
    episodes = json.load(f)
    
epi_score_list = get_score_dict_from_dialogue(episodes)
for epi in tqdm(epi_score_list, desc="Extracting features"):
    text = epi['intent_dialogue']
    features = extract_all_features(text)
    epi['features'] = features

with open("taskeval_llama_dialogue_coop_with_actual_features.json", 'w') as f:
    json.dump(epi_score_list, f)   
    