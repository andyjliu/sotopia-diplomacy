#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint
import string
from random import shuffle


FILE_MF="moralFoundations.dic"
FILE_EMF="enhancedMoralFoundations.txt" 
CAT_DELIM = "%"

def parseMF():
  f = open(os.path.join(os.path.dirname(__file__),FILE_MF))
  cats_section = False
  words_to_cats = {}
  id_to_cat = {}
  weird_lines = {}
  for l in f:
    l = l.strip()
    if not l:
      continue
    if l == CAT_DELIM:
      cats_section = not cats_section
      continue

    if cats_section:
      try:
        i, cat = l.split()
        id_to_cat[int(i)] = cat
      except: pass # likely hierarchical category tags
    else:
      w, cat_str = l.split("\t",maxsplit=1)
      w = w.strip()
      cats = cat_str.strip().split()

      if "(" in w and ")" in w:
        w = w.replace("(","").replace(")","")

      try:
        cats_str = [id_to_cat[int(i)] for i in cats]
        # adding meta categories
        cats_str.extend(set([c.replace("rtue","Virtue").replace("ce","Vice")
                             for cs in cats_str for c in cs.split("Vi")]))
        
        words_to_cats[w] = cats_str
      except:
        weird_lines[w] = cats

  return words_to_cats

def parseEnhancedMF():
  df = pd.read_csv(os.path.join(os.path.dirname(__file__),FILE_EMF),sep ="|",
                    names = ['token', 'pos','syn_label','lemma', 'm_desc', 'm_type'])
  df['token']= df['token'].str.split(" = ", n = 1, expand = True)[1]
  df['pos']= df['pos'].str.split(" = ", n = 1, expand = True)[1]
  df['syn_label'] = df['syn_label'].str.split(" = ", n = 1, expand = True)[1]
  df['lemma']= df['lemma'].str.split(" = ", n = 1, expand = True)[1]
  df['m_desc']= df['m_desc'].str.split(" = ", n = 1, expand = True)[1]
  df['m_type']= df['m_type'].str.split(" = ", n = 1, expand = True)[1]

  df["m_dim"] = df["m_desc"].str.replace("Virtue","").str.replace("Vice","")
  
  df["ViceOrVirtue"] = "Vice"
  df.loc[df["m_desc"].str.contains("Virtue"),"ViceOrVirtue"] = "Virtue"

  words_to_cats = df.set_index("token")[["m_desc","m_dim","ViceOrVirtue"]].apply(
    lambda x: x.tolist(),axis=1).to_dict()
  
  return words_to_cats
  
if __name__=="__main__":
  # l = parseMF()
  l = parseEnhancedMF()
