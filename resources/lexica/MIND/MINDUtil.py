#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint
import string
from random import shuffle


MIND_FILE="mind.dic"
CAT_DELIM = "%"

def parse_mind(whitelist=None):
  f = open(os.path.join(os.path.dirname(__file__),MIND_FILE))
  cats_section = False
  words_to_cats = {}
  id_to_cat = {}
  weird_lines = {}
  for l in f:
    l = l.strip()
    if l == CAT_DELIM:
      cats_section = not cats_section
      continue

    if cats_section:
      i, cat = l.split(" ")
      id_to_cat[int(i)] = cat
    else:
      w, cats = l.split(" ")[0], l.split(" ")[1:]
      w = w.lower()
      try:
        words_to_cats[w] = [id_to_cat[int(i)] for i in cats]
      except:
        weird_lines[w] = cats
          
  words_to_cats_d = {w: {c:1 for c in cs} for w,cs in words_to_cats.items()}

  print(weird_lines)
  ## If whitelist
  if whitelist:
    words_to_cats = {w: [c for c in cs if c in whitelist] for w,cs in words_to_cats.items()}
    words_to_cats = {w:cs for w,cs in words_to_cats.items() if cs}

  return words_to_cats



if __name__=="__main__":
  l = parse_mind()
  
  print(l)
