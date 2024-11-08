#!/usr/bin/env python3

from IPython import embed
from empath import Empath

def reverse_dict(d):
  cats_to_words = {}
  for w, cs in d.items():
    for c in cs:
      l = cats_to_words.get(c,set())
      l.add(w)
      cats_to_words[c] = l
  return cats_to_words

def loadEmpathObject():
  d = Empath()
  return d
  
def loadEmpath():
  d = Empath()
  return reverse_dict(d.cats)
