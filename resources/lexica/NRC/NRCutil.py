#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint
import string
from random import shuffle

FILES = {
  "AffInt": "NRC-AffectIntensity-Lexicon.txt",
  "VAD": "NRC-VAD-Lexicon.txt",
  "EmoLex": "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt",
  "OptPess": ["Optimism-Pessimism-Lexicons/unigrams-pmilexicon.txt",
              "Optimism-Pessimism-Lexicons/bigrams-pmilexicon.txt"],
  "HashtagSent": ["NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-unigrams.txt",
                  "NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-bigrams.txt"],
  "GenderedWords": "gendered_words.txt"
}
HEADER = "......................................................................."

def parse_valence_aroursal_dominance(remove_gendered_words=False,only_top_n=0):
  fn = os.path.join(os.path.dirname(__file__),FILES["VAD"])
  df = pd.read_csv(fn,sep="\t")
  df.columns = [c.lower() for c in df.columns]
  df["word"] = df["word"].astype(str)
  df = df.set_index("word")
  if remove_gendered_words:
    fn = os.path.join(os.path.dirname(__file__),FILES["GenderedWords"])
    genderedWords = set(open(fn).read().strip().split())
    df = df.loc[df.index.difference(genderedWords)]
    
  if only_top_n:
    if only_top_n < 1: # percentage
      only_top_n = len(df) * only_top_n
    
    ss = {c: df[c].nlargest(int(only_top_n)) for c in df.columns}
    df = pd.DataFrame(ss)#.fillna(0)
    words_to_cats = {
      x: r[~r.isnull()].to_dict() for x,r in df.iterrows()
    }
    return words_to_cats
  
  return df.T.to_dict()


def parse_affect_intensity():
  fn = os.path.join(os.path.dirname(__file__),FILES["AffInt"])
  df = pd.read_csv(fn,sep="\t",skiprows=36)

  words_to_cats = {}
  # return a dict of {word: {cat: weight}}
  for (t,w,c) in df.itertuples(index=False):
    words_to_cats[t] = words_to_cats.get(t,{})
    words_to_cats[t][c] = w
  return words_to_cats


def parse_emolex():
  f = open(os.path.join(os.path.dirname(__file__),FILES["EmoLex"]))
  # File has a header in 2 parts, starts after second set of periods

  header = 0
  words_to_cats = {}
  weird_lines = {}
  for l in f:
    l = l.strip()
    if l == HEADER:
      header += 1
      continue

    if header < 2: continue
    elif not l: continue
    else:
      w, cat, weight = l.split("\t")
      if weight == "1":
        words_to_cats[w] = words_to_cats.get(w,[])
        words_to_cats[w].append(cat)      
  return words_to_cats

def parse_opt():
  fs = FILES["OptPess"]

  words_to_cats = {}
  weird_lines = {}
  skip_lines = ["the list below is for","the lsit below is for"]
  
  for fn in fs:
    f = open(os.path.join(os.path.dirname(__file__),fn))
    header = 0
    for l in f:
      l = l.strip()
      if l == HEADER:
        header += 1
        continue

      if header < 2: continue
      elif not l: continue
      elif any([s in l.lower() for s in skip_lines]): continue
      else:
        out = l.split("\t")
        if len(out) == 1 or not any([str.isdigit(i) for i in l]):
          weird_lines[l] = out
          continue
        else:
          w, weight, _, _ = out
          words_to_cats[w] = words_to_cats.get(w,{})

          words_to_cats[w]["opt"] = float(weight)
          
  return words_to_cats
    
def parse_hashtagsent():
  fs = FILES["HashtagSent"]
  words_to_cats = {}
  category = "sent"
  for fn in fs:
    f = open(os.path.join(os.path.dirname(__file__),fn))
    rows = [l.strip().split("\t") for l in f]
    # score column is index 1 (from readme)
    
    words_to_cats.update({
      row[0]: {category: float(row[1])}
      for row in rows
    })

  return words_to_cats
