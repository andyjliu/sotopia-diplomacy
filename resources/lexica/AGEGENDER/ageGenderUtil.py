#!/usr/bin/env python3

import os
import pandas as pd
from IPython import embed
from nltk.tokenize import TweetTokenizer

fs = {
  "age":"emnlp14age.csv",
  "gender": "emnlp14gender.csv"
}
twtTokzr = TweetTokenizer()

def parse_ageGender():
  data = []
  for cat, lex in fs.items():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),lex))
    
    df["term"] = df["term"].astype(str)
    df = df.set_index("term")
    df = df.rename(columns={"weight": cat})
    data.append(df)
  df = pd.concat(data,axis=1).fillna(0.0)
  
  return df.T.to_dict()

def preprocess_twt(doc):
  toks = twtTokzr.tokenize(doc)
  return toks, len(toks)

if __name__=="__main__":
  l = parse_ageGender()
