#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint
import string
from random import shuffle

CONC_FILE = "ConcretenessRatings.csv"

def parse():
  
  df = pd.read_csv(os.path.join(os.path.dirname(__file__),CONC_FILE),converters={'Word': str})
  df["Category"] = "concreteness"

  # Changing scale from (1-5] to (0-1]
  df["Conc.M"] = (df["Conc.M"]-1) / 4
  
  words_to_cat = {str(w): {c: float(v)} for w,c,v in df[["Word","Category","Conc.M"]].itertuples(index=False)}
  
  return words_to_cat

if __name__ == "__main__":
  l = parse()
  embed()
