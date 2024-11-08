#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint
import string
from random import shuffle


GENDERFILE = "genderDecoder.csv"

def parse_genderdecoder():
  df = pd.read_csv(os.path.join(os.path.dirname(__file__),GENDERFILE))
  words_to_cat = {w:[c] for w,c in zip(df["word"],df["category"])}
  return words_to_cat

if __name__ == "__main__":
  l = parse_genderdecoder()
  embed()
