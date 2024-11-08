#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint

from util import extract
from LIWC import LIWCutil as liwc
from NRC import NRCutil as nrc
from CONNOTATION import CFutil as conno
from GENDERJOBS import genderUtil as genderJobs

import argparse

def choose_lex(w):
  if "liwc" in w.lower():
    d = liwc.parse_liwc(w[-4:])
  elif "nrc_emolex" in w.lower():
    d = nrc.parse_emolex()
  elif "nrc_opt" in w.lower():
    d = nrc.parse_opt()
  elif "agency" in w.lower() or "authority" in w.lower():
    d = conno.parse_connotation()
  elif "genderjobs" in w.lower() or "genderdecoder" in w.lower():
    d = genderJobs.parse_genderdecoder()
  else:
    print("Unrecognized lexicon name. Available: "
          "LIWC2007 LIWC2015 NRC_emolex NRC_opt agency authority genderJobs")


def main(args):
  lex = choose_lex(args.lexicon)
  
  if args.inputfile:
    df = pd.read_csv(args.inputfile,header=None)
    # TODO: don't hardcode!
    docs = df.iloc[:,0].tolist()
  else:
    # Test if that actually works?
    docs = sys.stdin.readlines()

  pprint(docs)
  results = [extract(lex,d) for d in docs]
  pprint(results)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Extract lexica from text')
  parser.add_argument("-l","--lexicon")
  parser.add_argument("-i","--inputfile")
  parser.add_argument("-o","--outputfile")
  args = parser.parse_args()
  if not args.lexicon:
    parser.print_help()
    exit()
  else:
    main(args)
