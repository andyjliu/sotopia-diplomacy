#!/usr/bin/env python3

import sys, os, re
from IPython import embed
import pandas as pd
from pprint import pprint
import string
from random import shuffle
import argparse
from collections import Counter

from LIWC import LIWCutil as liwc
from NRC import NRCutil as nrc
from CONNOTATION import CFutil as conno
from GENDERJOBS import genderUtil as genderJobs
from AGEGENDER import ageGenderUtil as ageGender
from EMPATH import empathUtil as empath
from CONCRETENESS import concrUtil as concr
from MORALF import mfUtil as moralF
from MIND import MINDUtil as mindPercept
from AGENCYCOMMUNION import util as agencyComm
from VULNERABILITY import VULNutil as vuln

def preprocess(doc):
  """Document is a string.
  Tokenizes, removes trailing punctuation from words, counts how many words"""
  better = doc.lower().replace("kind of", "kindof")
  def strip_punct(x):
    if all([c in string.punctuation for c in x]):
      return x
    else:
      return x.strip(string.punctuation)
  # wb = re.compile(r'\b\S+?\b')
  toks = [strip_punct(w) for w in better.split()]
  l = len(toks)# len(wb.findall(better))
  # return better, l
  return toks, l


def _extract(lex,toks,n_words,percentage=True,wildcard="*",store_matches=False):
  extracted = {}
  is_weighted = isinstance(list(lex.items())[0][1],dict)
  
  if store_matches:
    matches = {}
    
  if wildcard == "":
    wildcard = "~$¬" # highly unlikely combo
  
  for w, cats in lex.items():
    w_split = w.split()
    # split -> bigram expression
    if not any([t.replace(wildcard,"") in " ".join(toks) for t in w_split]):
      continue

    if wildcard in w:
      ngrams = [[t.startswith(w_t.replace(wildcard,"")) for t,w_t in zip(tp,w_split)]
                for tp in zip(*[
                    toks[i:] for i in range(len(w_split))])]
      # matches = [a[0] for a in ngrams if a[1]]
      #ngramsMatches = [[a[1] for a in aa] for aa in ngrams]
      count = sum(map(all,ngrams))
      
    else:
      ngrams = [list(t) for t in zip(*[
        toks[i:] for i in range(len(w_split))])]
      count = ngrams.count(w_split)
      
    if count > 0 and store_matches:
      matches[w] = {"count": count}
      
    if count:
      for c in cats:
        if is_weighted:
          wg = cats[c]
        else:
          wg = 1
        extracted[c] = extracted.get(c,0) + (count*wg)
        
  
  
  if percentage:
    ## Turn into percentages
    extracted = {k: v/n_words for k,v in extracted.items()}

  if store_matches:

    for m, d in matches.items():
      d["cats"] = lex[m]

    # matches2cats = m, lex[m]) for m in matches])
    return extracted, matches
  return extracted

def extractFast(lex,doc,percentage=True):
  # For backwards compatibility
  return extract(lex,doc,percentage)

def extract(lex,doc,percentage=True,wildcard="*",returnNWords=False,store_matches=False):
  """
  Counts all categories present in the document given the lexicon dictionary.
  percentage (optional) indicates whether to return raw counts or
  normalize by total number of words in the document
  """
  toks, n_words = preprocess(doc)
  lexOutput = _extract(lex,toks,n_words,percentage,wildcard=wildcard,store_matches=store_matches)
  
  if returnNWords:
    return lexOutput, n_words
  
  return lexOutput

def loadAgeGender():
  global ageGenderLex
  ageGenderLex = ageGender.parse_ageGender()

def predictAgeGender(doc):
  try:
    d = ageGenderLex
  except:
    loadAgeGender()
    d = ageGenderLex

  toks, n_words = ageGender.preprocess_twt(doc)
  res = _extract(d,toks,n_words)

  for cat,val in d["_intercept"].items():
    res[cat] = res[cat]+val
  # Binarize gender
  res["gender_bin"] = "F" if res["gender"] > 0 else "M"
  return res

def extractVerbsDocs(lex,docs,pct=True):
  print("Loading spacy")
  nlp = conno.spacy.load("en")
  # return [extractVerbs(lex,d,pct=pct,nlp=nlp) for d in docs]
  return [extractVerbs(lex,d,pct=pct) for d in docs]
  
def extractVerbs(lex,doc,pct=True,returnNWords=False):# ,nlp=conno.NLP):
  verbs = conno.findVerbs(doc)
  n_verbs = len(verbs)
  lex_output = _extract(lex,verbs,n_verbs,pct)
  if returnNWords:
    return lex_output, n_verbs
  return lex_output

def reverse_dict(d):
  cats_to_words = {}
  for w, cs in d.items():
    for c in cs:
      if isinstance(cs,dict):
        weight = cs[c]
        l = cats_to_words.get(c,dict())
        l[w] = weight
      else:
        l = cats_to_words.get(c,set())
        l.add(w)
      cats_to_words[c] = l
  return cats_to_words

# def reverse_dict(d):
#   cats_to_words = {}
#   for w, cs in d.items():
#     for c in cs:
#       l = cats_to_words.get(c,set())
#       l.add(w)
#       cats_to_words[c] = l
#   return cats_to_words

def sample_cat(rev_d, cat,n=10):
  l = list(rev_d[cat])
  shuffle(l)
  return l[:n]

def main(args):
  w = args.lexicon
  if "liwc" in w.lower():
    d = liwc.parse_liwc(w[-4:])
  elif "nrc_vad" in w.lower():
    d = nrc.parse_valence_aroursal_dominance()
  elif "nrc_affint" in w.lower():
    d = nrc.parse_affect_intensity()
  elif "nrc_emolex" in w.lower():
    d = nrc.parse_emolex()
  elif "nrc_opt" in w.lower():
    d = nrc.parse_opt()
  elif "nrc_sent" in w.lower():
    d = nrc.parse_hashtagsent()
  elif "agencycommunion" in w.lower():
    d = agencyComm.load_agency_communion()
  elif "agency" in w.lower() or "authority" in w.lower():
    d = conno.parse_connotation()
  elif "genderjobs" in w.lower() or "genderdecoder" in w.lower():
    d = genderJobs.parse_genderdecoder()
  elif "agegender" in w.lower():
    d = ageGender.parse_ageGender()
  elif "empath" in w.lower():
    d = empath.loadEmpath()
  elif "concrete" in w.lower():
    d = concr.parse()
  elif "enhancedmoral" in w.lower():
    d = moralF.parseEnhancedMF()
  elif "moralF" in w.lower():
    d = moralF.parseMF()
  elif "mind" in w.lower():
    d = mindPercept.parse_mind()
  elif "vuln" in w.lower():
    d = vuln.parse_vulnerability()
    
  else:
    print("Unrecognized lexicon name. Available: "
          "LIWC2007 LIWC2015 NRC_emolex NRC_opt agency authority genderJobs ageGender moralF enhancedmoral mind")
    sys.exit(2)
    
  test = "We should preserve. This is abnormally cool cause we rock."
  # test += "Because, I don't like being social :/ :) abnormally likeable really. "
  # test += "Give a man a fish and you feed him for a day; teach a man to fish and you feed him for a lifetime. We're looking for exceptional talent"
  if not args.text:
    args.text = test

  pprint(preprocess(args.text))
  from time import time
  start = time()
  if "agegender" in w.lower():
    d1 = predictAgeGender(args.text)
  else:
    d1 = extractFast(d,args.text)
  pprint(d1)
  print("Time:",time()-start)
  
  embed()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-l","--lexicon",default="liwc_2015")
  parser.add_argument("-t","--text")
  
  args = parser.parse_args()
  
  main(args)

# Deprecated AF
def extract_slow(lex, doc, percentage=True):
  """
  Counts all categories present in the document given the lexicon dictionary.
  percentage (optional) indicates whether to return raw counts or
  normalize by total number of words in the document"""
  doc, n_words = preprocess(doc)
  doc = " ".join(doc)
  extracted = {}

  is_weighted = isinstance(list(lex.items())[0][1],dict)
  if is_weighted:
    raise NotImplementedError("For weighted lexica, use extractFast")
  
  for w, cats in lex.items():
    if all([c in string.punctuation for c in w]):
      w_re = re.escape(w)
    else:
      w_re = r"\b"+w
      if "*" in w:
        w_re = w_re.replace("*",r"\w*\b")
      if w_re[-2:] != r"\b": w_re += r"\b"
        
        # else: w_re += r"\b"
    r = re.compile(w_re)
    matches = r.findall(doc)
    if matches:
      for c in cats:
        extracted[c] = extracted.get(c,0)+len(matches)

  if percentage:
    ## Turn into percentages
    extracted = {k: v/n_words for k,v in extracted.items()}
  return extracted
