import os

files = {"AGENCY" : "agency.dic", "COMMUNION": "communion.dic"}

def load_agency_communion():
  cats_to_words = {}
  words_to_cat = {}
  for cat, fn in files.items():
    f = os.path.join(os.path.dirname(__file__),fn)
    words = [w for w in open(f).read().split("\n") if w]
    cats_to_words[cat] = set(words)
    words_to_cat.update({w:[cat] for w in words})
  assert len(cats_to_words["AGENCY"] & cats_to_words["COMMUNION"]) == 0
  print(words_to_cat)
  return words_to_cat
  
  

