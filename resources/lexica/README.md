# Lexica #

Extract lexicon features from text. Available lexica are:
- [LIWC 2007](http://web.archive.org/web/20170717133246/liwc.wpengine.com/compare-dictionaries/)
- [LIWC 2015](http://web.archive.org/web/20170717133246/liwc.wpengine.com/compare-dictionaries/)
- [NRC EmoLex v0.92](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
- [NRC Hashtag sentiment v1.0](http://sentiment.nrc.ca/lexicons-for-research/)
- [NRC Valence Arousal Dominance lexica](https://saifmohammad.com/WebPages/nrc-vad.html) (VAD)
- NRC Optimism Pessimism Lexicon
- [Agency and Authority Connotation frame](https://homes.cs.washington.edu/~msap/movie-bias/)
- [Gender Decoder](http://gender-decoder.katmatfield.com/)
- [Age Gender Lexica (Sap et al., 2016)](http://wwbp.org/lexica.html)
- [Empath (Fast et al., 2016)](https://github.com/Ejhfast/empath-client)
- [Concreteness Ratings (Brysbaert et al., 2014)](http://crr.ugent.be/papers/Brysbaert_Warriner_Kuperman_BRM_Concreteness_ratings.pdf)
- [Moral Foundations (Graham, Jesse et al., 2009)](https://pdfs.semanticscholar.org/9959/070decef014776aaccd2eb2b027b909f7978.pdf)
- [Enhanced Moral Foundations (Rezapour, Shah & Diesner, 2019)](https://www.aclweb.org/anthology/W19-1305/)
- [Mind Perception Dictionary (Schweitzer, S., & Waytz, A., 2023)](https://www.adamwaytz.com/measures-and-materials/the-mind-perception-dictionary)
- [Agency & Communion Dictionary (Pietraszkiewicz et al., 2019)](https://doi.org/10.1002/ejsp.2561)
- [Vulnerability Dictionary (Hu et al. 2023)](https://link.springer.com/chapter/10.1007/978-3-031-43129-6_11)

## Running the code ##
Import it into python using `from util import *`

Select the dictionary you want to use:
- `lex = liwc.parse_liwc("2007")` for LIWC 2007
- `lex = liwc.parse_liwc("2015")` for the 2015 version
- `lex = nrc.parse_emolex()` for the NRC EmoLex
- `lex = nrc.parse_hashtagsent()` for the NRC Hashtag sentiment lexicon
- `lex = nrc.parse_valence_aroursal_dominance()` for the NRC VAD
- `lex = nrc.parse_optpess()` for the NRC Optimism/Pessimism lexicon (weighted)
- `lex = conno.parse_connotation("agency")` for the Agency connotation frame
- `lex = conno.parse_connotation("authority")` for the Authority connotation frame
- `lex = genderJobs.parse_genderdecoder()` for the gender decoder lexicon
- `lex = ageGender.parse_ageGender()` for the Age Gender Lexica
- `lex = concr.parse()` for the Concreteness Ratings
- `lex = moralF.parseMF()` for the MoralFoundations
- `lex = moralF.parseEnhancedMF()` for the Enhanced Moral Foundations
- `lex = mindPercept.parse_mind()` for the mind perception dictionary
- `lex = agencyComm.load_agency_communion()` for the agency and communion dictionary
- `lex = vuln.parse_vulnerability()` for the vulnerability dictionary

Optionally only select certain categories:
`lex = liwc.parse_liwc("2007",whitelist=["posemo","negemo"])`

Extract features using the `extract` function:
- `extract(lex,"this is a text")`:
    will return a dictionary of {category: percentage}
- `extract(lex,"this is a text",percentage=False)`:
    will return a dictionary of {category: raw word count}
    
If lex is a weighted lexicon, each matched word is multiplied by it's category weight

Example:

```python
In [1]: from util import *

In [2]: lex = nrc.parse_emolex()

In [3]: extract(lex,"This is a story about a girl named lucky",False)
Out[3]: {'anticipation': 1, 'joy': 2, 'positive': 2, 'surprise': 2}

In [4]: lex = liwc.parse_liwc("2015")

In [5]: extract(lex,"This is a story")
Out[5]:
{'article': 0.25,
 'auxverb': 0.25,
 'focuspresent': 0.25,
 'function': 0.75,
 'ipron': 0.25,
 'pronoun': 0.25,
 'social': 0.25,
 'verb': 0.25}
       
```
## Connotation frames ##
You can also work at a verb-level, instead of the word-level.
Specifically, the connotation frames of *agency* and *authority* only work with verbs.

Use `extractVerbs` to only count verbs towards connotation frames.
Verbs are detected using the SpaCy POS tagger and lemmatizer.

*Note that the results will be different since some verbs are nouns/adjectives sometimes.*

Example:
```python
In [1]: from util import *

In [2]: lex = conno.parse_connotation()

In [3]: extractVerbs(lex,"They grabbed and pulled the cool machine")
Out[3]: {'agency_pos': 1.0}

In [4]: extract(lex,"They grabbed and pulled the cool machine")
Out[4]: {'agency_neg': 0.14285714285714285}

```

## Age Gender Lexica ##
Because of the way the lexica were created (see paper), you need to use the specific function `predictAgeGender` which uses better tokenizing, and adds a lexicon intercept after extraction.

Example:
```python
In [1]: from util import *

In [2]: lex = conno.parse_ageGender()

In [3]: predictAgeGender("Because, I don't like being social :/ :) abnormally likeable really.")
Out[3]: {'age': 14.574716515146918, 'gender': 9.565091876339233, 'gender_bin': 'F'}
```

Sap, M., Park, G., Eichstaedt, J. C., Kern, M. L., Stillwell, D. J., Kosinski, M., Ungar, L. H., & Schwartz, H. A. (2014). *Developing Age and Gender Predictive Lexica over Social Media*. EMNLP

## Empath ##
Note: requires `pip install empath`

Example using the regular lexicon extraction code:
```
In [1]: from util import *

In [2]: lex = empath.loadEmpath()

In [3]: extract(lex,"They grabbed and pulled the cool machine")
Out[3]:
{'cold': 0.14285714285714285, 'tool': 0.14285714285714285, 'weather': 0.14285714285714285}
```

Example using the native Empath extraction code:
```
In [1]: from util import *

In [2]: empath = empath.loadEmpathObject()

In [3]: empath.analyze("They grabbed and pulled the cool machine",normalize=True)
Out[3]:
{'achievement': 0.0, 'affection': 0.0, ...}
```
  
