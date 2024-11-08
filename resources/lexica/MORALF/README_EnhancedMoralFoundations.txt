========================================================================
README file for the Enhanced Morality Lexicon
Release: V1.1 (06/12/2019)

Contact: 
Rezvaneh Rezapour (rezapou2@illinois.edu)
Jana Diesner (jdiesner@illinois.edu)
========================================================================
License information: CC BY (https://creativecommons.org/licenses/by/4.0/)

Please cite this dataset, using the following citations:

Rezapour, R., & Diesner, J. (2019). Enhanced Morality Lexicon. University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-3805242_V1

Rezapour, R., Shah, S., & Diesner, J. (2019). Enhancing the measurement of social effects by capturing morality. Proceedings of the 10th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (WASSA). Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Minneapolis, MN.

========================================================================
This lexicon is the expanded/enhanced version of the Moral Foundation Dictionary created by Graham and colleagues (Graham et al., 2013).
Our Enhanced Morality Lexicon (EML) contains a list of 4,636 morality related words.

This lexicon was used in the following paper - please cite this paper if you use this resource in your work.

Rezapour, R., Shah, S., & Diesner, J. (2019). Enhancing the measurement of social effects by capturing morality. Proceedings of the 10th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (WASSA). Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Minneapolis, MN.

In addition, please consider citing the original MFD paper:

Graham, J., Haidt, J., Koleva, S., Motyl, M., Iyer, R., Wojcik, S. P., & Ditto, P. H. (2013). Moral foundations theory: The pragmatic validity of moral pluralism. In Advances in experimental social psychology (Vol. 47, pp. 55-130). 
========================================================================
About the lexicon entries:

In the (.txt) format of the lexicon, the attributes are seperated by "|". 
The attributes are explained below:

token: shows morality related words.

pos: Part of the Speech of the words.
    n = Noun
    v = Verb 
    adj = Adjective
    adv = Adverb

syn_label: shows if the word is an original* one from the MFD.
    O = original word from the MFD
    S = synonym of a MFD word extracted from WordNet
    A = antonym of a MFD word extracted from WordNet
    H = direct hypernym of a MFD word extracted from WordNet

*For more information about WordNet, see:
	https://wordnet.princeton.edu/
	Christiane Fellbaum (1998, ed.) WordNet: An Electronic Lexical Database. Cambridge, MA: MIT Press.


lemma: shows if the word is lemmatized or not.
    yes = word is lemmatized **
    no = word is not lemmatized

** in this lexicon, all verbs are lemmatized.


m_desc: shows the label of morality classes. The lexicon consists of 12 morality classes.
    Note: We renamed HarmVirtue and HarmVice of the MFD to CareVirtue and CareVice in our lexicon.

m_type: shows the assigned number for each morality class (12 morality classes). 
    1 = CareVirtue (care)
    2 = CareVice (harm)
    3 = FairnessVirtue (fairness)
    4 = FairnessVice (cheating)
    5 = IngroupVirtue (loyalty)
    6 = IngroupVice (betrayal)
    7 = AuthorityVirtue (authority)
    8 = AuthorityVice (subversion)
    9 = PurityVirtue (purity)
    10 = PurityVice (degradation)
    11 = GeneralVirtue
    12 = GeneralVice


========================================================================
If you want to change the .txt lexicon to .csv, you can use the following python code.

###Python code for reading and converting the lexicon to csv:
## requirements: pandas, python3

import pandas as pd
EML = pd.read_csv('path_to_read_lexicon.txt',sep ="|", names = ['token', 'pos','syn_label','lemma', 'm_desc', 'm_type'])
EML['token']= EML['token'].str.split(" = ", n = 1, expand = True)[1]
EML['pos']= EML['pos'].str.split(" = ", n = 1, expand = True)[1]
EML['syn_label']= EML['syn_label'].str.split(" = ", n = 1, expand = True)[1]
EML['lemma']= EML['lemma'].str.split(" = ", n = 1, expand = True)[1]
EML['m_desc']= EML['m_desc'].str.split(" = ", n = 1, expand = True)[1]
EML['m_type']= EML['m_type'].str.split(" = ", n = 1, expand = True)[1]
EML.shape 
#(4636, 6)
EML.to_csv ('path_to_write_lexicon.csv', index = None, header=True)

========================================================================
Please do not hesitate to contact us if you have any feedback or suggestion. 
========================================================================
Acknowledgments:
This research is sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-17-2-0196. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation here on. We thank Craig Evans, Tiffany Lu, Max McKittrick, and Jessica Schmiederer for their assistance with expanding and evaluating the morality lexicon. We also thank Chieh-Li (Julian) Chin for her assistance with this project.
