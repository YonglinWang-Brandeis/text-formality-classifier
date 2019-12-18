"""
extract all features
"""

import corenlp, textblob
from pickle import dump, load

def extract_all_features(sent_list):
    """
    return a list of all features extracted from the input string, by loading existing data or processing corpus
    """

    # TODO 3 types of features:
    #  1) directly vectorized (w2v, ngram...),
    #  2) non-number, need to be vectorized (entity types -> count vec, pos-num -> dict vec)
    #  3) just numbers (entity length, lexical, subjective...)
    pass