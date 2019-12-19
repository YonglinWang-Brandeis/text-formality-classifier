"""
extract all features
"""

import corenlp, textblob

from sklearn.feature_extraction import dict_vectorizer


import pickle
import joblib

ALL_VEC_FEATURES_PATH = "vectorized-features-all.jbl"


def extract_all_features(sent_dict):
    """
    return a list of all features extracted from the input string, by loading existing data or processing corpus
    """

    # TODO 3 types of features:
    #  1) directly vectorized (ngram),
    #  2) non-number, need to be vectorized (entity types -> count vec, pos-num -> dict vec)
    #  3) just numbers (entity length, lexical, subjective...)
    pass

def get_vectorized_features(sent_dict):
    """
    return an array of an array of features for each sentence
    """
    # TODO ngram vectorized sentence
    pass

def get_number_features(sent_dict):
    pass


if __name__ == "__main__":
    pass