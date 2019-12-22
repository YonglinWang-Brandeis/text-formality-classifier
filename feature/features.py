"""
extract all features
"""

import textblob
import time
import numpy as np

from feature import simple_stat
from preprocessing import corpus

from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer

from joblib import load, dump
import pickle


# feature paths
ALL_VEC_FEATURES_PATH = "vectorized-features-all.jbl"
NGRAM_FEAT_PATH = "ngram_feature.jbl"
ENT_FEAT_PATH = "ent_feature.jbl"
POS_FEAT_PATH = "pos_feature.jbl"
FAST_NUM_FEAT_PATH = "fast_num_feature.jbl"
SUBJ_FEAT_PATH = "subjectivity_feature.jbl"
READ_FEAT_PATH = "readability_feature.jbl"
ENT_LEN_FEAT_PATH = "entity_length_feature.jbl"

# vector paths
NGRAM_VEC_PATH = "../vectorizer/ngram_count_vectorizer.jbl"
ENT_VEC_PATH = "../vectorizer/ent_count_vectorizer.jbl"
POS_VEC_PATH = "../vectorizer/pos_feat_harsher.jbl"


def extract_all_features():
    """
    return a list of all features extracted from the input string, by loading existing data or processing corpus
    """
    try:
        return load(open(ALL_VEC_FEATURES_PATH, "rb"))
    except IOError:

        # get the sentence dictionary
        try:
            sent_dict = pickle.load(open("../preprocessing/corpus_dict.pkl", "rb"))
        except IOError:
            sent_dict = corpus.load_corpus_dict()

        sent_list = []
        sent_list.extend(sent_dict["formal"])
        sent_list.extend(sent_dict["informal"])

        # load feature list
        print("loading all features")
        all_feat_vec = load_all_features(sent_list)

        # horizontally stack all features
        print("stacking all features")
        output = all_feat_vec[0]

        for i in range(1, len(all_feat_vec)):
            output = np.hstack((output, all_feat_vec[i])) # debug pause here

        # save the output
        print("dumping all features")
        dump(output, ALL_VEC_FEATURES_PATH)

        print("done!")
        return output



def load_all_features(sent_list):
    """
    return a list of all the feature group arrays for horizontal stacking
    """

    all_feat_list = []

    all_feat_list.append(get_ngram_features(sent_list).toarray())
    all_feat_list.append(get_entity_features(sent_list).toarray())
    all_feat_list.append(get_pos_features(sent_list).toarray())
    all_feat_list.append(get_fast_number_features(sent_list))
    all_feat_list.append(get_readability_features(sent_list))
    all_feat_list.append(get_entity_length_features(sent_list))

    return all_feat_list

def get_ngram_features(sent_list: list):
    """
    return an array of an array of ngram features for each sentence
    """
    try:
        return load(open(NGRAM_FEAT_PATH, "rb"))
    except IOError:
        print("Cannot load ngram vectors, now extracting creating vectorizer and extracting feature...")
        cv = CountVectorizer(ngram_range=(1,3))
        output = cv.fit_transform(sent_list)

        # save the feature vectores
        dump(output, NGRAM_FEAT_PATH)

        # save the vectorizer
        dump(cv, NGRAM_VEC_PATH)

        return output


def get_entity_features(sent_list: list):
    """
    return an array of an array of features for each sentence
    """
    try:
        return load(open(ENT_FEAT_PATH, "rb"))
    except IOError:
        print("Cannot load entity vectors, now extracting creating vectorizer and extracting feature...")
        cv = CountVectorizer(tokenizer=dummy_processor, preprocessor=dummy_processor, binary=True)

        # get list of entities for each sentence
        # TODO Delete test printing!!
        print("getting raw features... NER will take about 15 minutes.")
        ent_lists = [simple_stat.get_entity_types(s) for s in sent_list]
        print("features created")
        output = cv.fit_transform(ent_lists)

        # save the feature vectores
        dump(output, ENT_FEAT_PATH)

        # save the vectorizer
        dump(cv, ENT_VEC_PATH)

        return output


def get_pos_features(sent_list: list):
    """
    return an array of an array of features for each sentence
    """
    try:
        return load(open(POS_FEAT_PATH, "rb"))
    except IOError:
        print("Cannot load POS vectors, now extracting creating vectorizer and extracting feature...")
        dv = FeatureHasher()

        # get list of entities for each sentence
        print("getting raw features...")
        pos_dicts = [simple_stat.get_pos_number(s) for s in sent_list]
        print("features created")
        output = dv.fit_transform(pos_dicts)

        # save the feature vectores
        dump(output, POS_FEAT_PATH)

        # save the vectorizer
        dump(dv, POS_VEC_PATH)

        return output


def get_fast_number_features(sent_list):
    """
    return an array of an array of features for each sentence
    """
    try:
        return load(open(FAST_NUM_FEAT_PATH, "rb"))
    except IOError:
        print("Cannot load number vectors, now extracting feature...")

        # get list of raw number stats for each sentence
        print("getting raw features...")
        num_lists = []
        for s in sent_list:
            num_lists.append(simple_stat.get_fast_num_stats(s))
        print("features created")

        # list into vec array
        output = np.array(num_lists)

        # save the feature vectores
        dump(output, FAST_NUM_FEAT_PATH)

        return output


def get_subjectivity_features(sent_list):
    """
    return an array of an array of features for each sentence
    """
    # TODO generate these if time allows
    try:
        return load(open(SUBJ_FEAT_PATH, "rb"))
    except IOError:
        print("Cannot load number vectors, now extracting feature...")

        # get list of raw number stats for each sentence
        print("getting raw features...")
        num_lists = []
        for s in sent_list:
            num_lists.append(simple_stat.get_subjectivity(s))
        print("features created")

        # list into vec array
        output = np.array(num_lists)

        # save the feature vectores
        dump(output, SUBJ_FEAT_PATH)

        return output


def get_readability_features(sent_list):
    """
    return an array of an array of features for each sentence
    """
    try:
        return load(open(READ_FEAT_PATH, "rb"))
    except IOError:
        print("Cannot load number vectors, now extracting feature...")

        # get list of raw number stats for each sentence
        print("getting raw features...")
        num_lists = []
        for s in sent_list:
            num_lists.append(simple_stat.get_readability(s))
        print("features created")

        # list into vec array
        output = np.array(num_lists)

        # save the feature vectores
        dump(output, READ_FEAT_PATH)

        return output


def get_entity_length_features(sent_list):
    """
    return an array of an array of features for each sentence
    """
    try:
        return load(open(ENT_LEN_FEAT_PATH, "rb"))
    except IOError:
        print("Cannot load entity length vectors, now extracting feature...")

        # get list of raw number stats for each sentence
        print("getting raw features...")
        num_lists = []
        for s in sent_list:
            num_lists.append(simple_stat.get_entity_length(s))
        print("features created")

        # list into vec array
        output = np.array(num_lists)

        # save the feature vectores
        dump(output, ENT_LEN_FEAT_PATH)
        print("done!")

        return output


def dummy_processor(word):
    return word


if __name__ == "__main__":
    test = extract_all_features()
