"""
functions for preprocessing each sentence from the corpus (Part 2)

"""

import corenlp, textblob
from pickle import dump, load


def get_readability(str):
    """
    return readability Length of the sentence, in words and characters; Flesch-Kincaid Grade Level score.
    """
    pass


def get_subjectivity(str):
    """
    return number of passive constructions; number of hedge words, according to a word list; number of 1st
    person pronouns; number of 3rd person pronouns; subjectivity according to the TextBlob sentiment
    module; binary indicator for whether the sentiment is positive or negative, according to the TextBlob
    """
    pass


def get_word2vec(str):
    """
    return average of word vectors using pre-trained word2vec embeddings, skipping OOV words.
    """
    pass