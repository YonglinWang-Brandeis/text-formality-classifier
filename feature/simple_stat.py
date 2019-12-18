"""
functions for preprocessing each sentence from the corpus (Part 1)
"""

import corenlp, textblob
from nltk.tokenize import word_tokenize
from pickle import dump, load


def get_case(str):
    """
    return a number list containing: the number of entirely-capitalized words; binary indicator for whether sentence is lowercase; binary indicator
    for whether the first word is capitalized.
    """
    return [len([w for w in word_tokenize(str) if w.isupper()]), int(str.islower()), int(str[0].isupper())]


def get_dependency(str):
    """
    return the following dependency tuples, with lexical items backed off to POS tag: (gov,
    typ, dep), (gov, typ), (typ, dep), (gov, dep).
    """
    pass


def get_entity(str):
    """
    return the entity types (e.g. PERSON, LOCATION) occurring in the sentence; average
    length, in characters, of PERSON mentions.
    """
    pass


def get_lexical(str):
    """
    return the number of contractions in the sentence, normalized by length; average word length; average word
    log-frequency according to Google Ngram corpus; average formality score as computed by Pavlick and
    Nenkova (2015).
    """
    pass


def get_ngram(str):
    """
    return the unigrams, bigrams, and trigrams appearing in the sentence.
    """
    pass


def get_parse(str):
    """
    return depth of constituency parse tree, normalized by sentence length; number of times each production rule
    appears in the sentence, normalized by sentence length, and not including productions with terminal
    symbols (i.e. lexical items).
    """
    pass


def get_pos_number(str):
    """
    return POS Number of occurrences of each POS tag, normalized by the sentence length.
    """
    pass


def get_punctuation_number(str):
    """
    punctuation Number of ‘?’, ‘...’, and ‘!’ in the sentence.
    """
    pass


if __name__ == "__main__":
    inf1 = "WOW this website IS amazing!!"
    inf2 = "i dunno if they're cool with it"
    for1 = "I shall not comment further on this issue."
    for2 = "Listening to what she thinks would be utterly beneficial."
    exs = [inf1, inf2, for1, for2]

    for sent in exs:
        print("result: %s, sentence: \"%s\"" % (str(get_case(sent)), sent))
