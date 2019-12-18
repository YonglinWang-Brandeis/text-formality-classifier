"""
functions for preprocessing each sentence from the corpus (Part 1)
"""

import corenlp, textblob, stanfordnlp
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag, ngrams
from pickle import dump, load
from nltk.parse import CoreNLPParser


# ++++++++++++++++++++++++++++++++++
#
# Features that need to be one hot encoded (return single immutable variable)
#
# ++++++++++++++++++++++++++++++++++


def get_dependency(string):
    """
    return the following dependency tuples, with lexical items backed off to POS tag: (gov,
    typ, dep), (gov, typ), (typ, dep), (gov, dep).
    """
    pass


def get_entity_types(string):
    """
    return the entity types (e.g. PERSON, LOCATION) occurring in the sentence (binary count vectorizer)
    """
    # process the sentence
    tree = ne_chunk(pos_tag(word_tokenize(string)))
    ners = set()

    # collect unique NE types
    for l in tree:
        try:
            ners.add(l._label)
        except:
            pass

    return list(ners)


def get_ngram(string):
    """
    return the unigrams, bigrams, and trigrams appearing in the sentence. (binary count vectorizer)
    """
    token = word_tokenize(string.lower())
    ngram_list = []

    # get unigrams
    ngram_list.extend([" ".join(t) for t in ngrams(token, 1)])

    # get bigrams
    ngram_list.extend([" ".join(t) for t in ngrams(token, 2)])

    # get trigrams
    ngram_list.extend([" ".join(t) for t in ngrams(token, 3)])

    return ngram_list


# ++++++++++++++++++++++++++++++++++
#
# Features that are just numbers to be appended (return single immutable variable)
#
# ++++++++++++++++++++++++++++++++++


def get_case(string):
    """
    return a number list containing: the number of entirely-capitalized words; binary indicator for whether sentence
    is lowercase; binary indicator for whether the first word is capitalized.
    """
    return [len([w for w in word_tokenize(string) if w.isupper()]), int(string.islower()), int(string[0].isupper())]


def get_entity_length(string):
    """
    return average length, in characters, of PERSON mentions.
    """
    pass


def get_lexical(string):
    """
    return the number of contractions in the sentence, normalized by length; average word length; average word
    log-frequency according to Google Ngram corpus; average formality score as computed by Pavlick and
    Nenkova (2015).
    """
    pass


def get_parse(string):
    """
    return depth of constituency parse tree, normalized by sentence length; number of times each production rule
    appears in the sentence, normalized by sentence length, and not including productions with terminal
    symbols (i.e. lexical items).
    """
    pass


def get_pos_number(string):
    """
    return POS Number of occurrences of each POS tag, normalized by the sentence length.
    """
    pass


def get_punctuation_number(string):
    """
    punctuation Number of ‘?’, ‘...’, and ‘!’ in the sentence.
    """
    pass


if __name__ == "__main__":
    # stanfordnlp.download('en')

    inf1 = "WOW this website IS amazing!!"
    inf2 = "i dunno if Amy is cool with going to New York"
    for1 = "Google shall not comment further on this issue."
    for2 = "Listening to what Sam thinks about New York would be utterly beneficial."
    exs = [inf1, inf2, for1, for2]

    for sent in exs:
        print("result: %s, sentence: \"%s\"" % (str(get_ngram(sent)), sent))
