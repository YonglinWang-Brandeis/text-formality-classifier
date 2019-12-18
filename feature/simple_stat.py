"""
functions for preprocessing each sentence from the corpus (Part 1)
"""

import corenlp, textblob, stanfordnlp, collections
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
    #TODO instead use: bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
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


def get_pos_number(string):
    """
    return POS Number of occurrences of each POS tag, normalized by the sentence length. (Dict vectorizer)
    """
    return dict(collections.Counter([j for i,j in pos_tag(word_tokenize(string))]))




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
    tokens = word_tokenize(string)
    tree = ne_chunk(pos_tag(tokens))

    total_length = 0
    persons_found = 0
    # collect unique NE types
    for l in tree:
        try:
            if l._label == "PERSON":
                total_length += len(l[0][0])
                persons_found += 1
        except:
            pass

    if persons_found:
        return [total_length/persons_found]
    else:
        return [0]


def get_lexical(string):
    """
    return the number of contractions in the sentence, normalized by length; average word length; average word
    log-frequency according to Google Ngram corpus; average formality score as computed by Pavlick and
    Nenkova (2015).
    """
    # prepare
    output = []
    tokens = word_tokenize(string.lower())
    length = len(tokens)

    # 1. number of contractions, norm by length
    cont_count = 0
    for w in tokens:
        if "\'" in w and len(w) > 1:
            cont_count += 1
    output.append(cont_count/length)

    # 2. average word length
    output.append(sum([len(w) for w in tokens])/length)

    # 3. average word log-freq

    # 4. average formality score

    return output


def get_parse(string):
    """
    return depth of constituency parse tree, normalized by sentence length; number of times each production rule
    appears in the sentence, normalized by sentence length, and not including productions with terminal
    symbols (i.e. lexical items).
    """
    pass






def get_punctuation_number(string):
    """
    punctuation Number of ‘?’, ‘...’, and ‘!’ in the sentence.
    """
    punct_number = 0
    tokens = word_tokenize(string)

    for w in tokens:
        if w in ["?", "...", "!"]:
            punct_number += 1
    return [punct_number]



if __name__ == "__main__":
    # stanfordnlp.download('en')

    inf1 = "WOW this website IS amazing!!"
    inf2 = "i dunno if Johnny said Alex's cool with going to New York...?"
    for1 = "Joe Biden shall not comment further on this issue."
    for2 = "Listening to what Sam thinks about New York would be utterly beneficial."
    exs = [inf1, inf2, for1, for2]

    for sent in exs:
        print("result: %s, sentence: \"%s\"" % (str(get_pos_number(sent)), sent))
