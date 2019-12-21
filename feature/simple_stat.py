"""
functions for preprocessing each sentence from the corpus (Part 1)
"""

import spacy, collections, readability
from textblob import TextBlob
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


def get_slow_num_stats(string):
    # get the slower-to-compute features

    output = []
    # 1 num
    output.extend(get_entity_length(string))
    # 2 nums
    output.extend(get_readability(string))
    # 6 nums
    output.extend(get_subjectivity(string))

    # total of 9 nums per input sentence
    return output


def get_fast_num_stats(string):
    # get the easier-to-compute features
    output = []
    # 3 nums
    output.extend(get_case(string))
    # 2 nums
    output.extend(get_lexical(string))
    # 1 num
    output.extend(get_punctuation_number(string))

    # total of 6 nums per input sentence
    return output



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
        return [round(total_length/persons_found, 2)]
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
    output.append(round(cont_count/length, 2))

    # 2. average word length
    output.append(round(sum([len(w) for w in tokens])/length, 2))

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


HEDGE_WORDS = "largely generally often rarely sometimes frequently occasionally seldom usually most several some almost practically apparently virtually basically approximately roughly somewhat somehow partially actually like something someone somebody somewhere think thinks thought believe believed believes consider considers considered assume assumes assumed understand understands understood find found finds appear appears appeared seem seems seemed suppose supposes supposed guess guesses guessed estimate estimates estimated speculate speculates speculated suggest suggests suggested may could should might surely probably likely maybe perhaps unsure probable unlikely possibly possible read say says looks like look like don't know necessarily kind of much bunch couple few little really and all that and so forth et cetera in my mind in my opinion their impression my impression in my understanding my thinking is my understanding is in my view if i'm understanding you correctly something or other so far at least  about around can effectively evidently fairly hopefully in general mainly more or less mostly overall presumably  pretty quite clearly quite rather sort of supposedly  tend appear to be doubt be sure indicate will must would certainly definitely clearly conceivably certain definite clear assumption possibility probability  many almost never improbable always rare consistent with doubtful suggestive diagnostic inconclusive apparent alleged allege a bit presumable".split()
FIRST_PERSON_PRONOUNS = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'my', 'mine', 'our', 'ours']
THIRD_PERSON_PRONOUNS = ['he', 'she', 'it', 'one', 'they', 'him', 'her', 'it', 'one', 'them', 'his', 'hers', 'theirs',
                         'himself', 'herself', 'itself', 'oneself', 'themselves']


def get_readability(text):
    """
    return readability Length of the sentence, in words and characters; Flesch-Kincaid Grade Level score.
    """
    try:
        blob = TextBlob(text)
        results = readability.getmeasures(text, lang='en')
        return [len(blob.words), round(results['readability grades']['FleschReadingEase'], 2)]
    except ValueError:
        return[0, 0]


def get_subjectivity(text):
    """
    return number of passive constructions; number of hedge words, according to a word list; number of 1st
    person pronouns; number of 3rd person pronouns; subjectivity according to the TextBlob sentiment
    module; binary indicator for whether the sentiment is positive or negative, according to the TextBlob

    The subjectivity is a float within the range [0.0, 1.0]
    where 0.0 is very objective and 1.0 is very subjective

    pip install spacy
    python -m spacy download en_core_web_sm
    """
    res = []
    blob = TextBlob(text)

    def get_passive_count(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        count = 0
        for tok in doc:
            if tok.dep_.find("subjpass") == True:
                count += 1
        return count

    def get_count(pronouns):
        count = 0
        for pronoun in pronouns:
            count += blob.words.count(pronoun)
        return count

    length = len(blob.words)

    res.append(round(get_passive_count(text) / length, 2))
    res.append(round(get_count(HEDGE_WORDS) / length, 2))
    res.append(round(get_count(FIRST_PERSON_PRONOUNS) / length, 2))
    res.append(round(get_count(THIRD_PERSON_PRONOUNS) / length, 2))
    res.append([0, 1][blob.sentiment.polarity >= 0])
    res.append(round(blob.sentiment.subjectivity, 2))

    return res


if __name__ == "__main__":
    # stanfordnlp.download('en')

    inf1 = "WOW this website IS amazing!!"
    inf2 = "i dunno if Johnny said Alex's cool with going to New York...?"
    for1 = "Joe Biden shall not comment further on this issue."
    for2 = "Listening to what Sam thinks about New York would be utterly beneficial."
    exs = [inf1, inf2, for1, for2]

    for sent in exs:
        print("result: %s, sentence: \"%s\"" % (str(get_fast_num_stats(sent)), sent))
