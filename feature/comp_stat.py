"""
functions for preprocessing each sentence from the corpus (Part 2)

"""
import os
from textblob import TextBlob
import spacy 
from joblib import dump, load
import readability
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import gensim

HEDGE_WORDS = "largely generally often rarely sometimes frequently occasionally seldom usually most several some almost practically apparently virtually basically approximately roughly somewhat somehow partially actually like something someone somebody somewhere think thinks thought believe believed believes consider considers considered assume assumes assumed understand understands understood find found finds appear appears appeared seem seems seemed suppose supposes supposed guess guesses guessed estimate estimates estimated speculate speculates speculated suggest suggests suggested may could should might surely probably likely maybe perhaps unsure probable unlikely possibly possible read say says looks like look like don't know necessarily kind of much bunch couple few little really and all that and so forth et cetera in my mind in my opinion their impression my impression in my understanding my thinking is my understanding is in my view if i'm understanding you correctly something or other so far at least  about around can effectively evidently fairly hopefully in general mainly more or less mostly overall presumably  pretty quite clearly quite rather sort of supposedly  tend appear to be doubt be sure indicate will must would certainly definitely clearly conceivably certain definite clear assumption possibility probability  many almost never improbable always rare consistent with doubtful suggestive diagnostic inconclusive apparent alleged allege a bit presumable".split()
FIRST_PERSON_PRONOUNS = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'my', 'mine', 'our', 'ours']
THIRD_PERSON_PRONOUNS = ['he', 'she', 'it', 'one', 'they', 'him', 'her', 'it', 'one', 'them', 'his', 'hers', 'theirs', 'himself', 'herself', 'itself', 'oneself', 'themselves']

def get_readability(text):
    """
    return readability Length of the sentence, in words and characters; Flesch-Kincaid Grade Level score.
    """
    blob = TextBlob(text)
    results = readability.getmeasures(text, lang='en')
    return [len(blob.words), round(results['readability grades']['FleschReadingEase'], 2)]


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

    res.append(round(get_passive_count(text)/length, 2))
    res.append(round(get_count(HEDGE_WORDS)/length, 2))
    res.append(round(get_count(FIRST_PERSON_PRONOUNS)/length, 2))  
    res.append(round(get_count(THIRD_PERSON_PRONOUNS)/length, 2))
    res.append([0,1][blob.sentiment.polarity>=0])
    res.append(round(blob.sentiment.subjectivity, 2))
        
    return res
    
def pre_trained(corpus):
    corpus = api.load(corpus)  # download the corpus and return it opened as an iterable
    model = Word2Vec(corpus)  # train a model from the corpus
    return model
    
def get_word2vec(text):
    """
    return average of word vectors using pre-trained word2vec embeddings, skipping OOV words.
    
    corpus: https://github.com/RaRe-Technologies/gensim-data
    """
    
    if os.path.isfile("model/word2vec.bin"):
        model = Word2Vec.load("model/word2vec.bin")  # you can continue training with the loaded model!
    else: 
        model = pre_trained('text8')  # train a model from the corpus
        model.save("model/word2vec.bin")
        
    return model.build_vocab(text)


if __name__ == "__main__":
    inf1 = "WOW this website IS amazing!!"
    inf2 = "i dunno if they're cool with it"
    inf3 = "i bloody hate it"
    for1 = "I shall not comment further on this issue."
    for2 = "It was said that it was believed that he is a good person."
    for3 = "Listening to what she thinks would be utterly beneficial."
    exs = [inf1, inf2, inf3, for1, for2, for3]

    for sent in exs:
#        print("result: {}, sentence: '{}'" .format(get_readability(sent), sent))
#        print("result:{}, sentence: '{}'".format(get_subjectivity(sent), sent))
        print("result:{}, sentence: '{}'".format(get_word2vec(sent), sent))
        #%%
        

model.save("/Users/loewi/Documents/GitHub/text-formality-classifier/model/word2vec.bin")

model = Word2Vec.load("/Users/loewi/Documents/GitHub/text-formality-classifier/model/word2vec.bin")  # you can continue training with the loaded model!

#%%