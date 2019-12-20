"""
Extract all corpus sentence strings and put them in a list

WARNING: You should NOT expect to run this file because the author of the corpus requires the corpus not
be publicly shared, please use the preprocessed sent_list.pkl file instead. Contact us if the sent_list.pkl file
cannot be processed.
"""


from pickle import dump, load

SENT_TYPE = ["formal", "informal"]
CORPUS_PATH_DICT = {"formal": "../data/formal.txt", "informal": "../data/informal.txt"}

SENT_DICT_PATH = "corpus_dict.pkl"


def load_corpus_dict():
    try:
        # load the list of sentence strings
        return load(open(SENT_DICT_PATH, "r"))
    except IOError:
        # first time running, extract all sentences from corpus

        # dictionary of {"formal": all formal sentences, "informal": all informal sentences}
        sent_dict = {"label": []}

        for label in SENT_TYPE:
            sent_list = []
            f = open(CORPUS_PATH_DICT[label], "r")

            # get each line into the list
            sent = f.readline()
            while sent:
                sent_list.append(sent)
                sent_dict["label"].append(label)
                sent = f.readline()

            # enter the list to corpus
            sent_dict[label] = sent_list

        # pickle the corpus dictionary
        dump(sent_dict, open(SENT_DICT_PATH, "wb"))

        return sent_dict

if __name__ == "__main__":
    corp_d = load_corpus_dict()
    print(len(corp_d))
