# Formality Classifier

## Package Requirements

Here are the packages required to run our program

- Stanford CoreNLP 3.9.2
- TextBlob 0.15.3
- Scikit-learn
- readability 0.3.1
- NLTK
- (possibly) Keras 2.3.1

To install CoreNLP, TextBlot, Keras, and readability, run the following commands in the exact order:

```
$ pip install stanfordnlp
$ pip install textblob
$ pip install Keras
$ pip install readability
```

**For the dependency parser to work, make sure that you also do the following in the interpreter:** (this download will be large, around 235MB)

```
>>> import stanfordnlp
>>> stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
```