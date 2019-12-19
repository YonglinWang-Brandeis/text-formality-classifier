# Formality Classifier

## Package Requirements

Here are the packages required to run our program

- Stanford CoreNLP 3.9.2
- TextBlob 0.15.3
- Scikit-learn
- readability 0.3.1
- NLTK
- Spacy
- (possibly) Keras 2.3.1

To install CoreNLP, TextBlot, Keras, and readability, run the following commands in the exact order:

```
$ pip install stanfordnlp
$ pip install textblob
$ pip install Keras
$ pip install readability
$ pip install spacy
$ python -m spacy download en_core_web_sm
```



MAYBE NOT:

For the dependency parser to work, make sure that you also do the following in the interpreter:** (this download will be large, around 235MB)

```
>>> import stanfordnlp
>>> stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
```



## Troubleshooting

1. Getting *"OSError: [E050] Can't find model 'en_core_web_sm'. "* when running get_subjectivity:

   - Follow the installation instruction in *Package Requirements* section in this file. 

   - Use the following command to check where your spacy package is located. 

     ```$ spacy info```

     Then, make sure the project interpreter of your IDE is in the same directory. 

   - If error persists, try other installation methods listed [here](https://stackoverflow.com/questions/49964028/spacy-oserror-cant-find-model-en)

   