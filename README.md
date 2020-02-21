# Formality Classifier

## Documentation

This is a text formality classifier project that examines how different machine learning models perform in classifying a text as formal or informal. 

Detailed documentations of the project, such as the project proposal and the project report, can be found in [`doc`](doc) directory. 

## Package Requirements

Here are the packages required to run our program

- [`TextBlob 0.15.3`](https://textblob.readthedocs.io/en/dev/)
- [`Scikit-learn`](https://scikit-learn.org/stable/)
- [`readability 0.3.1`](https://pypi.org/project/readability/)
- [`NLTK`](https://www.nltk.org)
- [`spaCy`](https://spacy.io)
- [`Keras 2.3.1`](https://keras.io)

To install the latest version of TextBlot, Keras, readability, and spacy, run the following commands in the exact order:

```
$ pip install textblob
$ pip install Keras
$ pip install readability
$ pip install spacy
$ python -m spacy download en_core_web_sm
```



## Running the Code

Since all the feature vectors, vectorizers, and models are saved with joblib, you can skip right to the **Model Training** and **Predicting** sections. 

#### Loading Corpus

As requested by the original owner of the corpus, the raw text version of the corpus will **NOT** be included in this public repository. 

- Instead, we have pickled the dictionary object containing the corpus and corresponding tags. To load the corpus, run ***corpus.py*** under *preprocessing* folder.

#### Feature Extraction and encoding (very time consuming)

To perform all the feature extraction/load the vectorized features, run ***features.py*** under *feature* folder. 

#### Model Training (slightly time consuming)

To see how the model training works, run the following Jupyter Notebooks in the toplevel of the repository:

- **models_raw.ipynb**: model training using the raw text of a sentence
  - Models Used: Naive Bayes (MultinomialNB), Logistic Regression (LogisticRegression), Decision Tree (DecisionTreeClassifier) 
- **models_features.ipynb**: model training using different features extracted from a given sentence (not the sentence itself)
  - Features: entity types, entity length, simple numerical stats (fast number), n-gram, readability
  - Models Used: Naive Bayes (Multinomial NB), Logistic Regression (LogisticRegression), Decision Tree (DecisionTreeClassifier) 
- **model_GloVe.ipynb**: model training using LSTM

See the write-up for an interpretation for the model evaluation. 

#### Predicting (Beta Feature)

To see how the model makes a prediction for a few example sentences, open ***prediction.ipynb*** in the toplevel of the repository. 

- Due to some compatibility issues, there's a portion of models that cannot be loaded for prediction. We are currently working on this. 



## Troubleshooting

1. Getting *"OSError: [E050] Can't find model 'en_core_web_sm'. "* when running get_subjectivity:

   - Follow the installation instruction in *Package Requirements* section in this file. 

   - Use the following command to check where your spacy package is located. 

     ```$ spacy info```

     Then, make sure the project interpreter of your IDE is in the same directory. 

   - If error persists, try other installation methods listed [here](https://stackoverflow.com/questions/49964028/spacy-oserror-cant-find-model-en)

   