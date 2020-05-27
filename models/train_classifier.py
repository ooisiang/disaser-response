import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split


def load_data(database_filepath):
    """
    This function loads data from the sqlite database given in database_filepath and construct inputs (X and y) for
    the training of machine learning model.

    Args:
        database_filepath (str) -- filepath to the sqlite database

    Return:
        X (df) -- a dataframe consists of the disaster messages
        y (df) -- a dataframe consists of the disaster categories
        category_names - a list of the category names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponseTable', engine)
    X = df['message']
    y = df.iloc[:, -36:]
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):
    """
    This function aims to tokenize, normalize and lemmatitze each disaster message in the dataset.
    This will make all words to lowercase, remove all punctuations and english stopwords from the disaster message.

    Args:
        text (str) -- single disaster message in the dataset

    Return:
        words (list) -- a list of words/tokens extracted from the disaster message
    """

    words = []
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()

    for token in tokens:
        if (token not in stopwords.words("english")) & (token not in string.punctuation):
            words.append(lemmatizer.lemmatize(token, pos='v').strip())

    return words


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()