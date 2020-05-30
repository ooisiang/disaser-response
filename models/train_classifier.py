import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, f1_score, multilabel_confusion_matrix, classification_report
from sklearn.utils import parallel_backend
import pickle


def load_data(database_filepath):
    """
    This function loads data from the sqlite database given in database_filepath and construct inputs (X and y) for
    the training of machine learning model.

    Args:
        database_filepath (str) -- filepath to the sqlite database

    Return:
        X (df) -- a dataframe consists of the disaster messages
        y (df) -- a dataframe consists of the disaster categories
        category_names (df.columns)- a list of the category names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponseTable', engine)
    X = df['message']
    y = df.iloc[:, -36:]
    category_names = y.columns

    return X, y, category_names


def get_wordnet_tag(pos):
    """
    This function aims to convert limited NLTK pos tags to WordNet tags based on a self-constructed dict.
    This function only considers adjectives, nouns, verbs and adverbs. The other pos tags will be converted to nouns.

    Args:
        pos (str) -- first character of the nltk pos tag

    Return:
        wordnet tag (str) -- WordNet tag
    """
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(pos, wordnet.NOUN)


def tokenize(text):
    """
    This function aims to tokenize, normalize and lemmatitze each disaster message in the dataset.
    This will make all words to lowercase, remove all punctuations and english stopwords from the disaster message.
    Token will be lemmatized according to its WordNet tag.

    Args:
        text (str) -- single disaster message in the dataset

    Return:
        words (list) -- a list of words/tokens extracted from the disaster message
    """

    words = []
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()

    for token in tokens:
        if token not in stopwords.words("english"):
            token_pos = nltk.pos_tag(tokens)[0][1][0]
            words.append(lemmatizer.lemmatize(token, pos=get_wordnet_tag(token_pos)).strip())

    return words


def build_model():
    """
    This function aims to build a machine learning pipeline model with GridSearchCV.
    The pipeline consists of CountVectorizer with the tokenize() function in this file, TfidfTransfomer and
    RandomForestClassifier.
    GridSearchCV is used to tune the hyperparameters of the transformers and classifier.

    Args:
        none

    Return:
        cv (sklearn model) -- a scikit-learn machine learning pipeline model after hyperparameters grid search
         for data training and prediction.
    """

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {
        'vect__ngram_range': ((1, 1), (2, 2)),
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__min_samples_leaf': [1, 5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function aims to evaluate the performance of a multioutput classifier by returning the overall
    precision, recall, f1-score and accuracy of the model. Overall scores are computed by taking the mean of the scores
    of each output/category.

    Args:
        model (sklearn model) -- a multioutput classifier model
        X_test (df) -- a test dataset consists of the disaster messages
        Y_test (df) -- a test dataset consists of the disaster categories (one hot encoding)
        category_names (df.columns) - a list of the category names

    Return:
        none
    """

    # Predict using model
    y_pred = model.predict(X_test)

    # Convert one hot encodings to the original labels
    y_pred_labeled = np.multiply(y_pred, category_names)
    y_test_labeled = np.multiply(np.array(Y_test), category_names)

    # Convert the y_pred_labeled and y_test_labeled from 2d arrays to 1d arrays
    y_pred_labeled_1d = np.reshape(y_pred_labeled, y_pred_labeled.shape[0] * y_pred_labeled.shape[1])
    y_test_labeled_1d = np.reshape(y_test_labeled, y_test_labeled.shape[0] * y_test_labeled.shape[1])

    # Compute the overall accuracy of the model
    accuracy = cal_accuracy(np.array(Y_test), y_pred)

    # Compute the classification results for each category
    classification_results = classification_report(y_test_labeled_1d, y_pred_labeled_1d, labels=category_names,
                                                   output_dict=True, zero_division=1)
    classification_results = pd.DataFrame(classification_results).T

    # Compute the mean of the classification results (precision, recall, f1-score, support)
    precision, recall, f1_score, support = classification_results.mean()

    print("    Precision: {}\n    Recall: {}\n    F1-Score: {}\n    Accuracy: {}\n"
          .format(precision, recall, f1_score, accuracy))


def cal_accuracy(y_act, y_pred):
    """
    This function aims to calculate the accuracy of the multioutput classifier.

    Args:
        y_act (np array) -- actual labels of the test datasets (one hot encoding)
        y_pred (np array) -- predicted labels of the test datasets (one hot encoding)

    Return:
        accuracy (float) -- accuracy of the multioutput classifier.
    """

    wrong_counts = 0
    total_rows = y_pred.shape[0]

    for row in range(total_rows):
        wrong_counts += sum(abs(y_act[row] - y_pred[row]))

    accuracy = 1 - (wrong_counts / (total_rows * y_pred.shape[1]))

    return accuracy


def save_model(model, model_filepath):
    """
    This function aims to save a trained model to the given destination with the given pickle file name.

    Args:
        model (sklearn model) -- trained model for predictions
        model_filepath (str) -- filepath with name of the pickle file where the trained model should be saved.

    Return:
        none
    """

    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        with parallel_backend('multiprocessing'):
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