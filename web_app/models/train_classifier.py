from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle


def load_data(database_filepath):
    """Function that loads data and returns features, labels and category names.
    Args:
        database_filepath (str): file path to database containing cleaned_data table
    Returns:
        X (Series): features (message)
        Y (DataFrame): labels (categories)
        category_names (Index): category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('cleaned_data', engine)  # load data from database

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """ Function to process text in the following order: normalize, tokenize, 
        remove stop words, and lemmatize.
    Args:
        text (str): text to be processed
    Returns:
        tokens (list): list of tokens after processing text
    """

    # normalize text by removing punctuation and converting to lower case
    text = re.sub(r'[^\w\s]', ' ', text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # remove stop words and lemmatize
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """ Function to build ML model.
    Args:
        None
    Returns:
        pipeline (Pipeline): ML pipeline CountVectorizer -> TfidfTransformer -> MultiOutputClassifier
    """
    # build pipeline using best parameters
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(bootstrap=False)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """ Function that prints classification report for each category.
    Args:
        model (Pipeline): Trained ML pipeline CountVectorizer -> TfidfTransformer -> MultiOutputClassifier
        X_test (Series): features from test data (message)
        Y_test (DataFrame): labels from test data (categories)
        category_names (Index): category names
    Returns:
        None
    """
    Y_pred = model.predict(X_test)  # predict on test data

    # convert y_pred to dataframe
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred.columns = category_names

    # print classification report for each category
    for category in category_names:
        report = classification_report(Y_test[category], Y_pred[category])
        print(report)
        print('---------------------------------------------------\n')


def save_model(model, model_filepath):
    """ Function that exports model as pickle file.
    Args:
        model (Pipeline): Trained ML pipeline CountVectorizer -> TfidfTransformer -> MultiOutputClassifier
        model_filepath (str): file path to pickle file
    Returns:
        None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()