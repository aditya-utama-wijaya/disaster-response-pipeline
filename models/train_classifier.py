import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Load data from SQLite database and split it into features and target variables.
    
    Args:
        database_filepath (str): Filepath of the SQLite database.
    
    Returns:
        X (pandas.DataFrame): Features.
        y (pandas.DataFrame): Target variables.
        category_names (list): List of category names.
    """
    # load data from database
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("disaster_response", engine)
    X = df[["message"]]
    y = df[df.columns[4:]]
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):
    """
    Tokenize the given text.
    
    Args:
        text (str): Text to be tokenized.
    
    Returns:
        tokens (list): List of tokens.
    """
    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Define stop words
    stop_words = set(stopwords.words("english"))

    # Convert the tokens to lowercase
    tokens = [t.lower() for t in tokens if t not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def build_model():
    """
    Build the machine learning pipeline.
    
    Returns:
        pipeline (sklearn.pipeline.Pipeline): Machine learning pipeline.
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer = tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Define the parameters for grid search
    parameters = {
        "vect__max_df": (0.5, 0.75, 1.0),
        "clf__estimator__random_state": [0],
        "clf__estimator__n_estimators": range(20, 301, 10),
        "clf__estimator__max_features": list(range(7, 20, 2)) + ["auto"],
        "clf__estimator__max_depth": list(range(3, 20, 2)) + [None],
        "clf__estimator__min_samples_split": range(2, 1003, 100),
        "clf__estimator__min_samples_leaf": range(1, 72, 10),
    }

    # Perform grid search
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model by calculating precision, recall, and F1-score for each category.
    
    Args:
        model: Trained machine learning model.
        X_test (pandas.Series): Test set features.
        Y_test (pandas.DataFrame): Test set targets.
        category_names (list): List of category names.
    """
    # Make predictions on the test set
    Y_pred = model.predict(X_test)

    # Iterate through each output category
    for i, c in enumerate(category_names):
        print("Category: ", c)
        print(classification_report(Y_test[column], Y_pred[:, i]))
        print()


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Args:
        model: Trained machine learning model.
        model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


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