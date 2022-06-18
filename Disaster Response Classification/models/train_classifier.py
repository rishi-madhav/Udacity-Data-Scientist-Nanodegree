import warnings
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn import multioutput
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


warnings.simplefilter(action='ignore', category=FutureWarning)

# List of stopwords
stop = stopwords.words('english')


def load_data(database_filepath):
    """
    Load the data

    Inputs:
    database_filepath: String. Filepath for the db file containing the cleaned data.

    Output:
    X: dataframe. Contains the feature data.
    y: dataframe. Contains the labels (categories) data.
    category_names: List of strings. Contains the labels names.
    """
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table("labeledmessages", con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Normalize, tokenize and lemmatize text string

    Input:
    text: string- String containing message for processing

    Returns:
    stemmed: list of strings- List containing normalized and lemmatize word tokens
    """
    # get list of URLS using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token and lemmatize, normalize, remove leading/trailing white space and return clean tokenized text
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds a ML pipeline and performs gridsearch.
    Args:
    None
    Returns:
    cv: gridsearchcv object.
    """
    # Create pipeline of the best performing classifier (ADA Boost)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier(
            AdaBoostClassifier(random_state=42)))
    ])

    # Parameters
    parameters = {
        'tfidf__use_idf': [True],
        'clf__estimator__n_estimators': [100],
        'clf__estimator__random_state': [42],
        'clf__estimator__learning_rate': [0.5]
    }
    # Create GridsearchCV
    cv = GridSearchCV(pipeline, param_grid=parameters, refit=True,
                      cv=2, n_jobs=-1, verbose=1, return_train_score=True)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Returns Accuracy, Precision, Recall scores in a Dataframe

    Inputs:
    model: model object & instantiated model
    X_test: Pandas dataframe of the test dataset
    y_test: Pandas dataframe of the target labels
    category_names: list of strings containing category names

    Returns:
    None
    """
    y_pred = model.predict(X_test)
    Y_pred_pd = pd.DataFrame(y_pred, columns=category_names)
<<<<<<< HEAD

    print(classification_report(Y_test, y_pred, target_names=category_names))

    #metrics = []
    # for col in category_names:
    # Store metrics in a list
    #   report = classification_report(Y_test[col], pred[col])
    #   scores = report.split('accuracy')[0].split()
    #   metrics.append([float(scores[i]) for i in [0, 4, 5, 6, 10, 11, 12]])

    # Convert metrics list into a Dataframe
    # metric_names = ['accuracy', 'macro_avg_precision', 'macro_avg_recall',
    #                'macro_avg_f1', 'weighted_avg_precision', 'weighted_avg_recall', 'weighted_avg_f1']
    # metrics_df = pd.DataFrame(
    #   metrics, columns=metric_names, index=category_names)

    # print(metrics_df)
    # print(metrics_df.sum)
    # return metrics_df
=======
    
    print(classification_report(Y_test, y_pred, target_names = category_names))

    #metrics = []
    #for col in category_names:
        # Store metrics in a list
     #   report = classification_report(Y_test[col], pred[col])
     #   scores = report.split('accuracy')[0].split()
     #   metrics.append([float(scores[i]) for i in [0, 4, 5, 6, 10, 11, 12]])

    # Convert metrics list into a Dataframe
    #metric_names = ['accuracy', 'macro_avg_precision', 'macro_avg_recall',
    #                'macro_avg_f1', 'weighted_avg_precision', 'weighted_avg_recall', 'weighted_avg_f1']
    #metrics_df = pd.DataFrame(
     #   metrics, columns=metric_names, index=category_names)

    #print(metrics_df)
    #print(metrics_df.sum)
    #return metrics_df
>>>>>>> 8f3c55f697389df6349f2cb1c693d23058d82852


def save_model(model, model_filepath):
    """
    Saving model's best estimator_ using pickle
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.25, random_state=42)

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
