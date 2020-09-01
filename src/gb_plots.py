#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import operator
from text_cleaning import get_data, get_X_y, filter_data_text
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier

##GET DATA
#get dataframes
df_gd, df_pf, df_phish, df_beatles, df = get_data()

#get X and y
X, y = get_X_y(df)

#get training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#count vectorizer: tf
# tf = CountVectorizer(tokenizer=filter_data_text, max_features=5000)
# document_tf_matrix = tf.fit_transform(X_train).todense()
# test_document_tf_matrix = tf.transform(X_test)

#tfidf vectorizer: tf-idf
tfidf = TfidfVectorizer(tokenizer=filter_data_text, max_features=5000)
document_tfidf_matrix = tfidf.fit_transform(X_train).todense()
test_document_tfidf_matrix = tfidf.transform(X_test)

# X_train_tokens = [filter_data_text(doc) for doc in X_train]
# X_test_tokens = [filter_data_text(doc) for doc in X_test]

##GRADIENT BOOSTING CLASSIFIER ACCURACY PLOTS

def stage_score_plot(estimator, X_train, y_train, X_test, y_test, estimator_name):
    '''
    Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                X_train: 2d numpy array
                y_train: 1d numpy array
                X_test: 2d numpy array
                y_test: 1d numpy array

    Returns: A plot of the number of iterations vs the log loss for the model for
    both the training set and test set.
    '''

    estimator.fit(X_train, y_train)
    train_score = np.zeros(estimator.n_estimators)
    test_score = np.zeros(estimator.n_estimators)

    for i, y_pred in enumerate(estimator.staged_predict_proba(X_train)):
        train_score[i] = log_loss(y_train, y_pred)

    for i, y_pred in enumerate(estimator.staged_predict_proba(X_test)):
        test_score[i] = log_loss(y_test, y_pred)


    plt.plot(np.arange(estimator.n_estimators) + 1, train_score,
            label=f"{estimator_name} Training Error")
    plt.plot(np.arange(estimator.n_estimators) + 1, test_score,
            label=f"{estimator_name} Testing Error")
    plt.title("Training and Hold Out Error by Boosting Stages")
    plt.xlabel('Number of Boosting Stages', fontsize=14)
    plt.ylabel('Log Loss', fontsize=14)
    plt.legend(loc="upper right")
    _ = plt.ylim([0, 50])

#define model
gb = GradientBoostingClassifier(learning_rate=.1,
                                  max_depth=3,
                                  n_estimators=100,
                                  random_state=1,
                                #   min_samples_split=2,
                                #   min_samples_leaf=1,
                                  subsample=0.5)
#call plotting function
fig, ax = plt.subplots()
ax = stage_score_plot(gb, document_tfidf_matrix, y_train, test_document_tfidf_matrix,
                        y_test, estimator_name='gradient_boosting_classifier')
plt.legend()
plt.savefig('images/gb_staged_score_plot.png')
plt.show()