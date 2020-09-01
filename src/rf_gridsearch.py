#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
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
# #df
# tf_df = pd.DataFrame(document_tf_matrix, columns=tf.get_feature_names())


#tfidf vectorizer: tf-idf
tfidf = TfidfVectorizer(tokenizer=filter_data_text, max_features=5000)
document_tfidf_matrix = tfidf.fit_transform(X_train).todense()
test_document_tfidf_matrix = tfidf.transform(X_test)
#df
tfidf_df = pd.DataFrame(document_tfidf_matrix, columns=tfidf.get_feature_names())


# X_train_tokens = [filter_data_text(doc) for doc in X_train]
# X_test_tokens = [filter_data_text(doc) for doc in X_test]


##RANDOM FOREST GRIDSEARCH
random_forest_grid = {'max_depth': [None],
                      'max_features': [35, 70],
                      'min_samples_split': [6],
                    #   'min_samples_leaf': [2, 4],
                    #   'bootstrap': [True, False],
                      'n_estimators': [100],
                      'random_state': [1]}

rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                             random_forest_grid,
                            #  n_jobs=-1,
                             verbose=True,
                             cv=2,
                             scoring='neg_log_loss'
                             )

rf_gridsearch.fit(document_tfidf_matrix, y_train)

best_rf_model = rf_gridsearch.best_params_


