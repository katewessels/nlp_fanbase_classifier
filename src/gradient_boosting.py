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
from sklearn.ensemble import GradientBoostingClassifier

##GET DATA
#get dataframes
df_gd, df_pf, df_phish, df_beatles, df = get_data()

#get X and y
X, y = get_X_y(df)

#get training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#count vectorizer: tf
tf = CountVectorizer(tokenizer=filter_data_text, max_features=5000)
document_tf_matrix = tf.fit_transform(X_train).todense()
test_document_tf_matrix = tf.transform(X_test)

#tfidf vectorizer: tf-idf
tfidf = TfidfVectorizer(tokenizer=filter_data_text, max_features=5000)
document_tfidf_matrix = tfidf.fit_transform(X_train).todense()
test_document_tfidf_matrix = tfidf.transform(X_test)

# X_train_tokens = [filter_data_text(doc) for doc in X_train]
# X_test_tokens = [filter_data_text(doc) for doc in X_test]

## GRADIENT BOOSTING CLASSIFIER
#fit
model = GradientBoostingClassifier(random_state=0)
model.fit(document_tfidf_matrix, y_train)
#predict
y_pred = model.predict(test_document_tfidf_matrix)
y_pred_proba = model.predict_proba(test_document_tfidf_matrix)
#metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

###CROSS VALIDATE (default cv: stratified kfold (5-folds), which works for multi class)
# precision_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(precision_score, average='macro'))
# accuracy_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(accuracy_score))
# recall_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(recall_score, average='macro'))
# auc_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring='roc_auc_ovr')

# print(f'Training Mean CV Accuracy: {round(np.mean(accuracy_scores), 5)}')
# print(f'Training Mean CV Precision: {round(np.mean(precision_scores), 5)}')
# print(f'Training Mean CV Recall: {round(np.mean(recall_scores), 5)}')
# print(f'Training Mean CV AUC Score: {round(np.mean(auc_scores), 5)}')
