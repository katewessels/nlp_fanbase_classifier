
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline
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
from text_cleaning import get_data, get_X_y, filter_data, filter_data_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, classification_report, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler

#get dataframes
df_gd, df_pf, df_phish, df_beatles, df = get_data()

#get X and y
X, y = get_X_y(df)

#get training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# #count vectorizer: tf
# tf = CountVectorizer(tokenizer=filter_data_text, max_features=5000)
# document_tf_matrix = tf.fit_transform(X_train).todense()
# test_document_tf_matrix = tf.transform(X_test)
# #scale data
# scaler = StandardScaler()
# scaled_tf_matrix = scaler.fit_transform(document_tf_matrix)
# test_scaler = StandardScaler()
# scaled_test_tf_matrix = test_scaler.fit_transform(test_document_tf_matrix)

#tfidf vectorizer: tf-idf
tfidf = TfidfVectorizer(tokenizer=filter_data_text, max_features=5000)
document_tfidf_matrix = tfidf.fit_transform(X_train).todense()
test_document_tfidf_matrix = tfidf.transform(X_test)
#scale data
scaler = StandardScaler(with_mean=False)
scaled_tfidf_matrix = scaler.fit_transform(document_tfidf_matrix)
test_scaler = StandardScaler(with_mean=False)
scaled_test_tfidf_matrix = test_scaler.fit_transform(test_document_tfidf_matrix)

# X_train_tokens = [filter_data_text(doc) for doc in X_train]
# X_test_tokens = [filter_data_text(doc) for doc in X_test]

# ## LOGISTIC REGRESSION MODEL USING TF COUNT VECTORIZER
# model = LogisticRegression()
# model.fit(scaled_tf_matrix, y_train)
# y_pred = model.predict(scaled_test_tf_matrix)
# y_pred_proba = model.predict_proba(scaled_test_tf_matrix)

## LOGISTIC REGRESSION MODEL USING TF-IDF  VECTORIZER
model = LogisticRegression(max_iter=1000, multi_class='multinomial')
model.fit(scaled_tfidf_matrix, y_train)
y_pred = model.predict(scaled_test_tfidf_matrix)
y_pred_proba = model.predict_proba(scaled_test_tfidf_matrix)

## METRICS
#accuracy score on test data
accuracy = model.score(scaled_test_tfidf_matrix, y_test)
#predicted classes for test data
y_pred = model.predict(scaled_test_tfidf_matrix)
#predicted probabilities for test data
y_pred_proba = model.predict_proba(scaled_test_tfidf_matrix)
#roc AUC score
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
#classification report
report = classification_report(y_test, y_pred)

###CROSS VALIDATE (default cv: stratified kfold (5-folds), which works for multi class)
precision_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(precision_score, average='macro'))
accuracy_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(accuracy_score))
recall_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(recall_score, average='macro'))
auc_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring='roc_auc_ovr')

print(f'Training Mean CV Accuracy: {round(np.mean(accuracy_scores), 5)}')
print(f'Training Mean CV Precision: {round(np.mean(precision_scores), 5)}')
print(f'Training Mean CV Recall: {round(np.mean(recall_scores), 5)}')
print(f'Training Mean CV AUC Score: {round(np.mean(auc_scores), 5)}')

##RESULTS
#without scaling:
#accuracy=0.787
#auc = .952

#with standard scaler:
#accuracy=0.776
#auc=0.946
