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
from math import sqrt
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


##RANDOM FOREST ACCURACY PLOTS
#max features v. accuracy
test_acc = []
train_acc = []
num_features = np.arange(1, sqrt(document_tfidf_matrix.shape[1])+1, 1)
for n in num_features:
    model = RandomForestClassifier(max_features=n)
    model.fit(document_tfidf_matrix, y_train)
    #test predict, accuracy
    y_test_pred = model.predict(test_document_tfidf_matrix)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_acc.append(test_accuracy)
    #train predict, accuracy
    y_train_pred = model.predict(document_tfidf_matrix)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_acc.append(train_accuracy)
fig, ax = plt.subplots()
ax.scatter(num_features, test_acc, color='blue', label='test accuracy')
ax.scatter(num_features, train_acc, color='red', label='train accuracy')
ax.grid(b=True)
ax.set_xlabel('Max Features')
ax.set_ylabel('Accuracy')
ax.set_title('Random Forest Classifier Accuracy v. Max Features')
ax.legend()
plt.savefig('images/rf_max_features.png')
plt.show()

# #number of estimators v. accuracy
# test_acc = []
# train_acc = []
# n_estimators = np.arange(5, 50, 5)
# for n in n_estimators:
#     model = RandomForestClassifier(n_estimators=n)
#     model.fit(document_tfidf_matrix, y_train)
#     #test predict, accuracy
#     y_test_pred = model.predict(test_document_tfidf_matrix)
#     test_accuracy = accuracy_score(y_test, y_test_pred)
#     test_acc.append(test_accuracy)
#     #train predict, accuracy
#     y_train_pred = model.predict(document_tfidf_matrix)
#     train_accuracy = accuracy_score(y_train, y_train_pred)
#     train_acc.append(train_accuracy)
# fig, ax = plt.subplots()
# ax.scatter(n_estimators, test_acc, color='blue', label='test accuracy')
# ax.scatter(n_estimators, train_acc, color='red', label='train accuracy')
# ax.grid(b=True)
# ax.set_xlabel('Number of Estimators')
# ax.set_ylabel('Accuracy')
# ax.set_title('Random Forest Classifier Accuracy v. Number of Estimators')
# ax.legend()
# plt.savefig('images/rf_n_estimators.png')
# plt.show()