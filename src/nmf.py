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
from sklearn.decomposition import NMF

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

## NON-NEGATIVE MATRIX FACTORIZATION
#fit
#try 10 latent features
nmf = NMF(n_components=10)
nmf.fit(document_tfidf_matrix)
W = nmf.transform(document_tfidf_matrix)
H = nmf.components_
print(f'nmf_reconstruction_error: {nmf.reconstruction_err_}')

#make interpretable
W, H = (np.around(x, 4) for x in (W, H))
W_df = pd.DataFrame(W) #rows correspond to documents (columns: latent features)
H_df = pd.DataFrame(H, columns=tfidf.get_feature_names()) #columns correspond to bag of words (rows: latent features)


# interpret latent features
#get each document's highest weighted latent feature
latent_features=[]
for i in range(W.shape[0]):
    latent_features.append(np.argmax(W[i, :]))
latent_features = np.array(latent_features)

lat_0_docs = X_train[latent_features==0]
print(f'latent feature 1: {lat_0_docs.shape}')
print('-------------------------------------')
print(lat_0_docs[:20])
lat_1_docs = X_train[latent_features==1]
lat_2_docs = X_train[latent_features==2]
lat_3_docs = X_train[latent_features==3]
lat_4_docs = X_train[latent_features==4]
lat_5_docs = X_train[latent_features==5]
lat_6_docs = X_train[latent_features==6]
lat_7_docs = X_train[latent_features==7]
lat_8_docs = X_train[latent_features==8]
lat_9_docs = X_train[latent_features==9]

#get each word's highest weighted latent feature
tfidf_feature_names = tfidf.get_feature_names()
latent_features_h=[]
#array of length of bag of words, each value corresponding to highest weighted latent feature
for i in range(H.shape[1]):
    latent_features_h.append(np.argmax(H[:, i]))
latent_features_h = np.array(latent_features_h)

index_0 = (np.argwhere(latent_features_h==0)).reshape(-1,)
words_0 = []
for idx in index_0:
    words_0.append(tfidf_feature_names[idx])

index_4 = (np.argwhere(latent_features_h==4)).reshape(-1,)
words_4 = []
for idx in index_4:
    words_4.append(tfidf_feature_names[idx])



# #try a model
# #NAIVE BAYES USING TF COUNT VECTORIZER
# #transform test data into latent features as well
# test_W = nmf.transform(test_document_tfidf_matrix)
# #model
# model = MultinomialNB()
# model.fit(W, y_train)
# #predict
# y_pred = model.predict(test_W)
# y_pred_proba = model.predict_proba(test_W)
# #metrics
# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred, average='macro')
# precision = precision_score(y_test, y_pred, average='macro')
# auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
# #classification report
# report = classification_report(y_test, y_pred)



#metrics
# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred, average='macro')
# precision = precision_score(y_test, y_pred, average='macro')
# auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

###CROSS VALIDATE (default cv: stratified kfold (5-folds), which works for multi class)
# precision_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(precision_score, average='macro'))
# accuracy_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(accuracy_score))
# recall_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring=make_scorer(recall_score, average='macro'))
# auc_scores = cross_val_score(model, scaled_tfidf_matrix, y_train, scoring='roc_auc_ovr')

# print(f'Training Mean CV Accuracy: {round(np.mean(accuracy_scores), 5)}')
# print(f'Training Mean CV Precision: {round(np.mean(precision_scores), 5)}')
# print(f'Training Mean CV Recall: {round(np.mean(recall_scores), 5)}')
# print(f'Training Mean CV AUC Score: {round(np.mean(auc_scores), 5)}')


#RESULTS
#for 50 latent features:
#accuracy=.6176
#precision=0.6595
#recall=0.6183
#auc=0.8432
