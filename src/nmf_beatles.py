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
X_train_beatles = X_train[y_train=='beatles']
tfidf_beatles = document_tfidf_matrix[y_train=='beatles']
nmf = NMF(n_components=10)
nmf.fit(tfidf_beatles)
W = nmf.transform(tfidf_beatles)
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

lat_0_docs = X_train_beatles[latent_features==0]
lat_1_docs = X_train_beatles[latent_features==1]
lat_2_docs = X_train_beatles[latent_features==2]
lat_3_docs = X_train_beatles[latent_features==3]
lat_4_docs = X_train_beatles[latent_features==4]
lat_5_docs = X_train_beatles[latent_features==5]
lat_6_docs = X_train_beatles[latent_features==6]
lat_7_docs = X_train_beatles[latent_features==7]
lat_8_docs = X_train_beatles[latent_features==8]
lat_9_docs = X_train_beatles[latent_features==9]

#get each word's highest weighted latent feature
tfidf_feature_names = tfidf.get_feature_names()
latent_features_h=[]
#array of length of bag of words, each value corresponding to highest weighted latent feature
for i in range(H.shape[1]):
    latent_features_h.append(np.argmax(H[:, i]))
latent_features_h = np.array(latent_features_h)

lat_0_words = np.array(H_df.columns[latent_features_h==0])
lat_1_words = np.array(H_df.columns[latent_features_h==1])
lat_2_words = np.array(H_df.columns[latent_features_h==2])
lat_3_words = np.array(H_df.columns[latent_features_h==3])
lat_4_words = np.array(H_df.columns[latent_features_h==4])
lat_5_words = np.array(H_df.columns[latent_features_h==5])
lat_6_words = np.array(H_df.columns[latent_features_h==6])
lat_7_words = np.array(H_df.columns[latent_features_h==7])
lat_8_words = np.array(H_df.columns[latent_features_h==8])
lat_9_words = np.array(H_df.columns[latent_features_h==9])




