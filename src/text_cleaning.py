import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline
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
from wordcloud import WordCloud, STOPWORDS

#get data
def get_data():
    df_gd = pd.read_csv('data/gratefuldead.csv')
    df_pf = pd.read_csv('data/pinkfloyd.csv')
    df_phish = pd.read_csv('data/phish.csv')
    df_beatles = pd.read_csv('data/beatles.csv')
    #combine to single df
    df_sorted = pd.concat([df_gd, df_pf, df_phish, df_beatles], join  = 'outer', axis = 0, ignore_index = True)
    #shuffle rows so not sorted by band page
    df = df_sorted.sample(frac = 1, random_state=1)

    return df_gd, df_pf, df_phish, df_beatles, df

#get X and y
def get_X_y(df):
    df = df.set_index(df['id']).drop(columns=['id'])
    X = df['title'].to_numpy()
    y = df['subreddit'].to_numpy()
    return X, y

##clean text

#get unicode data
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

#tokenize and get stem of each word. see which stemmer is preferred
# def tokenize(text):
#     stemmer = SnowballStemmer('english')
#     # stemmer = PorterStemmer()
#     # stemmer = WordNetLemmatizer()
#     return [stemmer.stem(word) for word in word_tokenize(text.lower())]

#get rid of accents, punctuation, make lowercase
# def filter_data(corpus):
#     X = corpus.copy()
#     punctuation = string.punctuation
#     stop_words = set(stopwords.words('english'))
#     stemmer = SnowballStemmer('english')
#     for i in range(len(X)):
#         #get unicode data
#         X[i] = remove_accents(X[i])
#         #get rid of punctuation and make lowercase
#         X[i] = X[i].translate(str.maketrans('', '', punctuation)).lower()
#         #make bag of words, without stop words
#         tokenized_words = [word for word in X[i].split() if word not in stop_words]
#         #get the stem of each word
#         X[i] = [stemmer.stem(word) for word in tokenized_words]
#     return X

def filter_data_text(text):
    X = text
    punctuation = string.punctuation
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')

    #get unicode data
    X = remove_accents(X)
    #get rid of punctuation and make lowercase
    X = X.translate(str.maketrans('', '', punctuation)).lower()
    #make bag of words, without stop words
    tokenized_words = [word for word in X.split() if word not in stop_words]
    #get the stem of each word
    X = [stemmer.stem(word) for word in tokenized_words]

    return X

def get_cosine_similarities(document_tfidf_matrix):
    cosine_similarities = linear_kernel(document_tfidf_matrix, document_tfidf_matrix)
    output = []
    for i in range(len(document_tfidf_matrix)):
        for j in range(len(document_tfidf_matrix)):
            output.append(i, j, cosine_similarities[i, j])
    return output

def get_top_n_words(corpus, n=100):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.

    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) ->
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer(tokenizer=filter_data_text).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def plot_top_words(list_of_tuples, n_documents, xlabel, ylabel, title, string_file_name):
    '''
    takes a list of tuples: (word, count), sorted by top used words

    '''
    word = list(zip(*list_of_tuples))[0][:15]
    count = list(zip(*list_of_tuples))[1][:15]
    freq = tuple(ti/n_documents for ti in count)

    x_pos = np.arange(len(word))

    fig, ax = plt.subplots(figsize=(12,8))
    ax.barh(x_pos, freq, align='center')
    plt.yticks(x_pos, word)
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    plt.gca().invert_yaxis()
    plt.savefig(f'images/{string_file_name}.png')
    # plt.show()

def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")

# Generate word cloud from dictionary of 100 top words + their counts
def create_wordcloud(word_freq_dict, file_name_string):

    wordcloud = WordCloud(width = 3000, height = 2000,
                      random_state=1, background_color='black', colormap='Set2',
                      collocations=False, stopwords = STOPWORDS).generate_from_frequencies(word_freq_dict)
    # Plot and save file
    wordcloud.to_file(f'images/{file_name_string}.png')
    # plot_cloud(wordcloud)

if __name__ == "__main__":

    #get dataframes
    df_gd, df_pf, df_phish, df_beatles, df = get_data()

    #get X and y
    X, y = get_X_y(df)

    #get training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #filter X_train, tokenize words for each document
    # X_train_filtered = filter_data(X_train)
    X_train_tokens = [filter_data_text(doc) for doc in X_train]
    X_test_tokens = [filter_data_text(doc) for doc in X_test]

    #count vectorizer: tf
    #use filter_data_text function as tokenizer
    #fit to X_train
    tf = CountVectorizer(tokenizer=filter_data_text, max_features=5000)
    document_tf_matrix = tf.fit_transform(X_train).todense()
    #feature names
    tf_feature_names = tf.get_feature_names()
    #vocabulary dictionary
    tf_vocabulary_dict = tf.vocabulary_
    #df
    tf_df = pd.DataFrame(document_tf_matrix, columns=tf.get_feature_names())

    #tfidf vectorizer: tf-idf
    #use filter_data_text function as tokenizer
    #fit to X_train
    tfidf = TfidfVectorizer(tokenizer=filter_data_text, max_features=5000)
    document_tfidf_matrix = tfidf.fit_transform(X_train).todense()
    #feature names
    tfidf_feature_names = tfidf.get_feature_names()
    #vocabulary dictionary
    tfidf_vocabulary_dict = tfidf.vocabulary_
    #df
    tf_df = pd.DataFrame(document_tf_matrix, columns=tf.get_feature_names())

    #get top words across all documents/subreddits and plot
    top_words = get_top_n_words(X_train, n=100)
    plot_top_words(top_words, len(X_train), 'Word', 'Word Frequency Per Post', 'Top 15 Words Across Subreddits', 'all_word_freq_graph')

    #get top words for each individual subreddit and plot
    top_gd_words = get_top_n_words(X_train[y_train=='gratefuldead'], n=100)
    top_phish_words = get_top_n_words(X_train[y_train=='phish'], n=100)
    top_pf_words = get_top_n_words(X_train[y_train=='pinkfloyd'], n=100)
    top_beatles_words = get_top_n_words(X_train[y_train=='beatles'], n=100)
    #plot top words for each individual subreddit
    plot_top_words(top_gd_words, len(X_train[y_train=='gratefuldead']), 'Word', 'Word Frequencey Per Post', 'Top 15 Words In Grateful Dead Subreddit', 'gd_word_freq_graph')
    plot_top_words(top_phish_words, len(X_train[y_train=='phish']), 'Word', 'Word Frequency Per Post', 'Top 15 Words In Phish Subreddit', 'phish_word_freq_graph')
    plot_top_words(top_pf_words, len(X_train[y_train=='pinkfloyd']), 'Word', 'Word Frequency Per Post', 'Top 15 Words In Pink Floyd Subreddit', 'pf_word_freq_graph')
    plot_top_words(top_beatles_words, len(X_train[y_train=='beatles']), 'Word', 'Word Frequency Per Post', 'Top 15 Words In Beatles Subreddit', 'beatles_word_freq_graph')

    #create wordclouds
    create_wordcloud(dict(top_words), 'all_wordcloud' )
    create_wordcloud(dict(top_gd_words), 'gd_wordcloud')
    create_wordcloud(dict(top_phish_words), 'phish_wordcloud')
    create_wordcloud(dict(top_pf_words), 'pf_wordcloud')
    create_wordcloud(dict(top_beatles_words), 'beatles_wordcloud')