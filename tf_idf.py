# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:47:32 2019

@author: aliag
"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS,  WordCloud, ImageColorGenerator
import re
import pandas as pd


dataframe = pd.read_csv('train_500k.csv', nrows=20000)
dt = dataframe[0:15000]

dt_input_values = dt[['Title', 'Body']]
dt_input_values.head()

# Counting the number of words
dt_input_values['Title_count'] = dt_input_values['Title'].apply(
    lambda x: len(str(x).split(" ")))
dt_input_values['Body_count'] = dt_input_values['Body'].apply(
    lambda x: len(str(x).split(" ")))
dt_input_values.describe()

# Observing most common Words
common_words_title = pd.Series(
    ' '.join(dt_input_values['Title']).split()).value_counts()[:20]  # Top 20
common_words_body = pd.Series(
    ' '.join(dt_input_values['Body']).split()).value_counts()[:20]

# Observing most un-common Words
uncommon_words_title = pd.Series(' '.join(
    dt_input_values['Title']).split()).value_counts()[-20:]  # Top 20 from bottom
uncommon_words_body = pd.Series(
    ' '.join(dt_input_values['Body']).split()).value_counts()[-20:]



def cleanse_text(text):

    # lowercase
    text = text.lower()
    
    # removing non-alphabet characters and replace by space
    text= re.sub('[^a-zA-Z]', ' ', text)

    # remove tags
    text = re.sub("</?.*?>", " <> ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    return text


dt['text'] = dt['Title'] + dt['Body']
dt['text'] = dt['text'].apply(lambda x: cleanse_text(x))


wordcloud = WordCloud(
    background_color='white',
    max_words=100,
    max_font_size=50,
    random_state=42).generate(str(dt['text']))

fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

def get_stop_words(stop_file_path):

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


# load a set of stop words
stopwords = get_stop_words("stopwords.txt")

# get the text column
docs = dt['text'].tolist()

# create a vocabulary of words,
# ignore words that appear in 85% of documents,
# eliminate stop words
cv = CountVectorizer(max_df=0.85, stop_words=stopwords, max_features=10000)
word_count_vector = cv.fit_transform(docs)
word_count_vector.shape

list(cv.vocabulary_.keys())[:10]

# Most frequently occuring words
def get_top_one_word(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]


# Convert most freq words to dataframe for plotting bar plot
top_words = get_top_one_word(dt['text'], n=20)

top_df = pd.DataFrame(top_words)

top_df.columns = ["Word", "Freq"]

# Barplot of most freq words
sns.set(rc={'figure.figsize': (13, 8)})
g1 = sns.barplot(x="Word", y="Freq", data=top_df)
g1.set_xticklabels(g1.get_xticklabels(), rotation=30)


# Most frequently occuring Bi-grams
def get_top_two_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2, 2),
                           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]

top2_words = get_top_two_words(dt['text'], n=20)

top2_df = pd.DataFrame(top2_words)

top2_df.columns = ["Bi-gram", "Freq"]


print(top2_df)

# Barplot of most freq Bi-grams for title
sns.set(rc={'figure.figsize': (13, 8)})
h = sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)


# Most frequently occuring Tri-grams
def get_top_three_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3, 3),
                           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]


top3_words = get_top_three_words(dt['text'], n=20)

top3_df = pd.DataFrame(top3_words)

top3_df.columns = ["Tri-gram", "Freq"]


print(top3_df)

# Barplot of most freq Tri-grams for
sns.set(rc={'figure.figsize': (13, 8)})
j = sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(j.get_xticklabels(), rotation=45)


tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

# read test docs into a dataframe and concatenate title and body
df_test = dataframe[15000:]
df_test['text'] = df_test['Title'] + df_test['Body']
df_test['text'] = df_test['text'].apply(lambda x: cleanse_text(x))

# get test docs into a list
docs_test = df_test['text'].tolist()

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
  

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


feature_names = cv.get_feature_names()

# put the common code into several methods
def get_keywords(idx):

    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([docs_test[idx]]))

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    return keywords


#test for all in test set
df_test.reset_index(inplace = True)
df_test['Keywords'] = ""
for idx in range (len(df_test)):
    keywords = get_keywords(idx)
    df_test.at[idx, 'Keywords'] = keywords
    
