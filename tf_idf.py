from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import pandas as pd

dataframe = pd.read_csv("train_500k.csv", nrows=20000)
dt = dataframe[:15000].copy()

dt_input_values = dt[["Title", "Body"]].copy()
dt_input_values["Title_count"] = dt_input_values["Title"].apply(lambda x: len(str(x).split()))
dt_input_values["Body_count"] = dt_input_values["Body"].apply(lambda x: len(str(x).split()))


def cleanse_text(text):
    text = str(text).lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("</?.*?>", " <> ", text)
    text = re.sub(r"(\d|\W)+", " ", text)
    return text


dt["text"] = dt["Title"] + dt["Body"]
dt["text"] = dt["text"].apply(cleanse_text)

wordcloud = WordCloud(background_color="white", max_words=100, max_font_size=50, random_state=42).generate(
    str(dt["text"])
)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


def get_stop_words(stop_file_path):
    with open(stop_file_path, "r", encoding="utf-8") as f:
        return frozenset(m.strip() for m in f.readlines())


stopwords = get_stop_words("stopwords.txt")
docs = dt["text"].tolist()

cv = CountVectorizer(max_df=0.85, stop_words=stopwords, max_features=10000)
word_count_vector = cv.fit_transform(docs)


def get_top_ngrams(corpus, ngram_range=(1, 1), n=20):
    vec = CountVectorizer(ngram_range=ngram_range, max_features=2000).fit(corpus)
    bag = vec.transform(corpus)
    sum_words = bag.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]


for ngram_range, col_name, title in [
    ((1, 1), "Word", "Top Uni-grams"),
    ((2, 2), "Bi-gram", "Top Bi-grams"),
    ((3, 3), "Tri-gram", "Top Tri-grams"),
]:
    top = get_top_ngrams(dt["text"], ngram_range)
    top_df = pd.DataFrame(top, columns=[col_name, "Freq"])
    sns.set(rc={"figure.figsize": (13, 8)})
    ax = sns.barplot(x=col_name, y="Freq", data=top_df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

df_test = dataframe[15000:].copy()
df_test["text"] = df_test["Title"] + df_test["Body"]
df_test["text"] = df_test["text"].apply(cleanse_text)
docs_test = df_test["text"].tolist()

# get_feature_names() removed in sklearn 1.2 — use get_feature_names_out()
feature_names = cv.get_feature_names_out()


def sort_coo(coo_matrix):
    return sorted(zip(coo_matrix.col, coo_matrix.data), key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    return {feature_names[idx]: round(score, 3) for idx, score in sorted_items[:topn]}


def get_keywords(idx):
    tf_idf_vector = tfidf_transformer.transform(cv.transform([docs_test[idx]]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    return extract_topn_from_vector(feature_names, sorted_items, 10)


df_test = df_test.reset_index(drop=True)
df_test["Keywords"] = [get_keywords(i) for i in range(len(df_test))]