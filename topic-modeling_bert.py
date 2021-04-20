from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataset = pd.read_csv("csv/AM_210329_COVID7.csv", encoding="utf-8")
data = dataset["MESSAGE"].values.tolist()
category = dataset["CATEGORY"]
# data = fetch_20newsgroups(subset='all')['data']

print(len(data))

# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
embeddings = model.encode(data, show_progress_bar=True)

print(len(embeddings[0]), embeddings[:2])

print("Reduce Dimension Using UMAP...")
umap_embeddings = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine').fit_transform(embeddings)

print("Clustering Using KMeans...")
cluster = KMeans(n_clusters=5).fit_predict(umap_embeddings)
# print("Clustering Using HDBSCAN...")
# cluster = hdbscan.HDBSCAN(min_cluster_size=30, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)


print("Save pyplot Image...")
# Prepare data
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
# result['labels'] = cluster.labels_
result['labels'] = cluster
# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.colorbar()
plt.show()
plt.savefig("BERTopic.png", dpi=200)

docs_df = pd.DataFrame(data, columns=["Doc"])
# docs_df['Topic'] = cluster.labels_
docs_df['Topic'] = cluster
docs_df['Doc_ID'] = range(len(docs_df))
docs_df['CATEGORY'] = category
docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '을', '으로', '자', '에', '와', '한', '하다', '바랍니다', '습니다', '하십시오']
    count = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


print("c-TF-IDF...")
tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                   enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                   .Doc
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes


top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(docs_df)
print(len(topic_sizes), topic_sizes.head(10))

print(f"Reducing {len(topic_sizes) - 6} Dimensions...")
for i in range(len(topic_sizes) - 6):
    # Calculate cosine similarity
    similarities = cosine_similarity(tf_idf.T)
    np.fill_diagonal(similarities, 0)

    # Extract label to merge into and from where
    topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
    topic_to_merge = topic_sizes.iloc[-1].Topic
    topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

    # Adjust topics
    docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
    old_topics = docs_df.sort_values("Topic").Topic.unique()
    map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
    docs_df.Topic = docs_df.Topic.map(map_topics)
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

    # Calculate new topic words
    m = len(data)
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)

topic_sizes = extract_topic_sizes(docs_df)
print(len(topic_sizes), topic_sizes.head(10))
print(docs_df.head())

con_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

for i in range(0, len(docs_df)):
    con_matrix[docs_df["CATEGORY"][i]][docs_df["Topic"][i]] += 1

for i in range(0, 5):
    print(con_matrix[i])


# ALL
# con_matrix = [[4764, 388, 627, 4168, 53],
#               [5, 7578, 101, 236, 2],
#               [1, 117, 2781, 44, 229],
#               [1, 81, 2221, 97, 57],
#               [3, 790, 492, 39, 2019]]
# MAIN
# con_matrix = [[8356, 774, 870],
#               [96, 7729, 97],
#               [70, 1116, 7796]]
# SUB
con_matrix = [[2723, 272, 177],
              [2180, 225, 52],
              [418, 1037, 1888]]
accuracy = 0
for i in range(0, 3):
    accuracy += con_matrix[i][i]
accuracy /= len(dataset)
precision = 0
recall = 0
f1_score = 0
for i in range(0, 3):
    print(con_matrix[i])
    TP = con_matrix[i][i]
    FP = 0
    FN = 0
    for j in range(0, 3):
        if j != i:
            FP += con_matrix[j][i]
            FN += con_matrix[i][j]
    precision += (TP / (TP + FP))
    recall += (TP / (TP + FN))
precision /= 3
recall /= 3
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy : {accuracy:.6f}\nF1-Score : {f1_score:.6f}(Precision : {precision:.6f} Recall : {recall:.6f})")
