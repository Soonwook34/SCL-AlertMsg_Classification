import gensim
from eunjeon import Mecab
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import os.path
import pickle
import warnings

USE_PREV_MODEL = True
warnings.filterwarnings("ignore")
mecab = Mecab()


def tokenize_sentense(text):
    noun_list = mecab.nouns(text)
    final_list = []
    for noun in noun_list:
        if len(noun) >= 2:
            final_list.append(noun)
    return final_list


def get_sentence_mean_vector(morphs):
    vector = []
    for i in morphs:
        try:
            vector.append(w2v_model.wv[i])
        except KeyError as e:
            pass
    try:
        return np.mean(vector, axis=0)
    except IndexError as e:
        pass


dataset = pd.read_csv("../csv/AM_210329_COVID7.csv", encoding="utf-8")
message_list = dataset["MESSAGE"].values.tolist()
category_list = dataset["CATEGORY"].values.tolist()
tokenized_data = []

if USE_PREV_MODEL:
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec.bin")
    print("Word2Vec 모델 로드 완료")
else:
    for message in message_list:
        message_split = message.split("]")
        new_message = message_split[1]
        if len(message_split) > 2:
            for message_piece in message_split[2:]:
                new_message = f"{new_message}]{message_piece}"
        tokenized_data.append(tokenize_sentense(new_message))
    print("Word2Vec 모델 생성 중...")
    w2v_model = Word2Vec(sentences=tokenized_data, size=100, window=5, min_count=10, iter=20, sg=0, workers=8)
    w2v_model.wv.save_word2vec_format("word2vec.bin")
    print("Word2Vec 모델 생성 및 저장 완료")

# model_result = w2v_model.wv.most_similar("발생")
# print(model_result)

dataset["mecab_message"] = dataset["MESSAGE"].map(tokenize_sentense)
print("재난문자 데이터 명사 토큰화 완료")
dataset["mecab_len_message"] = dataset["mecab_message"].map(len)
dataset["wv"] = dataset["mecab_message"].map(get_sentence_mean_vector)
print("재난문자 데이터 벡터화 완료")
# print(dataset.head())

num_clusters = 5
word_vectors = dataset.wv.to_list()

kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)
dataset["CLASSIFICATION"] = idx

print(dataset.CATEGORY.value_counts())
print(dataset.CLASSIFICATION.value_counts())
print(dataset.head())

# # 클러스터 시각화
# X = dataset["wv"].to_list()
# y = dataset["CLASSIFICATION"].to_list()
#
# tsne_filepath = "tsne3000.pkl"
#
# # File Cache
# if not os.path.exists(tsne_filepath):
#     tsne = TSNE(random_state=42)
#     tsne_points = tsne.fit_transform(X)
#     with open(tsne_filepath, 'wb+') as f:
#         pickle.dump(tsne_points, f)
# else:  # Cache Hits!
#     with open(tsne_filepath, 'rb') as f:
#         tsne_points = pickle.load(f)
#
# tsne_df = pd.DataFrame(tsne_points, index=range(len(X)), columns=["x_coord", "y_coord"])
# tsne_df["user_bio"] = dataset["MESSAGE"].to_list()
# tsne_df["cluster_no"] = y
# print(tsne_df.sample(3))
