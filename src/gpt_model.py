#from transformers import OpenAIGPTTokenizer
#import openai
#import tensorflow as tf
import numpy as np
import pandas as pd
from function import get_API_key, gpt_Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from openai import OpenAI
#import os
#import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#import string
#import re
#from ast import literal_eval
import json

df = pd.read_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/news_data.csv')
df['Description'] = df['Description'].fillna('')

title = df['Title']
des = df['Description']
author = df['Author']


# Train GPT using 
key_file = '/Users/namgyulee/Personal_Project/News_Article_Classification/api-key.txt'
api_key_instance = get_API_key(key_file, 5)
API_key = api_key_instance.get_api_key(5)

client = OpenAI(
    api_key = API_key
    )

"""test = [['hahahah'], ['Apple'], ['Banana'], ['club'], ['start']]
tesst = []
for i in range(len(test)):
    response = client.embeddings.create(
        input=test[i],
        model="text-embedding-ada-002",
        encoding_format=float
    )
    tesst.append(i, response.data[0].embedding)
print(tesst)
"""
#def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
#    return client.embeddings.create(input=[text], model=model)["data"][0]["embedding"]

#def get_embedding(text, model="text-embedding-ada-002"):
#   return client.embeddings.create(input = [text], model=model).data[0].embedding
"""embeddings = []
for chunk in des:
    response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
    embeddings.append(response['data'][0]['embedding'])

final_embedding = np.concatenate(embeddings)

print(len(final_embedding))"""


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text],
        encoding_format=float
    )
    embedding = response.data[0].embedding.tolist()

    return embedding


title["embedding"] = get_embedding(title)
des["embedding"] = get_embedding(des)


#des_embedding = get_embedding(des.tolist())  # Convert Series to list

# Print the first 3 embeddings
#print(des_embedding[:3])

#title['Title ada embedding'] = []
#des['Description ada embedding'] = []
#author['Author ada embedding'] = []


#for index, row in title.iterrows():
#    title.loc[index, 'Title ada embedding'] = get_embedding(row['Title'])
#des['Description ada embedding'] = des.apply(lambda x: get_embedding(x))
#author['Author ada embedding'] = author.apply(lambda x: get_embedding(x))
"""
df_output = pd.DataFrame({
    'Original Title Text': list(title),
    'Original Description Text': list(des),
    'Original Author Text': list(author)
})

# Apply preprocessing and embedding to each text column
df_output['Title ada embedding'] = title.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
df_output['Description ada embedding'] = des.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
df_output['Author ada embedding'] = author.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
df_output.to_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/gpt_model.csv', index=True) 
"""
"""
# Perform k-means Clustering
k_mean_df = pd.read_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/gpt_model_test.csv')
k_mean_df = k_mean_df.loc[:, ['Title ada embedding', 'Description ada embedding', 'Author ada embedding']]
k_mean_df = k_mean_df.apply(literal_eval).apply(np.array)
matrix = np.vstack(k_mean_df.values)

n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
kmeans.fit(matrix)
labels = kmeans.labels_
k_mean_df["Cluster"] = labels

k_mean_df.groupby("Cluster").Score.mean().sort_values()


kmeans = KMeans(n_clusters = 5)
kmeans.fit(combined_embedding.numpy())

# 클러스터의 중심을 시각화하기 위해 차원 축소
pca = PCA(n_components = 2)
reduced_features = pca.fit_transform(combined_embedding.numpy())
reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

# 클러스터링 결과 시각화
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='red')
plt.title('K-Means Clustering of Text Embeddings')
plt.show()

filename = "GPT_model_kmeans_visualization.png"
plt.savefig(filename)
print(f"Plot saved successfully as '{filename}'")
"""
