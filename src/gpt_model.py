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
import json
import pickle
#from utils.embeddings_utils import get_embedding
#from openai import EmbeddingOpenAI

df = pd.read_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/news_data.csv')
df['Description'] = df['Description'].fillna('')

title = df['Title']
des = df['Description']
author = df['Author']

title = title.tolist()
des = des.tolist()
author = author.tolist()

title = json.dumps(title)
des = json.dumps(des)
author = json.dumps(author)

# Train GPT using 
key_file = '/Users/namgyulee/Personal_Project/News_Article_Classification/api-key.txt'
api_key_instance = get_API_key(key_file, 5)
API_key = api_key_instance.get_api_key(5)

client = OpenAI(
    api_key = API_key
    )

def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding

def get_embeddings(text_list, model="text-embedding-ada-002"):
    embeddings = []
    for text_chunk in text_list:
        embedding = get_embedding(text_chunk, model=model)
        embeddings.append(embedding)
    return embeddings

chunk_size = 8192

title_chunks = [title[i:i+chunk_size] for i in range(0, len(title), chunk_size)]
des_chunks = [des[i:i+chunk_size] for i in range(0, len(des), chunk_size)]
author_chunks = [author[i:i+chunk_size] for i in range(0, len(author), chunk_size)]

# Get embeddings for each chunk
title_embeddings = get_embeddings(title_chunks)
des_embeddings = get_embeddings(des_chunks)
author_embeddings = get_embeddings(author_chunks)

# Concatenate embeddings into lists or arrays
embedded_title = [emb for sublist in title_embeddings for emb in sublist]
embedded_des = [emb for sublist in des_embeddings for emb in sublist]
embedded_author = [emb for sublist in author_embeddings for emb in sublist]
    

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
