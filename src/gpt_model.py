import numpy as np
import pandas as pd
from function import get_API_key, gpt_Model
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.manifold import TSNE
import seaborn as sns

df = pd.read_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/gpt_data.csv')


title = df['Normalized Title']
des = df['Normalized Description']
author = df['Normalized Author']

# Train GPT using 
key_file = '/Users/namgyulee/Personal_Project/News_Article_Classification/api-key.txt'
api_key_instance = get_API_key(key_file, 5)
API_key = api_key_instance.get_api_key(5)

client = OpenAI(
    api_key = API_key
    )

gpt_model_instance = gpt_Model(client)

# Get embeddings for each chunk of 'Title', 'Description', and 'Author' columns
title_embeddings = gpt_model_instance.get_embeddings(title)
description_embeddings = gpt_model_instance.get_embeddings(des)
author_embeddings = gpt_model_instance.get_embeddings(author)

# Dimensionality reduction
X_title = np.array(title_embeddings, dtype=np.float32)
X_des = np.array(description_embeddings, dtype=np.float32)
X_author = np.array(author_embeddings, dtype=np.float32)

tsne = TSNE(random_state=0, n_iter=1000)
tsne_title = tsne.fit_transform(X_title)
tsne_des = tsne.fit_transform(X_des)
tsne_author = tsne.fit_transform(X_author)

combined1 = np.concatenate([tsne_title, tsne_des, tsne_author], axis=1)
combined2 = np.concatenate([tsne_title, tsne_des], axis=1)
combined3 = np.concatenate([tsne_title, tsne_author], axis=1)
combined4 = np.concatenate([tsne_author, tsne_des], axis=1)

# Divide the combinations to compare
# All three, title, description and author
kmeans_model1 = KMeans(n_clusters=5, random_state=1, n_init='auto').fit(combined1)
labels1 = kmeans_model1.fit_predict(combined1)

silhouette_score1 = metrics.silhouette_score(combined1, kmeans_model1.labels_)
print(f'All three Silhouette Score: {silhouette_score1}')

# Title and Description
kmeans_model2 = KMeans(n_clusters=5, random_state=1, n_init='auto').fit(combined2)
labels2 = kmeans_model2.fit_predict(combined2)

silhouette_score2 = metrics.silhouette_score(combined2, kmeans_model2.labels_)
print(f'Title - Description Silhouette Score: {silhouette_score2}')

# Title and Author
kmeans_model3 = KMeans(n_clusters=5, random_state=1, n_init='auto').fit(combined3)
labels3 = kmeans_model3.fit_predict(combined3)

silhouette_score3 = metrics.silhouette_score(combined3, kmeans_model3.labels_)
print(f'Title - Author Silhouette Score: {silhouette_score3}')

# Author and Description
kmeans_model4 = KMeans(n_clusters=5, random_state=1, n_init='auto').fit(combined4)
labels4 = kmeans_model4.fit_predict(combined4)

silhouette_score4 = metrics.silhouette_score(combined4, kmeans_model4.labels_)
print(f'Author - Description Silhouette Score: {silhouette_score4}')

news_data = pd.read_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/clustered_news_data.csv')
news_data['GPT Cluster'] = kmeans_model2.labels_
news_data.to_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/clustered_news_data.csv', index=False)

df2 = pd.read_csv("/Users/namgyulee/Personal_Project/News_Article_Classification/Data/clustered_news_data.csv")

for cluster_num in range(5):
    cluster_articles = df2[df2['GPT Cluster'] == cluster_num]['Title']
    print(f"\nCluster {cluster_num} Articles:")
    print(cluster_articles.head(5))
    
cluster_counts = df2['GPT Cluster'].value_counts()
print("Cluster Counts:")
print(cluster_counts)


# Visualization of clustering with 'Title' and 'Description'
# Due to high Silhouette score

cluster_centers = kmeans_model2.cluster_centers_
sns.scatterplot(x=combined2[:,0], y=combined2[:,1], hue=kmeans_model2.labels_, palette='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=150, c='red')
plt.title('GPT Model Clustering')
plt.xlabel('Title', fontsize=14)
plt.ylabel('Description', fontsize=14)
plt.show()

filename = "/Users/namgyulee/Personal_Project/News_Article_Classification/img/GPT_model_kmeans_visualization.png"
plt.savefig(filename)
print(f"Plot saved successfully as '{filename}'")


