import numpy as np
import pandas as pd
from function import get_API_key
import matplotlib.pyplot as plt
import google.generativeai as genai
from tqdm.auto import tqdm
from google.api_core import retry
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE

key_file = '/Users/namgyulee/Personal_Project/News_Article_Classification/api-key.txt'
api_key_instance = get_API_key(key_file, 8)
API_key = api_key_instance.get_api_key(8)

genai.configure(api_key=API_key)

df = pd.read_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/gpt_data.csv')
title = df['Normalized Title']
des = df['Normalized Description']
author = df['Normalized Author']

tqdm.pandas()

# Train model
def make_embed_text_fn(model):

  @retry.Retry(timeout=300.0)
  def embed_fn(text: str) -> list[float]:
    # Set the task_type to CLUSTERING.
    embedding = genai.embed_content(model=model,
                                    content=text,
                                    task_type="clustering")
    return embedding["embedding"]

  return embed_fn

def create_embeddings(df):
  model = 'models/embedding-001'
  embedding = df.progress_apply(make_embed_text_fn(model))
  return embedding

embedded_title = create_embeddings(title)
embedded_des = create_embeddings(des)
embedded_author = create_embeddings(author)


# Dimensionality reduction
X_title = np.array(embedded_title.to_list(), dtype=np.float32)
X_des = np.array(embedded_des.to_list(), dtype=np.float32)
X_author = np.array(embedded_author.to_list(), dtype=np.float32)

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
news_data['Genemi Cluster'] = kmeans_model2.labels_
news_data.to_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/clustered_news_data.csv', index=False)

# Due to Title and Description Silhouette score is high
# Visualize Author and Description factors
cluster_centers = kmeans_model2.cluster_centers_
num_clusters = 5
colors = ['red', 'green', 'blue', 'purple', 'orange']  # Adjust for different n_clusters

# Plot each data point with its assigned cluster color
for i, label in enumerate(labels2):
    plt.scatter(tsne_title, tsne_des, c=colors[label])

# Plot cluster centers with larger markers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=250, marker='*', c='black', label='Cluster Centers')

# Customize plot elements
plt.title('K-Means Clustering with {} Clusters'.format(num_clusters))
plt.xlabel('Title')
plt.ylabel('Description')
plt.legend()

plt.show()

filename = "/Genemi_kmeans_visualization.png"
plt.savefig(filename)
print(f"Plot saved successfully as '{filename}'")






