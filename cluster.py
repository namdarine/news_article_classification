import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

file_path = '/Users/namgyulee/Personal_Project/News_Article_Classification/train_sentence_vectors.csv'
df = pd.read_csv(file_path)

title_vector = df['Title Sentence Vector'].values.reshape(-1, 1)
des_vector = df['Description Sentence Vector'].values.reshape(-1, 1)
author_vector = df['Author Sentence Vector'].values.reshape(-1, 1)


scaler = RobustScaler()

scaled_title = scaler.fit_transform(title_vector)
scaled_des = scaler.fit_transform(des_vector)
scaled_author = scaler.fit_transform(author_vector)

scaled_vector = np.concatenate([scaled_title, scaled_des, scaled_author], axis=1)


# Use the elbow method to visualize the intertia for different values of K
inertia = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_vector)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Silhouette Score
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_vector)
    silhouette = silhouette_score(scaled_vector, kmeans.labels_)
    silhouette_scores.append(silhouette)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Clustering data
optimal_k = 6

# Perform K-Means clustering with the optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_vector)

# Add cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.scatter(df['Title Sentence Vector'], df['Description Sentence Vector'], c=df['Cluster'], cmap='viridis')
plt.title('Clustered Data')
plt.xlabel('Title Sentence Vector')
plt.ylabel('Description Sentence Vector')
plt.show()

# Evaluate clustering performance using silhouette score
silhouette = silhouette_score(scaled_vector, kmeans.labels_)
print(f'Silhouette Score: {silhouette}')

# Seems it is hard to divide more than one clusters.