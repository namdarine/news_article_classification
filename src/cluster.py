# -*- coding: utf-8 -*-
"""cluster.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rlIOiMsY4AAF8sOJDznqfCf0YZv58IEI
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

df = pd.read_csv("/Users/namgyulee/Personal_Project/News_Article_Classification/Data/normalized.csv")
df.vector = df.loc[:, ['Title Sentence Vector', 'Description Sentence Vector', 'Author Sentence Vector']]

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(df.vector)

# Evaluate the clustering performance using metrics
silhouette_score = metrics.silhouette_score(df.vector, kmeans.labels_)
print(f'Everything Silhouette Score: {silhouette_score}')

# Divide the combinations to compare
# Title and Description
df.vector1 = df.loc[:, ['Title Sentence Vector', 'Description Sentence Vector']]

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(df.vector1)

# Evaluate the clustering performance using metrics
silhouette_score1 = metrics.silhouette_score(df.vector1, kmeans.labels_)
print(f'Title - Description Silhouette Score: {silhouette_score1}')

# Title and Author
df.vector2 = df.loc[:, ['Title Sentence Vector', 'Author Sentence Vector']]

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(df.vector2)

silhouette_score2 = metrics.silhouette_score(df.vector2, kmeans.labels_)
print(f'Title - Author Silhouette Score: {silhouette_score2}')

# Description and Author
df.vector3 = df.loc[:, ['Description Sentence Vector', 'Author Sentence Vector']]

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(df.vector3)

silhouette_score3 = metrics.silhouette_score(df.vector3, kmeans.labels_)
print(f'Description - Author Silhouette Score: {silhouette_score3}')

# Extract features for visualization,
# 'Title Sentence Vector' and 'Description Senetence Vector'
# Due to higher Silhouette Score
features_to_visualize = ['Title Sentence Vector', 'Description Sentence Vector']
X_visualize = df[features_to_visualize]

# Assign colors to clusters for better differentiation:
cmap = plt.cm.get_cmap('viridis')  # Use a visually appealing colormap
colors = cmap(kmeans.labels_.astype(float) / num_clusters)

# Create scatter plot with clear labeling:
plt.figure(figsize=(8, 6))
plt.scatter(X_visualize.iloc[:, 0], X_visualize.iloc[:, 1], c=colors, s=50)

# Add informative plot title and axis labels:
plt.title('K-means Clustering (5 Clusters)', fontsize=16)
plt.xlabel(features_to_visualize[0], fontsize=14)
plt.ylabel(features_to_visualize[1], fontsize=14)

# Add legend for clarity:
plt.legend(fontsize=12)
plt.show()

filename = "kmeans_visualization.png"
plt.savefig(filename)
print(f"Plot saved successfully as '{filename}'")

df['Cluster'] = kmeans.labels_
cluster_counts = df['Cluster'].value_counts()
print("Cluster Counts:")
print(cluster_counts)

df2 = pd.read_csv("/news_data.csv")
df2['Cluster'] = kmeans.labels_
filename = "clustered_data.csv"
df.to_csv(filename, index=False)
df2.to_csv("clustered_news_data.csv", index=False)

print(f"DataFrame saved successfully as '{filename}'")

for cluster_num in range(5):
    cluster_articles = df[df['Cluster'] == cluster_num]['Original Title Text']
    print(f"\nCluster {cluster_num} Articles:")
    print(cluster_articles.head(5))
    
    
# Predict with new dataset 
new_df = pd.read_csv("/new_normalized.csv")
new_df.vector = new_df.loc[:, ['Title Sentence Vector', 'Description Sentence Vector']]
kmeans.fit(new_df.vector)
predict = kmeans.predict(new_df.vector)
print(predict)

import seaborn as sns

new_data_clusters = predict
sns.scatterplot(x=new_df.vector['Title Sentence Vector'], y=new_df.vector['Description Sentence Vector'], hue=new_data_clusters, palette='viridis')
plt.title('Cluster Assignments for New Data')
plt.show()

filename = "new_kmeans_visualization.png"
plt.savefig(filename)
print(f"Plot saved successfully as '{filename}'")

new_df2 = pd.read_csv("/new_data.csv")

new_df2['k-means Cluster'] = predict
filename = "clustered_new_data.csv"
df.to_csv(filename, index=False)
new_df2.to_csv("clustered_new_news_data.csv", index=False)





