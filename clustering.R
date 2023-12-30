setwd("/Users/namgyulee/Personal_Project/News_Article_Classification/")

library(cluster)
library(factoextra)
library(fpc)

df <- read.csv("normalized.csv", header=T, sep=",")
df <- (df[, c("Title.Sentence.Vector", "Description.Sentence.Vector", "Author.Sentence.Vector")])

df.scaled <- scale(df)

k <- kmeans(df, centers=5) 
k.scaled <- kmeans(df.scaled, centers=5)

fviz_cluster(k, data=df, main="Unscaled clusters") 
fviz_cluster(k.scaled, data=df.scaled, main="Scaled clusters") 

# As the plot shows, the data does not need to be scaled.

# Using WSS and Silhouette method, deiced ideal number of clusters.
# wss method
fviz_nbclust(df, kmeans, method = "wss", k.max = 15) 

# silhouette method
fviz_nbclust(df, kmeans, method = "silhouette", k.max = 15)

# Visualize Clustering with center 2
fviz_cluster(kmeans(df, centers=5, nstart=25), data=df)
