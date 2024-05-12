# Important Notes Week 1

## Clustering In Machine Learning

Clustering is an unsupervised machine learning technique that involves grouping a set of data points into clusters based on their similarity. The goal of clustering is to identify patterns or structures within the data, where data points within the same cluster are more similar to each other than to those in other clusters.

In clustering, the algorithm automatically discovers the inherent groupings or clusters present in the data without any prior knowledge of the labels or categories. This is in contrast to supervised learning techniques, where the data is labeled, and the algorithm learns to map input data to the corresponding labels or categories.

There are various clustering algorithms, each with its own approach and assumptions.<br>
Some of the commonly used clustering algorithms include:

1. **K-Means Clustering:** This algorithm partitions the data into K clusters, where each data point belongs to the cluster with the nearest mean (centroid). The algorithm iteratively updates the centroids and reassigns data points to the closest centroid until convergence.
2. **Hierarchical Clustering:** This algorithm builds a hierarchy of clusters, either by merging smaller clusters into larger ones (agglomerative) or by dividing larger clusters into smaller ones (divisive). The result is typically visualized as a dendrogram.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** This algorithm groups together data points that are close to each other based on density reachability. It can identify clusters of arbitrary shape and is robust to noise and outliers.
4. **Mean-Shift Clustering:** This algorithm shifts data points towards the mean of their neighboring points, and clusters are formed around the densest regions in the data.
