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

### Intuition Behind K-Means Clustering

The intuition behind K-means clustering revolves around the concept of partitioning data into K distinct clusters based on similarity. The goal is to minimize the variance (or distance) within each cluster while maximizing the variance (or distance) between different clusters.

Here's a step-by-step breakdown of the intuition behind K-means clustering:

#### 1. Initialization:

- Start by randomly selecting K points in the feature space as initial cluster centers. These points are often referred to as centroids.

#### 2. Assignment of Data Points:

- For each data point, calculate its distance (commonly using Euclidean distance) to each centroid.
- Assign the data point to the nearest centroid, thereby forming K clusters.

#### 3. Update Centroids:

- After assigning all data points to clusters, calculate the new centroid (mean) of each cluster. This centroid represents the "center" of the cluster.
- Move each centroid to the calculated mean of its respective cluster.

#### 4. Iterative Process:

- Repeat the assignment of data points to clusters based on the updated centroids.
- Recalculate the centroids based on the new cluster assignments.
- Continue this process iteratively until convergence. Convergence is typically achieved when the centroids no longer change significantly between iterations, or when a specified number of iterations is reached.
