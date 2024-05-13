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

#### 5. Objective Function:

- The main objective of K-means clustering is to minimize the within-cluster sum of squares (WCSS), which is the sum of squared distances between each data point and its assigned centroid. Mathematically, this can be expressed as:
  $$\text{Minimize WCSS} = \sum_{i=1}^{K} \sum_{x \in C_i} \left\lVert x - \mu_i \right\rVert^2$$
  **$Where:$**<br>

  - $K$ is the number of clusters
  - $C_i$ is the set of data points belonging to the $i^{th}$ cluster
  - $x$ is a data point
  - $\mu_i$ is the centroid (mean) of the $i^{th}$ cluster
  - $\left\lVert x - \mu_i \right\rVert^2$ is the squared Euclidean distance between the data point $x$ and the centroid $\mu_i$ of the cluster it belongs to

- Here's a breakdown of the LaTeX expression:
  - The outer summation $\sum_{i=1}^{K}$ iterates over all $K$ clusters.
  - The inner summation $\sum_{x \in C_i}$ iterates over all data points $x$ belonging to the $i^{th}$ cluster $C_i$.
  - $\left\lVert x - \mu_i \right\rVert^2$ calculates the squared Euclidean distance between the data point $x$ and the centroid $\mu_i$ of the cluster it belongs to.
  - The squared distances are summed over all data points in all clusters, resulting in the total within-cluster sum of squares (WCSS).

#### 6. Convergence:

- K-means typically converges to a local minimum of the objective function. However, the final clustering result can be sensitive to the initial choice of centroids.

### Basic Implementation of The K-Means Clustering Algorithm in Python:

```Python
import numpy as np

def euclidean_distance(x1, x2):
    """
    Calculates the Euclidean distance between two data points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initialize_centroids(self, X):
        """
        Initializes the centroids randomly from the data points.
        """
        np.random.seed(self.random_state)
        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        return centroids

    def compute_centroids(self, X, labels):
        """
        Computes the new centroids based on the mean of the data points in each cluster.
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k] = np.mean(X[labels == k], axis=0)
        return centroids

    def fit(self, X):
        """
        Fits the K-Means algorithm to the data.
        """
        centroids = self.initialize_centroids(X)
        prev_centroids = None
        labels = np.zeros(X.shape[0])
        iter_count = 0

        while np.not_equal(centroids, prev_centroids).any() and iter_count < self.max_iter:
            iter_count += 1
            prev_centroids = centroids.copy()

            # Assign data points to the closest centroid
            for i, x in enumerate(X):
                distances = [euclidean_distance(x, centroid) for centroid in centroids]
                labels[i] = np.argmin(distances)

            # Update centroids
            centroids = self.compute_centroids(X, labels)

        self.centroids = centroids
        self.labels = labels

    def predict(self, X):
        """
        Predicts the cluster labels for new data points.
        """
        labels = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            distances = [euclidean_distance(x, centroid) for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        return labels
```

#### Here's how the code works:

1. The `euclidean_distance` function calculates the Euclidean distance between two data points.
2. The `KMeans` class has methods for initializing centroids, computing new centroids, fitting the model to data, and predicting cluster labels for new data.
3. The `initialize_centroids` method randomly selects `n_clusters` data points as the initial centroids.
4. The `compute_centroids` method calculates the new centroids by taking the mean of the data points in each cluster.
5. The `fit` method runs the K-Means algorithm by iteratively assigning data points to the closest centroid and updating the centroids until convergence or the maximum number of iterations is reached.
6. The `predict` method assigns new data points to the closest centroid based on the learned centroids.

To use this implementation, we can create an instance of the `KMeans` class and call the `fit` method with our data:

```Python
# Load or generate your data
X = ...  # shape (n_samples, n_features)

# Initialize the K-Means model
kmeans = KMeans(n_clusters=3, max_iter=300, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels for the input data
labels = kmeans.labels

# Get the learned centroids
centroids = kmeans.centroids

# Predict cluster labels for new data
new_data = ...  # shape (n_new_samples, n_features)
new_labels = kmeans.predict(new_data)
```

### Cost Function in K-Means Algorithm

In the K-Means clustering algorithm, the cost function, also known as the objective function or the distortion measure, is used to evaluate the quality of the clustering solution. The goal of the algorithm is to minimize this cost function, which quantifies the compactness or cohesion of the clusters.

The cost function for K-Means clustering is typically defined as the sum of squared distances between each data point and the centroid (mean) of the cluster it is assigned to.<br>
Mathematically, it can be expressed as:
