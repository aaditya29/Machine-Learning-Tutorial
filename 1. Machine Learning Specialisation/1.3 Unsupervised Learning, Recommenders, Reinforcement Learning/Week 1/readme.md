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
$$J = \frac {1}{m}\sum_{i=1}^{K} \sum_{x \in C_i} \left\lVert x - \mu_i \right\rVert^2$$

$Where:$

- $K$ is the number of clusters
- $C_i$ is the set of data points belonging to the $i^{th}$ cluster
- $x$ is a data point
- $\mu_i$ is the centroid (mean) of the $i^{th}$ cluster
- $\left\lVert x - \mu_i \right\rVert^2$ is the squared Euclidean distance between the data point $x$ and the centroid $\mu_i$ of the cluster it belongs to

The term $\sum_{x \in C_i} \left\lVert x - \mu_i \right\rVert^2$ represents the sum of squared distances between all data points in the $i^{th}$ cluster and its centroid. By summing over all clusters, we get the total cost or distortion measure for the entire clustering solution.

The intuition behind this cost function is that we want to minimize the sum of squared distances between each data point and its assigned cluster centroid. This encourages the formation of compact and well-separated clusters, where data points within the same cluster are close to each other (minimizing within-cluster distances), and data points in different clusters are far apart (maximizing between-cluster distances).

During the K-Means algorithm, the cost function is minimized iteratively by:

1. Assigning each data point to the cluster with the nearest centroid.
2. Updating the centroids by taking the mean of the data points in each cluster.

These two steps are repeated until the cost function converges or a maximum number of iterations is reached.

The cost function serves as a quantitative measure to evaluate the quality of the clustering solution. A lower value of the cost function indicates a better clustering, where the data points are more tightly clustered around their respective centroids.

### Initialise K-means to Choose Number of Clusters

Initializing the number of clusters (K) in the K-Means algorithm is a crucial step, as it can significantly impact the clustering results. There are several techniques and heuristics that can be used to determine a suitable value for K.<br>
Here are some common approaches:

1. **Elbow Method:**

- This method involves running the K-Means algorithm for different values of K and calculating the cost function (sum of squared distances) for each value of K.
- The cost function is then plotted against the number of clusters (K).
- The optimal value of K is chosen at the "elbow" point, where the cost function starts to flatten out, indicating that adding more clusters does not significantly improve the clustering quality.
- This method helps strike a balance between minimizing the cost function and avoiding overfitting (too many clusters).

```Python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_ gives the WCSS

# Plotting the elbow method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

```

2. **Silhouette Analysis:**

- The silhouette score is a measure of how well a data point matches its assigned cluster compared to other clusters.
- The silhouette score ranges from -1 to 1, where a higher value indicates that the data point is well-matched to its cluster and poorly matched to neighboring clusters.
- The K-Means algorithm is run for different values of K, and the silhouette score is calculated for each value of K.
- The value of K with the highest average silhouette score is chosen as the optimal number of clusters.

```Python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plotting silhouette scores
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

```

3. **Gap Statistic:**

- The gap statistic compares the clustering results with a null reference distribution (e.g., a uniform distribution).
- The K-Means algorithm is run for different values of K, and the gap statistic is calculated for each value of K.
- The optimal value of K is chosen as the smallest value where the gap statistic is within a standard error of the first maximum gap value.
- This method helps determine the appropriate number of clusters by comparing the clustering results to a null reference.

4. **Domain Knowledge or Prior Information:**

- In some cases, domain knowledge or prior information about the data may provide insights into the expected number of clusters.
- For example, if the data represents customer segments, experts may have an idea of the potential number of segments based on market research or historical data.

5. **Visual Inspection:**

- If the data is low-dimensional (2D or 3D), visualizing the data using scatter plots or other visualization techniques can sometimes reveal natural clusters or patterns.
- While not a quantitative method, visual inspection can provide a qualitative assessment of the appropriate number of clusters.

## Anomaly Detection in Machine Learning

Anomaly detection in machine learning is the process of identifying data points, events, or observations that deviate significantly from the majority of the data. These deviations are often referred to as anomalies, outliers, novelties, or exceptions. Anomaly detection has applications across various domains including fraud detection, network security, manufacturing quality control, health monitoring, and more.

### Types of Anomalies:

- **Point Anomalies:** Individual instances in the data that are considered anomalous.
- **Contextual Anomalies:** Instances that are anomalous only in a specific context or condition.
- **Collective Anomalies:** A collection of related instances that are anomalous when analyzed together.

### Density Estimation for Anomaly Detection

1. **Kernel Density Estimation (KDE):**

KDE is a non-parametric method for estimating the PDF of a random variable. It assumes that the data points are sampled from an unknown distribution and uses a kernel function (e.g., Gaussian kernel) to estimate the density at each point. The density estimation is based on the distance of a given point from its neighboring data points. Anomalies are identified as instances that have a low density value compared to the majority of the data points.

2. **Gaussian Mixture Models (GMM):** GMM is a parametric method that assumes the data is generated from a mixture of multiple Gaussian distributions. The model estimates the parameters (means, covariances, and weights) of these Gaussian components, and the density of a point is calculated as the weighted sum of the densities from each component. Anomalies are identified as instances that have a low probability density under the fitted GMM.

3. **Histogram-based Estimation:** This is a simple non-parametric method that involves dividing the data space into bins or intervals and estimating the density as the frequency of data points in each bin. Anomalies are identified as instances that fall into low-density bins or regions.

4. **Parzen Window Estimation:** Similar to KDE, Parzen Window estimation is a non-parametric method that estimates the density at a point as the sum of kernel functions centered on each data point. The kernel function and its bandwidth parameter determine the smoothness of the density estimate.

5. **One-Class Support Vector Machines (OC-SVM):** OC-SVM is a semi-supervised method that aims to learn a decision boundary that separates the majority of the data points (normal instances) from the origin in a high-dimensional feature space. Anomalies are identified as instances that fall outside the learned decision boundary.

### Gaussian(Normal) Distribution

#### 1. **Definition**

The Gaussian distribution, also known as the normal distribution, is a continuous probability distribution characterized by its bell-shaped curve. It is defined by two parameters:

- **Mean (μ)**: The average or central value of the distribution.
- **Standard Deviation (σ)**: A measure of the spread or dispersion of the distribution.

The probability density function (PDF) of a normal distribution is given by:

$f(x|\mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} e^{ -\frac{(x - \mu)^2}{2\sigma^2} }$

#### 2. **Properties**

- **Symmetry**: The normal distribution is symmetric around the mean.
- **Bell-shaped Curve**: The highest point is at the mean, and the curve approaches zero as it moves away from the mean.
- **68-95-99.7 Rule**: Approximately 68% of the data lies within one standard deviation of the mean, 95% within two, and 99.7% within three.
- **Mean = Median = Mode**: For a normal distribution, these three measures of central tendency are equal.

#### 3. **Standard Normal Distribution**

A special case of the normal distribution is the standard normal distribution, which has a mean of 0 and a standard deviation of 1. The PDF of the standard normal distribution is:

$ \phi(z) = \frac{1}{\sqrt{2\pi}} e^{ -\frac{z^2}{2} } $

where $z = \frac{x - \mu}{\sigma}$

#### 4. **Z-Score**

The z-score is a measure of how many standard deviations an element is from the mean. It is calculated as:

$z = \frac{x - \mu}{\sigma}$

The z-score allows us to standardize different normal distributions, converting them to the standard normal distribution.

#### 5. **Applications**

The normal distribution is widely used in statistics and various fields due to the Central Limit Theorem, which states that the sum (or average) of a large number of independent and identically distributed random variables tends to be normally distributed, regardless of the original distribution.

#### Common Applications:

- **Quality Control**: Assessing variation in manufacturing processes.
- **Finance**: Modeling asset returns.
- **Psychometrics**: Designing standardized tests.
- **Natural and Social Sciences**: Modeling phenomena like heights, test scores, etc.

#### Key Points to Remember:

1. **Mean $(\mu$)**: The center of the distribution.
2. **Standard Deviation $\sigma$**: Controls the spread of the distribution. Larger $\sigma$ values result in a wider curve.
3. **Symmetry**: The curve is symmetric around the mean.
4. **Total Area**: The total area under the curve is 1, representing the total probability.

#### Practical Example:

Consider the heights of adult men in a population that follow a normal distribution with a mean height of 175 cm and a standard deviation of 10 cm. We can use this information to find the probability of a randomly selected man being within a certain height range.

For example, to find the probability that a man is between 165 cm and 185 cm, we can calculate the area under the normal distribution curve between these two points. This area represents the probability.

#### Calculating Gaussian(Normal) Distribution

```Python
import math

def gaussian_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the Gaussian (normal) distribution.

    Args:
        x (float): The value at which to calculate the PDF.
        mean (float): The mean (center) of the distribution.
        std_dev (float): The standard deviation of the distribution.

    Returns:
        float: The probability density value at x.
    """
    coefficient = 1 / (std_dev * math.sqrt(2 * math.pi))
    exponent = -((x - mean) ** 2) / (2 * std_dev ** 2)
    pdf = coefficient * math.exp(exponent)
    return pdf

# Example usage
x = 1.5
mean = 0
std_dev = 1

probability_density = gaussian_pdf(x, mean, std_dev)
print(f"The probability density at x={x} for a Gaussian distribution with mean={mean} and standard deviation={std_dev} is: {probability_density}")
```

- The function takes three arguments: `x` (the value at which to calculate the PDF), `mean` (the mean or center of the distribution), and `std_dev` (the standard deviation of the distribution).
- Inside the function, we calculate the probability density value using the formula for the Gaussian (normal) PDF:

```Python
f(x) = (1 / (σ * sqrt(2π))) * e^(-(x - μ)^2 / (2σ^2))
```

where `μ` is the mean, `σ` is the standard deviation, and `e` is the base of the natural logarithm (approximately 2.71828).

- The formula is broken down into three parts:
  - `coefficient`: `1 / (std_dev * math.sqrt(2 * math.pi))` calculates the coefficient term `(1 / (σ * sqrt(2π)))`.
  - `exponent`: `-((x - mean) ** 2) / (2 \* std_dev ** 2)` calculates the exponent term `-(x - μ)^2 / (2σ^2)`.
  - pdf: `coefficient * math.exp(exponent)` calculates the final PDF value by multiplying the coefficient and the exponential term `e^(exponent)`.
- The function returns the calculated `pdf` value.

#### Conclusion

The normal distribution is a fundamental concept in statistics, widely used in various fields for its mathematical properties and the central role it plays in the Central Limit Theorem. Understanding its properties and how to work with it is crucial for data analysis and interpretation.

### Implementing Anomaly Detection Algorithms from Scratch

Let's implement a simple anomaly detection algorithm using the Z-Score method.

#### Z-Score Anomaly Detection

1. **Calculate Mean and Standard Deviation**: Compute the mean and standard deviation of the data.
2. **Compute Z-Scores**: For each data point, calculate the Z-score.
3. **Set a Threshold**: Decide on a threshold for the Z-score. Data points with a Z-score above the threshold are considered anomalies.

```python
import numpy as np

# Generate some data
data = np.array([10, 12, 12, 13, 12, 11, 14, 100, 12, 11, 12, 13, 14, 10])

# Step 1: Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Step 2: Compute Z-scores
z_scores = (data - mean) / std_dev

# Step 3: Set a threshold for Z-score
threshold = 3

# Identify anomalies
anomalies = data[np.abs(z_scores) > threshold]

print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Z-Scores:", z_scores)
print("Anomalies:", anomalies)
```

### Anomaly Detection Using Gaussian Distribution

#### Step 1: Understanding Gaussian Distribution

A Gaussian distribution is defined by two parameters:

- **Mean (μ)**: The average of the data.
- **Variance (σ²)**: The spread of the data.

The probability density function (PDF) for a Gaussian distribution is given by:<br>
$P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$

#### Step 2: Fit a Gaussian Distribution to the Data

To detect anomalies, we first need to fit a Gaussian distribution to our data. This involves calculating the mean and variance of the dataset.

#### Step 3: Calculate Probability Density for Each Data Point

Using the PDF, we calculate the probability of each data point under the Gaussian distribution. Data points with low probabilities are considered anomalies.

#### Step 4: Set a Threshold

Choose a threshold for the probability. Data points with probabilities below this threshold are labeled as anomalies.

### Implementation

Now we implement this step-by-step using Python:

```python
import numpy as np

# Generate some example data
data = np.array([10, 12, 12, 13, 12, 11, 14, 100, 12, 11, 12, 13, 14, 10])

# Step 1: Fit Gaussian distribution to the data

# Calculate mean (μ) and variance (σ²)
mean = np.mean(data)
variance = np.var(data)
std_dev = np.sqrt(variance)

print("Mean:", mean)
print("Variance:", variance)
print("Standard Deviation:", std_dev)

# Step 2: Calculate probability density for each data point
def gaussian_pdf(x, mean, std_dev):
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

probabilities = gaussian_pdf(data, mean, std_dev)

print("Probabilities:", probabilities)

# Step 3: Set a threshold for anomalies
threshold = 0.01  # This can be tuned based on the dataset

# Step 4: Identify anomalies
anomalies = data[probabilities < threshold]

print("Anomalies:", anomalies)
```

#### Detailed Explanation

1. **Fit Gaussian Distribution**:

   - **Mean (μ)**: Sum all data points and divide by the number of points.
   - **Variance (σ²)**: Sum the squared differences from the mean and divide by the number of points.

   ```python
   mean = np.mean(data)
   variance = np.var(data)
   std_dev = np.sqrt(variance)
   ```

2. **Calculate Probability Density**:

   - Use the Gaussian PDF formula to compute the probability for each data point.

   ```python
   def gaussian_pdf(x, mean, std_dev):
       return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

   probabilities = gaussian_pdf(data, mean, std_dev)
   ```

3. **Set Threshold**:

   - Choose a threshold value for the probability. Data points with a probability less than this threshold are considered anomalies. The choice of threshold can be subjective and might require tuning based on the specific dataset.

   ```python
   threshold = 0.01  # Example threshold
   ```

4. **Identify Anomalies**:

   - Compare the probability of each data point to the threshold. If the probability is less than the threshold, classify the point as an anomaly.

   ```python
   anomalies = data[probabilities < threshold]
   ```

Sure, let's dive into anomaly detection using the Gaussian distribution function (also known as the Normal distribution). This method assumes that the data follows a Gaussian distribution, and anomalies are identified based on their likelihood under this distribution.

### Step-by-Step Guide to Anomaly Detection Using Gaussian Distribution

#### Step 1: Understanding Gaussian Distribution

A Gaussian distribution is defined by two parameters:

- **Mean (μ)**: The average of the data.
- **Variance (σ²)**: The spread of the data.

The probability density function (PDF) for a Gaussian distribution is given by:

$P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$

#### Step 2: Fit a Gaussian Distribution to the Data

To detect anomalies, we first need to fit a Gaussian distribution to our data. This involves calculating the mean and variance of the dataset.

#### Step 3: Calculate Probability Density for Each Data Point

Using the PDF, we calculate the probability of each data point under the Gaussian distribution. Data points with low probabilities are considered anomalies.

#### Step 4: Set a Threshold

Choose a threshold for the probability. Data points with probabilities below this threshold are labeled as anomalies.

### Implementation

Let's implement this step-by-step using Python:

```python
import numpy as np

# Generate some example data
data = np.array([10, 12, 12, 13, 12, 11, 14, 100, 12, 11, 12, 13, 14, 10])

# Step 1: Fit Gaussian distribution to the data

# Calculate mean (μ) and variance (σ²)
mean = np.mean(data)
variance = np.var(data)
std_dev = np.sqrt(variance)

print("Mean:", mean)
print("Variance:", variance)
print("Standard Deviation:", std_dev)

# Step 2: Calculate probability density for each data point
def gaussian_pdf(x, mean, std_dev):
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

probabilities = gaussian_pdf(data, mean, std_dev)

print("Probabilities:", probabilities)

# Step 3: Set a threshold for anomalies
threshold = 0.01  # This can be tuned based on the dataset

# Step 4: Identify anomalies
anomalies = data[probabilities < threshold]

print("Anomalies:", anomalies)
```

### Detailed Explanation

1. **Fit Gaussian Distribution**:

   - **Mean (μ)**: Sum all data points and divide by the number of points.
   - **Variance (σ²)**: Sum the squared differences from the mean and divide by the number of points.

   ```python
   mean = np.mean(data)
   variance = np.var(data)
   std_dev = np.sqrt(variance)
   ```

2. **Calculate Probability Density**:

   - Use the Gaussian PDF formula to compute the probability for each data point.

   ```python
   def gaussian_pdf(x, mean, std_dev):
       return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

   probabilities = gaussian_pdf(data, mean, std_dev)
   ```

3. **Set Threshold**:

   - Choose a threshold value for the probability. Data points with a probability less than this threshold are considered anomalies. The choice of threshold can be subjective and might require tuning based on the specific dataset.

   ```python
   threshold = 0.01  # Example threshold
   ```

4. **Identify Anomalies**:

   - Compare the probability of each data point to the threshold. If the probability is less than the threshold, classify the point as an anomaly.

   ```python
   anomalies = data[probabilities < threshold]
   ```

### Tuning and Evaluation

- **Threshold Selection**: The threshold value can significantly affect the results. It might need to be adjusted based on the specific application and the proportion of anomalies expected.
- **Validation**: Use a separate validation set to test the performance of the anomaly detection algorithm and adjust the threshold accordingly.
- **Evaluation Metrics**: Use precision, recall, F1-score, and ROC-AUC to evaluate the performance if labeled data is available.

### Anomaly detection vs. Supervised Learning

#### 1. Definition and Purpose

**Anomaly Detection**:

- **Definition**: Anomaly detection focuses on identifying data points that deviate significantly from the majority of the data.
- **Purpose**: Its main goal is to detect unusual patterns or outliers that might indicate errors, fraud, or other significant but rare events.

**Supervised Learning**:

- **Definition**: Supervised learning involves training a model on a labeled dataset, where each data point is associated with a label.
- **Purpose**: The objective is to predict the label for new, unseen data based on the learned relationship between features and labels.

#### 2. Data Requirements

**Anomaly Detection**:

- **Labeled Data**: Often works with unlabeled data, or in cases where only a small fraction of data is labeled as anomalies. It can also be semi-supervised with labels only for the normal class.
- **Data Distribution**: Assumes that anomalies are rare and different from the normal data. The focus is on identifying these rare occurrences.

**Supervised Learning**:

- **Labeled Data**: Requires a fully labeled dataset where each instance has a corresponding label. The labels can be categorical (classification) or continuous (regression).
- **Data Distribution**: Assumes that data is representative and sufficiently balanced for all classes or outcomes.

#### 3. Techniques and Algorithms

**Anomaly Detection**:

- **Algorithms**: Includes statistical methods (e.g., Z-score, Gaussian distribution), clustering methods (e.g., DBSCAN), proximity-based methods (e.g., K-Nearest Neighbors), and machine learning methods (e.g., Isolation Forest, Autoencoders).
- **Focus**: Models are designed to identify deviations from the norm rather than learning explicit decision boundaries.

**Supervised Learning**:

- **Algorithms**: Includes linear regression, logistic regression, decision trees, support vector machines, neural networks, and ensemble methods (e.g., Random Forest, Gradient Boosting).
- **Focus**: Models learn a mapping from input features to output labels, optimizing for accuracy or error minimization.

#### 4. Evaluation Metrics

**Anomaly Detection**:

- **Metrics**: Often uses precision, recall, F1-score, ROC-AUC, and the confusion matrix. The focus is on minimizing false positives and false negatives.
- **Challenge**: Evaluation can be tricky due to the imbalanced nature of the data, where anomalies are much less frequent than normal instances.

**Supervised Learning**:

- **Metrics**: Common metrics include accuracy, precision, recall, F1-score, mean squared error (MSE), and ROC-AUC, depending on whether the task is classification or regression.
- **Challenge**: The key is to balance the performance across all classes, especially in imbalanced datasets.

#### 5. Applications

**Anomaly Detection**:

- **Applications**: Fraud detection, network security (intrusion detection), fault detection in manufacturing, medical diagnosis, and predictive maintenance.
- **Scenario**: Used in situations where abnormal instances are rare but critical to identify.

**Supervised Learning**:

- **Applications**: Spam detection, image recognition, speech recognition, medical diagnosis (classification), house price prediction, and sales forecasting (regression).
- **Scenario**: Used in scenarios where there is ample labeled data available for training and the goal is to predict outcomes based on learned patterns.

#### Example to Illustrate the Difference

**Anomaly Detection Example**:

- **Fraud Detection**: In credit card transactions, anomaly detection is used to identify transactions that are significantly different from typical spending patterns, suggesting potential fraud.

**Supervised Learning Example**:

- **Spam Email Classification**: In email filtering, supervised learning is used to train a model on a labeled dataset of emails (spam vs. non-spam) to classify incoming emails accordingly.
