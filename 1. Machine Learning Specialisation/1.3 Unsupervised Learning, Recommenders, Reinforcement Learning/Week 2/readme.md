# Important Notes Week 2

## Collaborative Filtering

Collaborative filtering is a popular technique used in recommendation systems to make automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The key idea is that if two users have agreed on certain items in the past, they are likely to agree again in the future.

### Types of Collaborative Filtering

There are two main types of collaborative filtering:

1. **User-based Collaborative Filtering**:

   - **Concept**: It recommends items to a user by finding other users who have similar preferences. The system looks for users who have a history of agreeing with the target user and suggests items that those similar users have liked.
   - **Steps**:
     1. Identify the target user.
     2. Find users who have similar tastes (using similarity metrics like cosine similarity, Pearson correlation, etc.).
     3. Recommend items that these similar users have liked and that the target user has not yet rated.

2. **Item-based Collaborative Filtering**:
   - **Concept**: It recommends items based on the similarity between items. Instead of looking for similar users, it looks for items that are similar to what the target user has liked in the past.
   - **Steps**:
     1. Identify items the target user has rated.
     2. Find items that are similar to those items (using similarity metrics).
     3. Recommend these similar items to the target user.

### How It Works

#### User-based Example

1. **Data Collection**: Collect user-item interaction data (e.g., user ratings for movies).
2. **Similarity Computation**: Compute the similarity between users based on their ratings. For instance, using cosine similarity:

   $\text{sim}(u, v) = \frac{\sum_{i \in I} r_{u,i} \times r_{v,i}}{\sqrt{\sum_{i \in I} r_{u,i}^2} \times \sqrt{\sum_{i \in I} r_{v,i}^2}}$

   where $r_{u,i}$ is the rating given by user $u$ to item $i$, and $I$ is the set of items rated by both users $u$ and $v$.

3. **Prediction**: Predict the rating for an item $i$ for user $u$ by taking a weighted sum of the ratings of similar users:
   $\hat{r}_{u,i} = \frac{\sum_{v \in U} \text{sim}(u, v) \times r_{v,i}}{\sum_{v \in U} |\text{sim}(u, v)|}$

   where $U$ is the set of users who have rated item $i$.

#### Item-based Example

1. **Data Collection**: Collect user-item interaction data.
2. **Similarity Computation**: Compute the similarity between items based on user ratings. For example, using cosine similarity:
   $\text{sim}(i, j) = \frac{\sum_{u \in U} r_{u,i} \times r_{u,j}}{\sqrt{\sum_{u \in U} r_{u,i}^2} \times \sqrt{\sum_{u \in U} r_{u,j}^2}}$

   where $r_{u,i}$ is the rating given by user $u$ to item $i$, and $U$ is the set of users who have rated both items $i$ and $j$.

3. **Prediction**: Predict the rating for an item $i$ for user $u$ by taking a weighted sum of the ratings of similar items:
   $\hat{r}_{u,i} = \frac{\sum_{j \in I} \text{sim}(i, j) \times r_{u,j}}{\sum_{j \in I} |\text{sim}(i, j)|}$

   where $I$ is the set of items rated by user $u$.

### Advantages and Disadvantages

**Advantages**:

- **Simplicity**: Easy to implement and understand.
- **Effectiveness**: Works well when there is a large amount of user-item interaction data.

**Disadvantages**:

- **Sparsity**: In many real-world scenarios, the user-item matrix is sparse (most users have rated only a few items), which can limit the effectiveness of collaborative filtering.
- **Scalability**: As the number of users and items grows, computing similarities and making predictions can become computationally expensive.
- **Cold Start Problem**: Difficult to recommend items to new users or recommend new items due to a lack of existing ratings.

### Applications

Collaborative filtering is widely used in various applications, including:

- **E-commerce**: Product recommendations (e.g., Amazon).
- **Streaming Services**: Movie, TV show, and music recommendations (e.g., Netflix, Spotify).
- **Social Networks**: Friend recommendations (e.g., Facebook, LinkedIn).

### Cost Function in Collaborative Filtering Using Per-Item Features

In collaborative filtering, especially when using matrix factorization techniques like Singular Value Decomposition (SVD) or when incorporating per-item (and per-user) features, the cost function is crucial for learning the latent factors that predict user-item interactions.

#### Matrix Factorization with Per-Item Features

In matrix factorization, we aim to factorize the user-item interaction matrix $R$ into two lower-dimensional matrices: a user matrix $U$ and an item matrix $V$. Each user and item are represented by vectors (latent factors) in these matrices.

When incorporating per-item features, we extend this idea by adding feature vectors that describe each item. This can be done in several ways, one of which is to modify the prediction formula and the cost function to include these features.

#### Cost Function

The typical prediction for a rating $r*{ui}$ (rating of user $u$ for item $i$ is given by:

$\hat{r}*{ui} = \mu + b_u + b_i + q_i^T p_u$

where:

- $\mu$ is the global bias (average rating across all users and items).
- $b_u$ is the bias term for user $u$.
- $b_i$ is the bias term for item $i$.
- $q_i$ is the latent factor vector for item $i$.
- $p_u$ is the latent factor vector for user $u$.

When incorporating per-item features, let $x*i$ be the feature vector for item $i$. The prediction formula can be extended to:

$\hat{r}*{ui} = \mu + b_u + b_i + q_i^T p_u + f(x_i, \theta)$

where $f(x_i, \theta)$ is a function of item features $x_i$ and some parameters $\theta$.

### Cost Function with Regularization

The cost function $J$ to be minimized, incorporating regularization to prevent overfitting, is given by:

$J = \frac{1}{2} \sum*{(u,i) \in \mathcal{K}} \left( r*{ui} - \hat{r}\_{ui} \right)^2 + \frac{\lambda}{2} \left( \sum_u \left\| p_u \right\|^2 + \sum_i \left\| q_i \right\|^2 + \left\| \theta \right\|^2 \right)$

where:

- $\mathcal{K}$ is the set of user-item pairs for which we have ratings.
- $r\_{ui}$ is the actual rating of user $u$ for item $i$.
- $\hat{r}\_{ui}$ is the predicted rating.
- $\lambda$ is the regularization parameter.
- $\left\| p_u \right\|^2$ and $\left\| q_i \right\|^2$ are the squared norms of the user and item latent factor vectors, respectively.
- $\left\| \theta \right\|^2$ is the squared norm of the feature weights vector.

### Gradient Descent

To minimize the cost function, we can use gradient descent. The update rules for the parameters are derived from the partial derivatives of $J$:

- For user latent factors $p_u$:
  $p_u \leftarrow p_u + \gamma \left( (r_{ui} - \hat{r}_{ui}) q_i - \lambda p_u \right)$

- For item latent factors $q_i$:
  $q_i \leftarrow q_i + \gamma \left( (r_{ui} - \hat{r}_{ui}) p_u - \lambda q_i \right)$

- For user biases $b_u$:
  $b_u \leftarrow b_u + \gamma \left( (r_{ui} - \hat{r}_{ui}) - \lambda b_u \right)$

- For item biases $b_i$:
  $b_i \leftarrow b_i + \gamma \left( (r_{ui} - \hat{r}_{ui}) - \lambda b_i \right)$

- For feature weights $\theta$:
  $\theta \leftarrow \theta + \gamma \left( (r_{ui} - \hat{r}_{ui}) x_i - \lambda \theta \right)$

where $\gamma$ is the learning rate.

### Binary Labels in Collaborative Filtering

In collaborative filtering, binary labels represent the presence or absence of an interaction between a user and an item, rather than a scalar rating. For example, a binary label might indicate whether a user has viewed, clicked, or purchased an item (1 if they have, 0 if they haven't). This approach is often used in scenarios where explicit ratings are unavailable but implicit feedback is plentiful.

#### Key Concepts of Binary Collaborative Filtering

1. **Binary Interaction Matrix**:

   - Instead of a rating matrix $R$, we have a binary interaction matrix $B$, where $b\_{ui}$ is 1 if user $u$ has interacted with item $i$, and 0 otherwise.
   - For example, if user $u$ watched movie $i$, $b*{ui} = 1$; if they haven't, $b*{ui} = 0$.

2. **Implicit Feedback**:
   - Implicit feedback is inferred from user behavior, such as clicks, views, or purchases. It is binary in nature and indicates whether an interaction occurred.
   - Implicit feedback can be more abundant but noisier compared to explicit ratings.

### Collaborative Filtering Techniques with Binary Labels

#### User-based Collaborative Filtering

1. **Similarity Computation**:

   - Calculate similarity between users based on binary interactions. Common measures include:
     - **Jaccard Similarity**: Measures the similarity between two sets of binary interactions.
       $\text{sim}(u, v) = \frac{|B_u \cap B_v|}{|B_u \cup B_v|}$
       where $B_u$ and $B_v$ are the sets of items interacted by users $u$ and $v$.

2. **Recommendation Generation**:
   - For a target user $u$, identify similar users and recommend items that these similar users have interacted with but the target user has not.

#### Matrix Factorization with Binary Labels

Matrix factorization can also be applied to binary interaction data. The objective is to learn latent factors that explain the observed interactions.

#### Logistic Matrix Factorization

1. **Prediction**:

   - Instead of predicting a rating, we predict the probability of an interaction using the logistic function.
     $\hat{b}\_{ui} = \sigma(p_u \cdot q_i) = \frac{1}{1 + \exp(-p_u \cdot q_i)}$
     where $p_u$ and $q_i$ are the latent factors for user $u$ and item $i$, and $\sigma$ is the logistic sigmoid function.

2. **Cost Function**:

   - The cost function is based on binary cross-entropy loss.
     $J = -\sum*{(u,i) \in \mathcal{K}} \left[ b*{ui} \log(\hat{b}_{ui}) + (1 - b_{ui}) \log(1 - \hat{b}\_{ui}) \right] + \frac{\lambda}{2} \left( \sum_u \|p_u\|^2 + \sum_i \|q_i\|^2 \right)$

     where $\mathcal{K}$ is the set of observed interactions, and $\lambda$ is the regularization parameter.

3. **Optimization**:
   - Use gradient descent to minimize the cost function. Update rules are derived from the gradients of the cost function with respect to the latent factors.

## Recommender Systems Implementation Detail

Mean normalization is a technique used in data preprocessing, especially for machine learning and statistical modeling, to make features of a dataset have a mean of zero. This helps in centering the data and can improve the performance of some machine learning algorithms. Here’s a step-by-step guide on what mean normalization is and how to apply it:

### What is Mean Normalization?

Mean normalization adjusts the values of a feature so that the new mean of the feature is zero. It typically also scales the feature values to lie within a certain range, often between -1 and 1.

### Why Use Mean Normalization?

1. **Centering Data**: By having features centered around zero, the optimization algorithms (e.g., gradient descent) can converge faster and more reliably.
2. **Feature Scaling**: Normalized features can prevent some features from dominating others due to their different scales, improving the performance of algorithms like gradient descent.
3. **Handling Different Units**: When features have different units (e.g., meters vs. kilograms), mean normalization helps to put them on a comparable scale.
