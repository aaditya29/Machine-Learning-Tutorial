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
