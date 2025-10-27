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

### Formula for Mean Normalization

The formula for mean normalization of a feature $( x )$ is:

$[ x' = \frac{x - \mu}{x_{max} - x_{min}})$

Where:

- $( x)$ is the original feature value.
- $( \mu)$ is the mean of the feature values.
- $(x_{max})$ and $( x_{min})$ are the maximum and minimum values of the feature, respectively.
- $( x')$ is the normalized feature value.

### Steps for Mean Normalization

1. **Calculate the Mean**: Compute the mean $( \mu)$ of the feature values.
2. **Determine the Range**: Find the maximum $(x_{max})$ and minimum $(x_{min})$ values of the feature.
3. **Apply the Formula**: Use the mean normalization formula to transform each feature value.

#### Example of Mean Normalisation

Let's consider a dataset with a single feature:

| Original Feature |
| ---------------- |
| 10               |
| 20               |
| 30               |
| 40               |
| 50               |

1. **Calculate the Mean**:
   $[ \mu = \frac{10 + 20 + 30 + 40 + 50}{5} = 30 ]$

2. **Determine the Range**:
   $[ x_{max} = 50]$
   $[ x_{min} = 10]$

3. **Apply the Formula**:

   - For the first value (10):
     $[ x' = \frac{10 - 30}{50 - 10} = \frac{-20}{40} = -0.5]$

   - For the second value (20):
     $[ x' = \frac{20 - 30}{50 - 10} = \frac{-10}{40} = -0.25]$

   - For the third value (30):
     $[ x' = \frac{30 - 30}{50 - 10} = \frac{0}{40} = 0]$
   - For the fourth value (40):
     $[ x' = \frac{40 - 30}{50 - 10} = \frac{10}{40} = 0.25]$

   - For the fifth value (50):
     $[ x' = \frac{50 - 30}{50 - 10} = \frac{20}{40} = 0.5]$

The normalized feature values are:

| Normalized Feature |
| ------------------ |
| -0.5               |
| -0.25              |
| 0                  |
| 0.25               |
| 0.5                |

### Applications

- **Linear Regression**: Mean normalization can make gradient descent converge more quickly.
- **Neural Networks**: Input normalization is often used to improve the stability and performance of training.
- **Principal Component Analysis (PCA)**: Helps in centering the data for better component extraction.

## TensorFlow Implementation of Collaborative Filtering

Implementing collaborative filtering using TensorFlow and gradient descent is important due to TensorFlow's scalability, efficiency, flexibility, and the ability to handle large and sparse datasets. The automatic differentiation and extensive optimization capabilities, along with a supportive ecosystem, make TensorFlow a powerful tool for developing robust and efficient recommender systems.

### Custom Training Loop

```Python
w = tf.Variable(3.0)#tf.variables we want to optimise
x = 1.0
y = 1.0#target value
alpha = 0.01#learning rate

iterations = 30

for iter in range(iterations):#for 30 iterations
   #Using tensorflow's gradient tape to record steps
   # and using to compute the cost j, to enable the auto differentitation
   with tf.GradientTape() as tape:
      fwb = w*x
      costJ = (fwb - y)**2#J = (wx-1)^2

   #Using the gradient tape to calculate the gradients of the cost
   # w.r.t the parameter w
   [dJdw] = tape.gradient(costJ, [w])

   #Run one step of gradient descent by updating
   #the value of w to reduce the cost
   w.assign_add(-alpha*dJdw)#modifing special function
```

### Implemntation in Tensorflow Syntax

```Python
#Instantiating an optimizer
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200

for iter in range(iterations):
   #Using tensorflow's GradientTape
   #to record the operations used to compute the cost
   with tf.GradientTape() as tape:

      #computing the cost(forward pass is included in cost)
      cost_value = cofiCostFuncV(X,W,b, Ynorm, R,
      num_users, num_movies, lambda)#repeating till convergence

   #Using the gradient tape to automatically retrieve
   #the gradients of the trainable variables w.r.t the loss
   grads = tape.gradient(cost_value, [X,W,b])

   #Running one stept of gradient descent by updating the value
   #of the variables to minimise the loss
   optimizer.apply_gradients(zip(grads, [X,W,b]))
```

## Content Based Filtering

Content-based filtering is a type of recommender system technique in machine learning that relies on the features of items and users to make recommendations. It uses the characteristics of items to recommend other items similar to what a user likes, based on their previous interactions or explicit preferences. Here’s a detailed look at how content-based filtering works:

### Key Concepts

1. **Feature Extraction**: Content-based filtering involves extracting features from items and users. Features can be various attributes or properties. For example:

   - For movies: Genre, director, cast, keywords, etc.
   - For books: Author, genre, keywords, publication date, etc.

2. **User Profile**: A user profile is created based on the features of items the user has interacted with. This can be a weighted average of the features of items the user has rated highly or consumed frequently.

3. **Similarity Measure**: The system uses a similarity measure to compare the user profile with the features of items. Common similarity measures include cosine similarity, Euclidean distance, or Pearson correlation.

### Steps in Content-Based Filtering

1. **Data Collection**: Gather data about items and users. This data can be explicit (ratings, likes, etc.) or implicit (browsing history, clicks, etc.).

2. **Feature Extraction**: Extract relevant features from items. For example, in a movie recommendation system, features might include genre, director, and cast.

3. **Profile Building**: Build a user profile based on the features of items the user has interacted with. This profile represents the user’s preferences.

4. **Recommendation Generation**: Calculate the similarity between the user profile and all items in the database. Recommend items with the highest similarity scores to the user.

### Example

Suppose you have a movie recommendation system:

1. **Data Collection**: You have data on movies, including attributes like genre, director, and cast. You also have user data, such as which movies a user has watched and their ratings.

2. **Feature Extraction**: Each movie is represented by a feature vector. For instance:

   - Movie 1: [Action, Sci-Fi, Christopher Nolan, Leonardo DiCaprio]
   - Movie 2: [Romance, Drama, Richard Linklater, Ethan Hawke]

3. **Profile Building**: If a user likes “Inception” (an action, sci-fi movie by Christopher Nolan with Leonardo DiCaprio), the user profile might emphasize action and sci-fi genres and prefer movies by Nolan or starring DiCaprio.

4. **Recommendation Generation**: Compute similarity scores between the user profile and the feature vectors of other movies. Recommend movies with high similarity scores, such as “Interstellar” (another action, sci-fi movie by Nolan) or “The Dark Knight” (an action movie by Nolan).

### Advantages and Disadvantages

**Advantages**:

- **Personalization**: Highly personalized recommendations based on individual user preferences.
- **No Cold Start for Items**: Can recommend new items as long as their features are known.

**Disadvantages**:

- **Limited to Known Features**: Can only recommend items that are similar to what the user has already liked.
- **Overspecialization**: May not introduce users to diverse content, leading to a narrow set of recommendations.
- **Cold Start for Users**: Difficult to recommend items to new users with no interaction history.

#### Implementation Example in Python

Here’s a simplified example using Python and the scikit-learn library:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data
data = {
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Pride and Prejudice'],
    'genre': ['Action Sci-Fi', 'Action Sci-Fi', 'Action', 'Romance Drama'],
}

df = pd.DataFrame(data)

# Vectorize the genre column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df.index[df['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Example usage
print(get_recommendations('Inception'))
```

This code snippet creates a content-based recommendation system using movie genres. It vectorizes the genre information, calculates the cosine similarity between movies, and recommends movies based on their similarity to a given title.

### Deep Learning for Content-Based Filtering

Deep learning enhances content-based filtering by leveraging its ability to automatically extract and learn complex features from raw data, such as text, images, and audio. This can be particularly powerful in domains where the data is high-dimensional and unstructured. Here's a detailed look at how deep learning can be applied to content-based filtering:

#### Key Concepts

1. **Feature Learning**: Deep learning models can automatically learn representations of items and users from raw data, removing the need for manual feature extraction.
2. **Neural Networks**: Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers can be used depending on the type of data (images, text, sequential data).
3. **Embedding Layers**: These layers convert items and user profiles into dense vector representations that can be used for similarity calculations.

#### Steps in Deep Learning-Based Content Filtering

1. **Data Preparation**: Gather and preprocess the data (e.g., text, images, audio).
2. **Model Selection**: Choose an appropriate neural network architecture (CNN, RNN, Transformer).
3. **Training**: Train the model to learn representations of items based on their features.
4. **User Profile Construction**: Aggregate the learned representations of items a user has interacted with to create a user profile.
5. **Recommendation Generation**: Compute similarities between the user profile and item representations to generate recommendations.

#### Example Applications

- **Text Data**: Using models like BERT for text features in articles, books, or movies.
- **Image Data**: Using CNNs for visual features in product recommendations.
- **Audio Data**: Using RNNs or CNNs for music or podcast recommendations.

### Example: Text-Based Content Filtering with Deep Learning

Here's an example using a pre-trained BERT model for movie plot summaries:

#### Step 1: Data Preparation

Prepare a dataset of movie plot summaries:

```python
import pandas as pd

data = {
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Pride and Prejudice'],
    'plot': [
        "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
        "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
        "When the menace known as the Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.",
        "Sparks fly when spirited Elizabeth Bennet meets single, rich, and proud Mr. Darcy. But Mr. Darcy reluctantly finds himself falling in love with a woman beneath his class."
    ]
}

df = pd.DataFrame(data)
```

#### Step 2: Model Selection and Feature Extraction

Use a pre-trained BERT model to encode the plot summaries:

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode plot summaries
def encode_plot(plot):
    inputs = tokenizer(plot, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

df['plot_embedding'] = df['plot'].apply(encode_plot)
```

#### Step 3: User Profile Construction

Assume the user has liked "Inception" and "Interstellar":

```python
user_liked_movies = ['Inception', 'Interstellar']
user_profile = df[df['title'].isin(user_liked_movies)]['plot_embedding'].mean()
```

#### Step 4: Recommendation Generation

Compute cosine similarity between the user profile and all movie embeddings:

```python
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(user_profile, df):
    df['similarity'] = df['plot_embedding'].apply(lambda x: cosine_similarity(user_profile.reshape(1, -1), x.reshape(1, -1))[0][0])
    return df.sort_values(by='similarity', ascending=False).head(3)['title']

recommended_movies = recommend_movies(user_profile, df)
print(recommended_movies)
```

### Cost Function in Deep Learning for Content-Based Filtering

In deep learning for content-based filtering, the cost function (also known as the loss function) is used to measure how well the model's predictions match the actual data. The goal during training is to minimize this cost function, thereby improving the model's accuracy.

For content-based filtering, the cost function can vary depending on the specific approach and data type. However, a common approach is to use a regression-based loss function like Mean Squared Error (MSE) when predicting continuous values or Binary Cross-Entropy when dealing with binary classification tasks. Here, we'll focus on MSE, which is widely used in content-based filtering scenarios.

### Steps and Formula for Cost Function in Deep Learning for Content-Based Filtering

1. **Data Preparation**:

   - **Item Features**: Represent items using their features. For example, movies could be represented by their plot summaries, genres, cast, etc.
   - **User Profile**: Construct a user profile based on the features of items the user has interacted with.

2. **Model Architecture**:

   - Design a neural network that can take item features as input and output a prediction (e.g., rating or relevance score).

3. **Forward Pass**:

   - Input data (item features and user profile) passes through the network to generate predictions.

4. **Cost Function Calculation**:

   - Compare the model's predictions with the actual values using the cost function.

5. **Backpropagation and Optimization**:
   - Compute gradients of the cost function with respect to the model parameters.
   - Update the model parameters to minimize the cost function.

### Example with Mean Squared Error (MSE) Cost Function

#### Step 1: Data Preparation

Let's assume you have a dataset with user ratings for different movies. Each movie has features (e.g., plot summary embeddings) and each user has a profile built from their interactions.

#### Step 2: Model Architecture

A simple neural network might have the following structure:

- Input Layer: Takes the item feature vector.
- Hidden Layers: Processes the input to learn complex representations.
- Output Layer: Outputs a predicted rating or relevance score.

#### Step 3: Forward Pass

For a single user and movie:

- \( \mathbf{x} \) is the feature vector of the movie.
- \( \mathbf{w}\_1, \mathbf{w}\_2, ..., \mathbf{w}\_n \) are the weights of the neural network layers.
- \( \mathbf{b}\_1, \mathbf{b}\_2, ..., \mathbf{b}\_n \) are the biases of the neural network layers.

The forward pass through the network can be represented as:
\[ \hat{y} = f(\mathbf{x}; \theta) \]
where \( f \) is the neural network function and \( \theta \) represents all the model parameters (weights and biases).

#### Step 4: Cost Function Calculation

Mean Squared Error (MSE) is calculated as:
\[ \text{MSE} = \frac{1}{N} \sum\_{i=1}^N (y_i - \hat{y}\_i)^2 \]
where:

- \( y_i \) is the actual rating or relevance score for the \( i \)-th item.
- \( \hat{y}\_i \) is the predicted rating or relevance score for the \( i \)-th item.
- \( N \) is the total number of items.

#### Step 5: Backpropagation and Optimization

The gradients of the MSE with respect to the model parameters are computed using backpropagation. The parameters are then updated using an optimization algorithm like Stochastic Gradient Descent (SGD) or Adam.

### Example in Python

Here's a simplified example using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Sample data: movie features (embeddings) and user ratings
movie_features = torch.tensor([[0.1, 0.2], [0.4, 0.5], [0.3, 0.8]], dtype=torch.float32)
user_ratings = torch.tensor([5.0, 3.0, 4.0], dtype=torch.float32)

# Simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # 100 epochs
    model.train()

    optimizer.zero_grad()
    outputs = model(movie_features).squeeze()

    loss = criterion(outputs, user_ratings)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

# Example prediction
model.eval()
new_movie_feature = torch.tensor([0.2, 0.3], dtype=torch.float32)
predicted_rating = model(new_movie_feature.unsqueeze(0))
print(f'Predicted Rating: {predicted_rating.item():.4f}')
```

### Explanation of Code

1. **Data Preparation**: Define movie features and user ratings.
2. **Model Architecture**: A simple neural network with one hidden layer.
3. **Forward Pass**: The model processes input movie features to produce predictions.
4. **Cost Function Calculation**: MSE is used to calculate the difference between predicted and actual ratings.
5. **Backpropagation and Optimization**: The model parameters are updated to minimize the MSE.

### Retrieve and Return Process for Recommender System

In recommender systems, the "retrieve and return" step refers to the process of generating recommendations for users based on various algorithms and models. This step involves selecting a set of items from a larger pool that are most relevant to the user and presenting these items as recommendations. Here’s a detailed breakdown:

#### Retrieve Step

The retrieve step involves identifying and fetching a subset of items from the entire catalog that are potentially relevant to the user. This is typically done using one or more retrieval methods, which can be broadly categorized into several types:

1. **Content-Based Filtering**: Items are retrieved based on their similarity to items the user has interacted with previously. Features such as keywords, genres, or item descriptions are used to find similar items.

2. **Collaborative Filtering**: Items are retrieved based on the preferences of similar users or the similarities between items. This can be user-based or item-based collaborative filtering.

3. **Hybrid Methods**: A combination of content-based and collaborative filtering techniques to leverage the strengths of both approaches.

4. **Matrix Factorization**: Techniques like Singular Value Decomposition (SVD) or Alternating Least Squares (ALS) decompose the user-item interaction matrix to retrieve latent factors representing users and items, facilitating the recommendation of items with high relevance scores.

5. **Deep Learning Methods**: Advanced neural networks such as autoencoders, neural collaborative filtering, or embeddings from models like Word2Vec for text data can be used to retrieve items by learning complex patterns in user-item interactions.

#### Return Step

The return step involves ranking and presenting the retrieved items to the user. The goal is to order the items in a way that maximizes the likelihood of user satisfaction and engagement. This typically involves:

1. **Ranking**: Items retrieved in the previous step are ranked based on a relevance score. This score can be derived from various models, such as:

   - Predictive models (e.g., regression or classification models predicting user ratings or clicks).
   - Learning to rank models (e.g., RankNet, LambdaRank) which are trained to optimize ranking directly.
   - Hybrid models combining various signals like item popularity, user preferences, and contextual information.

2. **Post-Processing**: Additional filters or adjustments may be applied to the ranked list. This can include:

   - **Diversity**: Ensuring that the recommendations are not too similar to each other to expose users to a broader range of items.
   - **Novelty**: Prioritizing items the user has not seen or interacted with before.
   - **Business Rules**: Incorporating specific business objectives such as promoting certain items, ensuring fairness, or adhering to regulatory requirements.

3. **Presentation**: Finally, the ranked list of items is presented to the user through the user interface. This can involve considerations of layout, usability, and personalization of the presentation format.

#### Example Workflow

Here's an example workflow combining these steps:

1. **Retrieve Step**:

   - **User Profile Creation**: Create a profile for the user based on past interactions (e.g., watched movies).
   - **Item Retrieval**: Use content-based filtering to retrieve a list of movies similar to those the user has liked, or collaborative filtering to find movies liked by similar users.

2. **Return Step**:
   - **Ranking**: Rank the retrieved movies based on predicted user ratings using a machine learning model.
   - **Diversity and Novelty**: Apply a diversification algorithm to ensure a mix of genres and prioritize movies the user hasn't seen.
   - **Presentation**: Display the top 10 ranked movies in the user’s recommendation feed.

### Example in Python

Here's a simplified example using collaborative filtering and ranking:

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample data: user-item interactions
user_item_matrix = pd.DataFrame({
    'User1': [5, 0, 0, 4, 0],
    'User2': [0, 3, 0, 0, 5],
    'User3': [4, 0, 0, 5, 0],
    'User4': [0, 5, 4, 0, 0]
}, index=['Item1', 'Item2', 'Item3', 'Item4', 'Item5'])

# Step 1: Retrieve - Using Item-Item Collaborative Filtering
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Get similar items to 'Item1' for 'User2'
similar_items = item_similarity_df['Item1'].sort_values(ascending=False).index
retrieved_items = [item for item in similar_items if user_item_matrix.loc[item, 'User2'] == 0]

# Step 2: Return - Rank based on similarity score and apply business rules
ranked_items = retrieved_items[:3]  # Select top 3 items as an example

# Display recommendations
print(f"Recommended items for User2: {ranked_items}")
```

#### Explanation of Code

1. **Data Preparation**: A user-item interaction matrix is created.
2. **Retrieve Step**: Item-item similarity is calculated using cosine similarity, and items similar to 'Item1' are retrieved for 'User2'.
3. **Return Step**: The retrieved items are ranked based on similarity scores, and the top items are selected for recommendation.

### TensorFlow Implementation of Content-Based Filtering

```Python
user_NN = tf.keras.models.Sequential([
   tf.keras.layers.Dense(256, activation = 'relu'),
   tf.keras.layers.Dense(128, activation = 'relu'),
   tf.keras.layers.Dense(32)
])

item_NN = tf.keras.models.Sequential([
   tf.keras.layers.Dense(256, activation = 'relu'),
   tf.keras.layers.Dense(128, activation = 'relu'),
   tf.keras.layers.Dense(32)
])

#creating the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.12_normalize(vu, axis = 1)

#creating the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_user_features))
vm = item_NN(input_user)
vm = tf.linalg.12_normalize(vm, axis = 1)

#measure the similarity of the two vector outputs
output = tf.keras.layers.Dot(axes=1)([vu,vm])

#specify the inputs and output of the model
model = Model([input_user, input_item], output)

#Specify the cost function
cost_fn = tf.keras.losses.MeanSquaredError()
```

## Principal Component Analysis(PCA)

Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction while preserving as much variability (information) as possible in a dataset. It transforms the data into a new coordinate system, such that the greatest variances by any projection of the data come to lie on the first coordinate (called the first principal component), the second greatest variances on the second coordinate, and so on.

Here's a step-by-step explanation of how PCA works:

#### 1. Standardize the Data

PCA is affected by the scale of the variables, so it's important to standardize the data. This involves subtracting the mean of each variable and dividing by the standard deviation, resulting in a dataset with a mean of 0 and a standard deviation of 1.

#### 2. Compute the Covariance Matrix

The covariance matrix is calculated to understand how the variables of the input data relate to each other. The covariance matrix is a square matrix that contains the covariance coefficients between the variables.

$[ \text{Cov}(X) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(X_i - \bar{X})^T]$

Where:

- $(X)$ is the matrix of input data.
- $(\bar{X})$ is the mean of the data.

#### 3. Compute the Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors of the covariance matrix are computed to understand the direction (eigenvectors) and magnitude (eigenvalues) of the data variance.

- **Eigenvectors**: These represent the directions of the axes where the data variance is maximized.
- **Eigenvalues**: These represent the magnitude of the variance in the direction of the eigenvectors.

#### 4. Sort Eigenvalues and Eigenvectors

The eigenvalues and their corresponding eigenvectors are sorted in descending order. This step ensures that the most important components (with the highest variance) come first.

#### 5. Select Principal Components

Select the top $(k)$ eigenvalues and their corresponding eigenvectors. These eigenvectors form the principal components. The choice of $(k)$ (number of principal components) depends on the amount of variance one wants to retain in the data.

#### 6. Transform the Data

The original dataset is transformed using the selected principal components. This is done by multiplying the original standardized data matrix by the matrix of the selected eigenvectors.

$[ Z = X \cdot W]$

Where:

- $(Z)$ is the transformed data matrix.
- $(X)$ is the original standardized data matrix.
- $(W)$ is the matrix of selected eigenvectors.

### Example

Now we gothrough a simple example with a small dataset to illustrate PCA.

#### Step 1: Standardize the Data

Assume we have the following dataset:

$[
\begin{array}{cc}
X_1 & X_2 \\
2.5 & 2.4 \\
0.5 & 0.7 \\
2.2 & 2.9 \\
1.9 & 2.2 \\
3.1 & 3.0 \\
2.3 & 2.7 \\
2.0 & 1.6 \\
1.0 & 1.1 \\
1.5 & 1.6 \\
1.1 & 0.9 \\
\end{array}
]$

Standardize each feature to have zero mean and unit variance.

#### Step 2: Compute the Covariance Matrix

Calculate the covariance matrix of the standardized data.

#### Step 3: Compute Eigenvalues and Eigenvectors

Calculate the eigenvalues and eigenvectors of the covariance matrix.

#### Step 4: Sort Eigenvalues and Eigenvectors

Sort the eigenvalues in descending order and sort their corresponding eigenvectors accordingly.

#### Step 5: Select Principal Components

Select the top $(k)$ eigenvalues and their eigenvectors.

#### Step 6: Transform the Data

Transform the original data into the new space defined by the principal components.

### Reducing Number of Features in PCA

Reducing the number of features in Principal Component Analysis (PCA) involves selecting the most significant principal components, which are the directions (eigenvectors) that capture the most variance in the data. Here's a detailed step-by-step guide on how to do this:

### Step-by-Step Guide to Reducing Features with PCA

1. **Standardize the Data**

   - Standardize the dataset so that each feature has a mean of 0 and a standard deviation of 1. This is crucial because PCA is sensitive to the variances of the original variables.

   $[
   X_{\text{standardized}} = \frac{X - \mu}{\sigma}
   ]$

   Where $(X)$ is the original data, $(\mu)$ is the mean, and $(\sigma)$ is the standard deviation of each feature.

2. **Compute the Covariance Matrix**

   - Calculate the covariance matrix to understand the relationships between different features.

   $[
   \text{Cov}(X) = \frac{1}{n-1} (X_{\text{standardized}})^T X_{\text{standardized}}
   ]$

   Where $(n)$ is the number of samples.

3. **Compute the Eigenvalues and Eigenvectors**

   - Compute the eigenvalues and eigenvectors of the covariance matrix. The eigenvectors (principal components) determine the directions of the new feature space, and the eigenvalues determine their magnitude (importance).

4. **Sort Eigenvalues and Eigenvectors**

   - Sort the eigenvalues in descending order and rearrange the eigenvectors correspondingly. This step helps in identifying the principal components that capture the most variance.

5. **Choose the Number of Principal Components (k)**

   - Select the number of principal components to keep. This can be done by looking at the explained variance ratio, which tells you how much variance each principal component explains.

   $[
   \text{Explained Variance Ratio} = \frac{\lambda_i}{\sum \lambda}
   ]$

   Where $(\lambda_i)$ is the $(i)-th$ eigenvalue.

   - Often, a cumulative explained variance threshold is chosen (e.g., 95%). You sum the explained variance ratios until you reach this threshold.

   $[
   \sum_{i=1}^{k} \frac{\lambda_i}{\sum \lambda} \geq 0.95
   ]$

6. **Project the Data onto the Selected Principal Components**

   - Form a feature vector $(W)$ by selecting the top $(k)$ eigenvectors (principal components).

   $[
   W = \left[ \text{eigenvector}_1, \text{eigenvector}_2, \ldots, \text{eigenvector}_k \right]
   ]$

   - Transform the original standardized data $(X_{\text{standardized}})$ to the new subspace $(Z)$.

   $[
   Z = X_{\text{standardized}} \cdot W
   ]$

#### Example

Assume we have a dataset $(X)$ with 5 features. Here's a simplified example to demonstrate the reduction process:

1. **Standardize the Data:**

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_standardized = scaler.fit_transform(X)
   ```

2. **Compute the Covariance Matrix:**

   ```python
   import numpy as np
   covariance_matrix = np.cov(X_standardized, rowvar=False)
   ```

3. **Compute the Eigenvalues and Eigenvectors:**

   ```python
   eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
   ```

4. **Sort Eigenvalues and Eigenvectors:**

   ```python
   sorted_index = np.argsort(eigenvalues)[::-1]
   sorted_eigenvalues = eigenvalues[sorted_index]
   sorted_eigenvectors = eigenvectors[:, sorted_index]
   ```

5. **Choose the Number of Principal Components (k):**

   ```python
   explained_variances = sorted_eigenvalues / np.sum(sorted_eigenvalues)
   cumulative_explained_variance = np.cumsum(explained_variances)
   k = np.argmax(cumulative_explained_variance >= 0.95) + 1
   ```

6. **Project the Data onto the Selected Principal Components:**
   ```python
   W = sorted_eigenvectors[:, :k]
   Z = np.dot(X_standardized, W)
   ```

After these steps, $(Z)$ is the new dataset with reduced dimensions, retaining most of the original data's variance.

### PCA In Code

> ### Pre-Defined Steps

**Optional pre-processing: Perform feature scaling**

1. "fit" the data to obtain 2(or 3) new axes(principal components) which includes mean normalisation.

2. Optionally examine how much variance is explained by each principal components.
   `explained_variance_ratio`

3. Transform(project) the data onto the new axes.
   `transform`

> ### Example in Code

```Python
X = np.array([1,1], [2,1], [3,2], [-1,-1], [-2,-1], [-3,-2])

#fitting one pricipal component
pca_1 = PCA(n_components = 1)
pca_1.fit(X)
pca_1.explained_variance_ratio_ #0.992

#Projecting the array
X_trans_1 = pca_1.transform(X)
X_reduced_1 = pca.inverse_transform(X_trans_1)
```
