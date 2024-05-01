# AdaBoost Method

Adaboost (Adaptive Boosting) is a popular ensemble learning algorithm used for classification problems. It combines the predictions of several weak classifiers (learners that perform slightly better than random guessing) to create a strong classifier. The key idea behind Adaboost is to iteratively train these weak classifiers while giving more weight to previously misclassified samples, thereby focusing subsequent learners on the hard-to-classify examples.

## Concept of Decision Stumps in AdaBoost

In the context of the Adaboost algorithm, a "stump" refers to a specific type of weak learner that is often used as the base classifier in Adaboost. Specifically, a stump is a decision tree classifier with a single decision node (also called a decision stump) that splits the feature space into just two regions based on a single feature.

Here are the key characteristics of decision stumps (or stumps) in Adaboost:

1. **Depth of 1:** A decision stump is a very shallow decision tree with a maximum depth of 1. This means that it only makes one decision based on one feature to classify the data.
2. **Binary Split:** The decision stump performs a binary split on a single feature of the dataset. It chooses a threshold value for this feature and classifies instances based on whether they fall above or below this threshold.
3. **Simple Decision Rule:** Because of its shallow nature, a decision stump is a simple classifier that implements a basic decision rule. For example, it might classify instances based on whether a specific feature value is greater than or less than a certain threshold.
4. **Efficiency:** Decision stumps are computationally efficient to train and use. They are quick to build and can be applied to large datasets without requiring extensive computational resources.

### Advantages of Using Stumps in Adaboost:

- **Efficiency:** Decision stumps are computationally efficient to train and use, making them suitable for boosting algorithms that require multiple iterations.
- **Interpretability:** Decision stumps are easy to interpret because they make decisions based on a single feature, making it clear which feature is most informative for classification.
- **Less Prone to Overfitting:** Decision stumps are simple models with low variance, which reduces the risk of overfitting especially when combined with boosting.

## Three Main Ideas Behind AdaBoost

1. **AdaBoost** combines a lot of _weak learners_ to make classifications. The weak learners are always aways stumps.
2. Some stumps get more say in the classification than others.
3. Each **stump** is made by taking the previous stump's mistake into account.

## Gini Index and AdaBoost

The Gini index, or Gini impurity, is a measure of node impurity used in decision tree algorithms, including those used as weak learners within the Adaboost algorithm. It quantifies the likelihood of incorrectly classifying a randomly chosen element in a dataset if it were randomly labeled according to the distribution of labels in the node.

### Gini Index Definition:

For a node $t$ in a decision tree that contains samples $S$, let:

- $p(i|t)$ be the proportion of samples in node $t4 that belong to class $i4 (where $i$ ranges over all classes).

The Gini index $G(t)$ for node $t$ is calculated as:<br>

$G(t) = 1- \sum_{i=1}^{C}[p(i|t)]^2$<br>

Where $C$ is the number of classes.

### Interpretation

- $G(t) = 0$ when all samples in node $t$ belong to the same class, indicating perfect purity.
- $G(t) = 1$ when the samples in node $t$ are evenly distributed across all classes, indicating maximum impurity.

### How Gini Index is Used in Decision Trees (and Adaboost):

- **Splitting Criteria:** When building a decision tree, the Gini index is used to evaluate candidate splits. The goal is to find the split that minimizes the Gini impurity in the resulting child nodes.
- **Node Impurity:** Decision trees recursively split nodes to maximize the reduction in Gini impurity at each step, aiming to partition the data into subsets that are increasingly homogeneous with respect to the target variable.

### Building a Stump with GINI Index

To build a decision stump (also known as a weak learner) using the Gini index as the criterion within the Adaboost algorithm, we can use the `DecisionTreeClassifier` from scikit-learn and specify the `criterion` parameter as `gini`. The decision stump will then be trained to minimize the Gini impurity when making splits on the data.

#### 1. Import Required Libraries

```Python
from sklearn.tree import DecisionTreeClassifier
```

#### 2. Prepare Data

We'll need to have our training data `X_train` (features) and `y_train` (labels) ready.

#### 3. Initialize and Train the Decision Stump

We use `DecisionTreeClassifier` to initialize a decision stump with the Gini index criterion. Setting `max_depth=1` to ensure that the stump only makes a single split.

```Python
# Initialize decision stump (weak learner) with Gini index
stump = DecisionTreeClassifier(criterion='gini', max_depth=1, random_state=42)

# Train the decision stump on the training data
stump.fit(X_train, y_train)

```

#### 4. Interpret the Decision Stump

After training, we can interpret the decision stump by examining its structure and the feature that it used for splitting.

```Python
# Print the feature importance of the decision stump
feature_importance = stump.feature_importances_
print("Feature Importance:", feature_importance)

# Print the selected feature for splitting
selected_feature_index = stump.tree_.feature[0]  # Assuming only one feature is used for splitting
selected_feature_name = feature_names[selected_feature_index]  # Replace `feature_names` with your feature names
print("Selected Feature for Splitting:", selected_feature_name)

```

#### Full Example

```Python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize decision stump (weak learner) with Gini index
stump = DecisionTreeClassifier(criterion='gini', max_depth=1, random_state=42)

# Train the decision stump on the training data
stump.fit(X_train, y_train)

# Evaluate the decision stump
accuracy = stump.score(X_test, y_test)
print("Decision Stump Accuracy:", accuracy)

# Print the feature importance of the decision stump
feature_importance = stump.feature_importances_
print("Feature Importance:", feature_importance)

# Print the selected feature for splitting
selected_feature_index = stump.tree_.feature[0]
selected_feature_name = f"Feature {selected_feature_index + 1}"
print("Selected Feature for Splitting:", selected_feature_name)

```

### Role of Gini Index in Adaboost with Decision Stumps:

In Adaboost, decision stumps (shallow decision trees with a single split) are commonly used as weak learners. Each decision stump is trained to minimize the Gini impurity when making a split on a selected feature, with the objective of capturing the most informative split to improve classification performance.

During the training of Adaboost:

- Each iteration focuses on training a new decision stump that minimizes the weighted error (misclassification rate) based on the current distribution of sample weights.
- The decision stump's ability to make effective splits (minimizing Gini impurity) on the training data contributes to the overall performance improvement of the Adaboost ensemble.

### Advantages of Gini Index in Adaboost:

- **Simple and Efficient:** Gini index provides a straightforward measure of node impurity, making it computationally efficient for evaluating potential splits during decision tree training.
- **Robustness**: Minimizing Gini impurity helps in creating decision stumps that focus on the most discriminative features, leading to effective weak learners in the boosting process.

## Steps in AdaBoost

Here's how the sample weights are updated in the AdaBoost algorithm step-by-step:

### 1. Initialize Sample Weights:

Start by assigning equal weights to all training samples. The initial weight for each sample $i$ is $w_i^{(1)} = \frac {1}{N}, $ where $N$ is the total number of training samples.

### 2. Train Weak Learner:

For each iteration $t = 1, 2, ..., T$ where $T$ is the number of iterations or base learners):

- Train a weak classifier $h_t{(x)}$ on the training data using the current sample weights $w^{(t)} = (w_1^{(t)}, w_2^{(t)}, ..., w_N^{(t)}).$

### 3. Compute Weighted Error:

Compute the weighted error $\epsilon_t$ of the weak classifier $h_t$ on the training set:

$$\epsilon_t = \sum_{i=1}^{N} w_i^{(t)}.1$$

$where$<br>
$y_i$ is the true label of sample $i$, $h_t(x_i)$ is the prediction of $h_t$ for sample $i$, and $1$ is the indicator function.

### 4. Compute Classifier Weight:

Calculate the classifier weight $\alpha_t$ using the formula:<br>
$\alpha_t = \frac{1}{2}\log\frac{1 - \epsilon_t}{\epsilon_t}$

### 5. Update Sample Weights:

Update the sample weights $w_i^{(t)}$ for the next iteration $t+1$ based on whether each sample was correctly or incorrectly classified by $h_t$:<br>
$w_i^{(t+1)} = w_i^{(t)} . \exp((-\alpha_t. y_i.h_t(x_i))$

$Where$<br>

- $y$ is the true label of sample $i$ (either -1 or +1).
- $h_t(x_i)$ is the prediction of $h_t$ for sample $i$ (either -1 or +1).
- If $h_t(x_i) = y_i$ (correct prediction) , $\exp(\alpha_t)$ will decrease $w_i^{(t)}$ otherwisse $\exp(\alpha_t)$ will increase $w_i^{(t)}.$

### 6. Normalize Sample Weights:

Normalize the updated sample weights so that they sum up to 1:
$w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^{N} w_j^{(t+1)}}$

### Python Implementation

```Python
import numpy as np

# Example update step
def update_sample_weights(w_prev, alpha, y_true, y_pred):
    """
    Update sample weights based on AdaBoost algorithm.

    Parameters:
    - w_prev: Array of previous sample weights
    - alpha: Weight of the weak classifier
    - y_true: True labels of training samples
    - y_pred: Predicted labels of training samples by the weak classifier

    Returns:
    - w_next: Updated sample weights for the next iteration
    """
    incorrect_mask = (y_true != y_pred)  # Mask of incorrectly classified samples
    exponent = alpha * incorrect_mask  # Exponent for weight update

    w_next = w_prev * np.exp(exponent)  # Update weights
    w_next /= np.sum(w_next)  # Normalize weights

    return w_next

# Example usage:
# Assuming alpha_t, y_true, y_pred, and w_prev are given
w_next = update_sample_weights(w_prev, alpha_t, y_true, y_pred)

```

$Where$

- `w_prev` is an array containing the sample weights from the previous iteration.
- `alpha_t` is the weight of the current weak classifier $h_t$
  ​
- `y_true` is the true labels of the training samples.
- `y_pred` is the predicted labels of the training samples by the weak classifier $h_t$.

The `update_sample_weights` function updates the sample weights based on the AdaBoost algorithm's weight update formula, and then normalizes the weights to ensure they sum up to 1 for the next iteration.
