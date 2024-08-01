# Important Notes Week 2

## Logistic Regression as a Neural Network

Binary classification is a fundamental concept in machine learning where the task is to classify data into one of two distinct categories. Here's an overview of the basics:

### 1. **What is Binary Classification?**

Binary classification involves categorizing data points into one of two classes. For example, in a spam detection system, emails are classified as either "spam" or "not spam."

### 2. **Data Preparation**

- **Dataset**: The dataset should have features (input variables) and labels (output variables). For binary classification, the labels are typically represented as 0 and 1.
- **Feature Scaling**: Normalize or standardize features to improve model performance.
- **Train-Test Split**: Divide the dataset into training and testing sets to evaluate the model's performance on unseen data.

### 3. **Model Selection**

Several algorithms can be used for binary classification:

- **Logistic Regression**: A linear model that uses the logistic function to model a binary outcome.
- **Support Vector Machine (SVM)**: Finds the hyperplane that best separates the two classes.
- **Decision Trees**: Splits the data into subsets based on feature values to make predictions.
- **Random Forest**: An ensemble method that uses multiple decision trees to improve accuracy.
- **Neural Networks**: Especially useful for complex datasets with non-linear relationships.

### 4. **Training the Model**

- **Choose an Algorithm**: Select an appropriate algorithm based on the problem and dataset.
- **Fit the Model**: Use the training data to fit the model. The algorithm learns the relationship between features and the target variable.

### 5. **Evaluating the Model**

Several metrics can be used to evaluate the performance of a binary classification model:

- **Accuracy**: The proportion of correctly classified instances out of the total instances.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall (Sensitivity)**: The proportion of true positive predictions among all actual positives.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
- **Confusion Matrix**: A table showing the true positives, true negatives, false positives, and false negatives.

### 6. **Common Issues and Solutions**

- **Class Imbalance**: When one class is much more frequent than the other, it can skew the results. Techniques like resampling (oversampling the minority class or undersampling the majority class) or using different evaluation metrics (e.g., ROC-AUC) can help.
- **Overfitting**: When the model performs well on training data but poorly on test data. Regularization techniques, pruning (in decision trees), or using simpler models can help.
- **Underfitting**: When the model is too simple to capture the underlying patterns. Using more complex models or feature engineering can help.

### 7. **Implementation Example**

Here's a simple implementation of binary classification using logistic regression in Python with the `scikit-learn` library:

```python
# Importing libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generating synthetic data
np.random.seed(0)
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initializing and training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
```

## Computation Graph

A computation graph is a way to visualize and understand the sequence of operations performed in a mathematical computation or an algorithm. Think of it like a flowchart that shows how different pieces of data are processed step-by-step.

### Basic Concepts

1. **Nodes**: Each node in the graph represents an operation or a variable. For example, in a simple arithmetic expression like $(a + b)$, there would be nodes for $(a)$, $(b)$, and the addition operation.
2. **Edges**: Edges (or arrows) connect the nodes and show the flow of data. They indicate which outputs from one operation are inputs to the next operation.

### Simple Example

Imagine you want to compute the expression $(c = (a + b) * d)$.

1. **Input Nodes**: You have input nodes for $(a)$, $(b)$, and $(d)$.
2. **Operation Nodes**: There’s an addition node to compute $(a + b)$ and a multiplication node to compute $((a + b) * d)$.
3. **Output Node**: The final result $(c)$.

The computation graph for this expression would look like this:

```
   a       b      d
    \     /        |
     \   /         |
      +           /
       \         /
        \       /
          *
           \
            \
             c
```

### Why Use Computation Graphs?

1. **Clarity**: They make it easy to see the sequence of operations and how data flows through the system.
2. **Debugging**: You can identify where things might be going wrong by following the flow of data.
3. **Optimization**: They help in optimizing calculations by reusing intermediate results and minimizing redundant computations.

### Real-World Application

1. **Machine Learning**: Computation graphs are heavily used in machine learning frameworks like TensorFlow and PyTorch. These frameworks define complex neural networks as computation graphs.
2. **Automatic Differentiation**: They are essential for automatic differentiation, a technique used to compute gradients for optimization problems. By following the graph, the system can automatically compute derivatives.

### Step-by-Step Example with More Operations

Let's consider a more complex expression: $(e = (a + b) * (c - d))$.

1. **Input Nodes**: $(a)$, $(b)$, $(c)$, $(d)$
2. **Intermediate Operations**:
   - Addition: $(a + b)$
   - Subtraction: $(c - d)$
3. **Final Operation**: Multiplication of the results from addition and subtraction

The computation graph would look like this:

```
   a       b       c      d
    \     /         \    /
     +             -
      \           /
       \         /
        *
         \
          \
           e
```

### Summary

A computation graph is a visual representation of a series of operations in a computation. It consists of nodes (operations or variables) and edges (data flow), making it easy to understand, debug, and optimize complex computations.

## Python And Vectorisation

### What is Vectorization in Machine Learning?

Vectorization is the process of converting operations that are performed on individual elements of a dataset (like scalars) to operations performed on entire vectors or arrays. This means instead of looping through individual elements one by one, you perform the operation on the entire array in one go. This technique is especially beneficial in the context of machine learning because it can significantly speed up computations and make code more concise and easier to read.

### Benefits of Vectorization

1. **Performance**: Vectorized operations leverage highly optimized, low-level implementations in libraries like NumPy, which are usually written in C or Fortran. These implementations can make use of SIMD (Single Instruction, Multiple Data) instructions in modern CPUs.
2. **Code Simplicity**: Vectorized code is often more concise and easier to understand since it removes the need for explicit loops.
3. **Reduced Memory Overhead**: Vectorization can sometimes reduce the memory overhead by minimizing the creation of temporary arrays and intermediate variables.

### Example of Vectorization in Python

Let's consider a simple example of adding two arrays element-wise. We'll compare a non-vectorized implementation using loops with a vectorized implementation using NumPy.

#### Non-Vectorized Implementation

```python
import time

# Example data
a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]

# Initialize result array
result = [0] * len(a)

# Non-vectorized addition
start_time = time.time()
for i in range(len(a)):
    result[i] = a[i] + b[i]
end_time = time.time()

print("Non-vectorized result:", result)
print("Time taken (non-vectorized):", end_time - start_time)
```

#### Vectorized Implementation

```python
import numpy as np

# Example data
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

# Vectorized addition
start_time = time.time()
result = a + b
end_time = time.time()

print("Vectorized result:", result)
print("Time taken (vectorized):", end_time - start_time)
```

### Comparison

The vectorized version is not only more concise but also runs faster because it leverages NumPy's internal optimizations. In larger datasets, the performance difference becomes even more significant.

### Example in Machine Learning

In machine learning, vectorization is crucial for efficient computation, especially when dealing with large datasets and high-dimensional data. Let's see an example of a simple linear regression model, where we compute the predicted values and the mean squared error (MSE).

#### Non-Vectorized Linear Regression Prediction

```python
import numpy as np

# Example data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([7, 9, 11])
theta = np.array([0.5, 1.5])

# Non-vectorized prediction
predictions = []
for i in range(len(X)):
    prediction = 0
    for j in range(len(theta)):
        prediction += X[i][j] * theta[j]
    predictions.append(prediction)
predictions = np.array(predictions)

# Compute MSE
mse = np.mean((predictions - y) ** 2)
print("Non-vectorized MSE:", mse)
```

#### Vectorized Linear Regression Prediction

```python
import numpy as np

# Example data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([7, 9, 11])
theta = np.array([0.5, 1.5])

# Vectorized prediction
predictions = X.dot(theta)

# Compute MSE
mse = np.mean((predictions - y) ** 2)
print("Vectorized MSE:", mse)
```
