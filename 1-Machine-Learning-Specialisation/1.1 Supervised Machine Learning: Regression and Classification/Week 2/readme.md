# Important Notes Of Week 2

## Multiple Linear Regression

**What is Multiple Linear Regression (MLR)?**<br>

- **Extension of Simple Regression:** Multiple linear regression is an extension of simple linear regression, where we had just one independent variable to predict our dependent variable. With MLR, we introduce multiple independent variables to explain the behavior of our dependent variable.
- **The Formula:** The core idea is represented by this equation:

  `Y = b0 + b1*X1 + b2*X2 + ... + bn*Xn + error`
  Where:

  - Y is the dependent variable (what we want to predict)
  - X1, X2, ..., Xn are the independent variables (predictors)
  - b0 is the intercept (the expected value of Y when all X's are zero)
  - b1, b2, ..., bn are the coefficients (they tell us how much a one-unit change in an independent variable affects the dependent variable)
  - error is the residual or unexplained variation

**Why Use Multiple Linear Regression?**<br>

- **Greater Predictive Power:** In most cases, a single factor can't fully explain the variation in a real-world phenomenon. Using multiple independent variables enhances your model's ability to understand and predict your dependent variable.
- **Unraveling Complex Relationships:** MLR helps you figure out which independent variables have the strongest impact on your outcome and whether certain variables interact with each other.

**For Example:**<br>
Imagine you're a big fan of superheroes and always wonder what makes someone super tall. You could use multiple linear regression to figure it out!

- **Goal:** Predict a person's height.

- **Ingredients (Independent Variables):**
  - Parents' Heights (X1): Do tall parents usually have tall kids?
  - Diet (X2): Does eating lots of healthy foods help a person grow? (We could measure this by counting healthy meals per week)
  - Sleep (X3): Does getting enough sleep matter for growing taller? (Hours of sleep per night)

**Multiple Linear Regression Recipe:**<br>

`Height = (Secret Number) + (Parent Power * Parent Height) + (Food Power * Healthy Meals) + (Sleep Power * Hours of Sleep)`

- **The Recipe Machine:** You'd need data on lots of people – their height, their parents' heights, how many healthy meals they eat, and how much sleep they get. The 'recipe machine' (MLR) would find the best 'secret numbers' to predict someone's height.

## Vectorisation

Vectorization is a programming technique where you perform mathematical operations on entire arrays or matrices at once, rather than processing individual elements using loops. It leverages highly optimized linear algebra libraries (like NumPy in Python) that are designed for efficiency and can take advantage of parallel processing capabilities in modern hardware.<br>

**Why Vectorization in Multiple Linear Regression?**<br>
Multiple linear regression involves several key computations:

- **Hypothesis calculation:** Predicting the outcome based on features and model parameters (weights).
- **Cost function calculation:** Measuring the error between predictions and true values.
- **Gradient descent:** Updating model parameters to minimize the error.<br>

Vectorization significantly improves the performance and readability of these computations:

- **Speed:** Vectorized operations are much faster than loops, especially when dealing with larger datasets. This matters a lot when you're training your model iteratively.
- **Conciseness:** Vectorization lets you express complex calculations in a few lines of code, making your implementation cleaner.
- **Parallelism:** The underlying linear algebra libraries can effectively utilize GPUs and multi-core CPUs for even greater speedups with large datasets.

### Vectorisation Implementation

Vectorization works in the main components of multiple linear regression in following way:<br>

- **Hypothesis Calculation:**

  - Non-vectorized (with loops):

  ```Python
  def predict(X, theta):
    predictions = []
    for sample in X:
        prediction = theta[0]
        for j in range(1, len(sample)):
            prediction += theta[j] * sample[j]
        predictions.append(prediction)
    return predictions
  ```

  - **Vectorized (using matrix operations):**

  ```Python
  def predict(X, theta):
    return np.dot(X, theta)
  ```

- **Gradient Descent (for optimizing parameters):**

  - **Non-vectorized:** Computation of gradients and updates using loops.

  - **Vectorized:** The entire gradient vector and parameter updates are calculated in a single matrix operation.

  #### Example with NumPy

```Python
import numpy as np

# Sample dataset
X = np.array([[1, 2, 5], [1, 3, 4], [1, 5, 2]])  # Each row is a data point with features
y = np.array([10, 15, 12])  # Corresponding target values
theta = np.array([0, 0, 0])  # Initial parameters

# Vectorized calculation of predictions
predictions = np.dot(X, theta)

# Vectorized gradient computation
gradient = np.dot(X.T, (predictions - y)) / X.shape[0]

# Vectorized parameter update (example with simple gradient descent)
learning_rate = 0.1
theta -= learning_rate * gradient
```

## Gradient Descent for Multiple Linear Regression

Gradient descent is an iterative optimization algorithm that helps us find those optimal coefficients.<br>
**Working:**

1. **Start with Random Coefficients:** We begin with an initial (random) guess for the coefficients.

2. **Cost Function:** We need a way to measure how bad our current guess is. This is where a cost function comes in. A common cost function for linear regression is the Mean Squared Error (MSE):

```
MSE = (1/N) \* Σ(yi - ŷi)^2
```

(N = number of data points, yi = actual values, ŷi = predicted values)

3. **Calculate Gradients:** The gradient is like a compass within our cost function landscape. It tells us the direction of steepest increase in the cost function. We calculate the gradients (partial derivatives) of the cost function with respect to each coefficient (b0, b1, b2, ...).

4. **Update Coefficients:** We update our coefficients in the opposite direction of the gradient, effectively moving downhill towards a lower cost value. The update looks like this:

```
bi = bi - α * (∂MSE / ∂bi)
```

- _α (alpha):_ The learning rate, a hyperparameter that controls our step size.

5. **Repeat and Converge:** We continue iterating steps 3 and 4 until we converge to a point where the cost function is minimized (or close enough). The coefficients at this point represent our best estimate for the multiple linear regression model.

**Python example demonstrating how to implement gradient descent for multiple linear regression:**

```Python
import numpy as np
import matplotlib.pyplot as plt

# Sample Data (Feel free to replace this with your own dataset)
x = np.array([[1, 2104, 5], [1, 1416, 3], [1, 1534, 3], [1, 852, 2]])  # Features
y = np.array([460, 232, 315, 178])  # Target variable

# Hyperparameters
learning_rate = 0.01
iterations = 1000

# Add a column of ones for the intercept term
x = np.hstack((np.ones((x.shape[0], 1)), x))

# Initialize parameters (weights/coefficients)
theta = np.zeros(x.shape[1])

# Gradient Descent Function
def gradient_descent(x, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for it in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        gradient = np.dot(x.T, error) / m
        theta = theta - learning_rate * gradient
        cost_history[it]  = compute_cost(x, y, theta)

    return theta, cost_history

# Cost Function (Mean Squared Error)
def compute_cost(x, y, theta):
    m = len(y)
    predictions = x.dot(theta)
    sq_error = (predictions - y) ** 2
    return (1 / (2 * m)) * np.sum(sq_error)

# Run gradient descent
final_theta, cost_history = gradient_descent(x, y, theta, learning_rate, iterations)

print("Final values of theta (coefficients):", final_theta)

# Visualize cost function convergence
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Convergence of Gradient Descent")
plt.show()

```

### Explanation:

1. **Data:** We create a simple dataset with features (e.g., size of the house, number of bedrooms) and target values (e.g., house prices).
2. **Hyperparameters:** Set the learning rate and the number of iterations.
3. **Intercept:** Add a column of ones to the feature matrix to account for the intercept term.
4. **Initialization:** Initialize the weights (theta) to zeros.
5. `gradient_descent` **Function:** This function performs the core of gradient descent calculation. It iterates, updating the weights and storing the cost in each iteration.
6. `compute_cost` **Function:** Calculates the Mean Squared Error (MSE).
7. **Running Gradient Descent:** Call the gradient_descent function to obtain the optimized coefficients (final_theta).
8. **Visualization:** Plot the cost history to see how the cost function decreased over iterations.

## Feature Scaling In Linear Regression

**What is Feature Scaling?**<br>

In machine learning, feature scaling is the process of adjusting the ranges of different features (variables) in a dataset to a common scale. This is often done to equalize the influence of features that might have vastly different numeric ranges.<br>

**Why Feature Scaling Matters for Linear Regression?**

1. **Gradient Descent Convergence:** Linear regression often uses algorithms like gradient descent to find the optimal model parameters. When features have vastly different scales, the gradient descent algorithm may take longer to converge or oscillate, potentially leading to a less optimal solution. Feature scaling helps create a smoother optimization landscape.

2. **Feature Dominance:** In linear regression, the weights (coefficients) assigned to each feature indicate their relative importance to the prediction. Without feature scaling, a feature with a large numeric range might artificially dominate the model, even if it has less predictive power than features on a smaller scale.

### Common Feature Scaling Techniques

1. **Normalization (Min-Max Scaling):**<br>

- Transform features to a specific range, usually between 0 and 1.
- Calculation: `X' = (X - X_min) / (X_max - X_min)`
- Good when you know the distribution is not Gaussian and doesn't have many outliers.

2. **Standardization (Z-score Scaling):**

- Rescales features to have a mean of zero and a standard deviation of one.
- Calculation: X' = (X - mean) / standard deviation
- Preferred when the data follows a Gaussian (bell curve) distribution or in cases where there might be outliers.

**When is Feature Scaling Especially Important?**<br>

- **Algorithms Sensitive to Scale:** Algorithms like linear regression and logistic regression are affected by the scale of features. Tree-based models are generally less sensitive to feature scaling.
- **Distance-Based Algorithms:** Algorithms that calculate distances between data points (e.g., k-Nearest Neighbors, Support Vector Machines) highly benefit from feature scaling to prevent inappropriate weighting of features based on their scale.

**Example:**<br>

```Python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load dataset
boston = load_boston()
X = boston.data
y = boston.target

# Experiment with different learning rates
for learning_rate in [0.001, 0.01, 0.1]:
    model = LinearRegression(learning_rate=learning_rate)
    model.fit(X, y)

    # ... evaluate the model's performance

```

### Checking Gradient Descent for Convergence in Feature Scaling for Linear Regression

Convergence is reached when successive updates to the model parameters result in minimal changes in the cost function. This indicates that the gradient descent algorithm has found a point that's close to a local or global minimum.<br>

#### Methods to Check Convergence

1.  **Monitoring the Cost Function:**

    - Plot the value of the cost function (e.g., Mean Squared Error) against the number of gradient descent iterations.

    - **Convergence sign:** The cost function should decrease and eventually plateau. Large changes in the cost function indicate that convergence has not been reached.

2.  **Monitoring Parameter Updates:**

    - Calculate the change in parameters between successive gradient descent iterations.

    - **Convergence sign:** When the changes in parameters become very small (below a defined threshold), it's a sign that convergence might have been reached.

3.  **Tolerance and Early Stopping:**

    - Set a tolerance value (epsilon) that defines an acceptable change in the cost function or parameters.

    - Stop gradient descent iterations when the change falls below this tolerance value.

4.  **Maximum Iterations:**

    - Define a maximum number of iterations. Gradient descent will be stopped after this limit is reached, regardless of convergence. This acts as a safeguard against very slow convergence.

#### Example (Conceptual)

Let's say we're performing linear regression with feature scaling. Here's how you could check for gradient descent convergence:

1. **Choose a feature scaling method:** (e.g., Standardization)
2. **Define convergence criteria:** (e.g., tolerance of 0.001 for cost function change)
3. **Initialize model parameters:** (randomly or strategically)
4. **Iterate through gradient descent:**
   - Calculate the gradient of the cost function.
   - Update model parameters based on the gradient and learning rate.
   - Calculate the new cost function.
   - Check if the change in cost function is below your tolerance. If yes, consider convergence reached.

### Choosing a Learning Rate for Linear Regression

1. **Trial and Error:**

   - Start with a small value (e.g., 0.01).
   - Train your model and observe the loss over training iterations (epochs).
   - If the loss decreases and converges, your learning rate is likely suitable.
   - If the loss diverges (increases or explodes), reduce the learning rate.
   - If the loss decreases very slowly, you might try a slightly larger learning rate.

2. **Grid Search:**

   - Set a range of potential learning rates (e.g., 0.001, 0.01, 0.1, 1).
   - Train models with each learning rate and evaluate their performance on a validation set.
   - Choose the learning rate that leads to the best validation performance.

3. **Learning Rate Schedulers:**

   - Utilize a scheduler that adjusts the learning rate throughout training.
   - **Common strategies include:**
     - **Step Decay:** Reducing the learning rate by a factor after a certain number of epochs.
     - **Exponential Decay:** Gradually decreasing the learning rate over time.

## Understanding Polynomial Regression

- **Beyond Straight Lines:** Linear regression is excellent for modeling relationships where the data roughly follows a straight line. But what if the relationship is curved? That's where polynomial regression comes into play. It allows us to model non-linear relationships by fitting a curved line to our data instead of a straight one.

- **The Power of Polynomials:** The essence of polynomial regression is the use of polynomial terms. A polynomial is an expression consisting of variables and coefficients, with terms combined using addition, subtraction, and multiplication.<br>
  **For example:**<br>
  `y = b0 + b1*x + b2*x^2 + b3*x^3 (Here, x^2 and x^3 are polynomial terms)`<br>

- **Degree Matters:** The degree of the polynomial determines how complex a curve we can fit.
  - Degree 2: Quadratic (parabola shape)
  - Degree 3: Cubic
  - Higher degrees: More flexible curves

### How Polynomial Regression Works

1. **Data Preprocessing:** Ensure your data is suitable for regression by handling missing values, outliers, and possibly scaling features.
2. **Feature Transformation:** Here's the core idea! We create new features by raising existing features to various powers.<br>
   **For example**, if we have a feature 'x', we might create new features like, x^2 and x^3.

3. **Model Fitting:** After creating these polynomial terms, polynomial regression actually becomes a form of linear regression. Just like in linear regression, we now find the coefficients (b0, b1, b2...) which will create a line (or curve) that best fits the relationship between our transformed features and the target variable.
4. **Prediction:** To make predictions on new data, we apply the same feature transformation (creating those powers of existing features) and then use the fitted model to find the predicted value.

### Key Considerations

- **Overfitting:** Polynomial models, especially with higher degrees, risk overfitting—fitting the training data too closely and failing to generalize well. It's essential to use techniques like regularization or cross-validation to prevent this.
- **Degree Selection:** Picking the right polynomial degree is crucial. Too low of a degree might underfit; too high could overfit. Experimentation and model evaluation are key.

#### Advantages

- **Captures non-linearity:** Can model complex relationships that simple linear regression can't.
- **Relatively simple:** Easy to understand and implement.

#### Disadvantages

- **Sensitive to outliers:** Outliers can significantly affect the fitted curve.
- **Prone to overfitting:** Choosing the right polynomial degree is critical.

### Python Implementation

```Python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Create Polynomial Features
polynomial_features = PolynomialFeatures(degree=2)

# Create a pipeline
model = Pipeline([('polynomial_features', polynomial_features),
                  ('linear_regression', LinearRegression())])

# Fit the model
model.fit(X, y)
```
