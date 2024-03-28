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
