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
