# Important Notes Of Week 3

## Classification With Logistic Regression

### Logistic Regression: The Basics

- **Not Just for Regression:** Even though it has "regression" in its name, logistic regression is a powerful classification algorithm. While linear regression predicts continuous values, logistic regression focuses on predicting the probability of something belonging to a particular category.
- **The Power of Probability:** Logistic regression helps determine the likelihood an event will occur, like:<br>
  - Whether a customer will buy a product
  - If an email is spam or not
  - The chances a patient has a certain disease
- **The Sigmoid Function:** Logistic regression uses a special S-shaped function called the sigmoid function. This function takes any real number input and 'squashes' it to a value between 0 and 1, representing a probability.

### Types of Classification in Logistic Regression

1. **Binary Logistic Regression:**

   - The Simplest Case: Deals with two possible classes (e.g., Yes/No, Pass/Fail).
   - **Example:** Predicting if a tumor is malignant or benign.

2. **Multinomial Logistic Regression:**

   - **More Than Two Possibilities:** Handles problems with three or more unordered categories.
   - **Example:** Classifying an animal image as a cat, dog, or rabbit.

3. **Ordinal Logistic Regression:**

   - **Categories with Order:** Used when there are three or more categories with a natural order to them.
   - **Example:** Classifying survey responses as "Poor", "Average", or "Excellent".

### How Logistic Regression Makes Decisions

- **The Model:** Logistic regression establishes a relationship between our features (independent variables) and the target category we're interested in (dependent variable).
- **Threshold:** We set a threshold value (often 0.5). If the predicted probability from the sigmoid function is above this threshold, the sample is classified as belonging to one class; below the threshold, it goes into the other class.

### Illustrating with an Example

Imagine we want to predict whether a student will pass an exam based on their hours of study.

1. **Data:** We gather data on past students including their study hours and whether they passed or failed.
2. **Model Training:** Logistic regression finds the best relationship between study hours and the probability of passing.
3. **New Prediction:** For a new student who studied for 5 hours, the model calculates a probability. If it's above 0.5, we predict "Pass"; otherwise, predict "Fail."

## Logistic Regression And Sigmoid Function

### What is the Sigmoid Function?

- **Mathematical Definition:** The sigmoid function is a mathematical function with a characteristic "S"-shaped curve. It's defined as<br>$\sigma(x) = \frac{1}{1 + e^{-x}}$ where 'x' is the input value and 'e' is the mathematical constant (approximately 2.718).

- **Key Properties:**

  - Output Range: The sigmoid function takes any real-numbered input and "squashes" it into a value between 0 and 1.
  - Probabilistic Interpretation: This output between 0 and 1 can be interpreted as a probability.

### How is the Sigmoid Function Used in Classification?

Classification problems in machine learning involve assigning a label or class to a data point. The sigmoid function is particularly useful in binary classification, where we're trying to distinguish between two classes.

1. **Generating Probability Scores:**

   - A machine learning model (often a logistic regression model) calculates a linear combination of the input features.
     This calculated value is passed through the sigmoid function.
   - The sigmoid function transforms this value into a probability representing the likelihood of that data point belonging to a particular class.

2. **Decision Threshold:**

   - A threshold (usually 0.5) is chosen as a decision boundary.
   - If the sigmoid output (probability) is greater than the threshold, the data point is classified into one class. If it's below the threshold, it's assigned to the other class.

#### Example

Imagine we have a model that predicts whether an email is spam or not spam.

- The model takes various features of the email as input (e.g., words in the subject line, sender address, etc.).
- It calculates a score based on these features.
- The sigmoid function transforms this score into a probability between 0 and 1 (how likely it is that the email is spam).
- If the probability is greater than 0.5, the email is classified as 'spam'; otherwise, it's classified as 'not spam'.

### Interpretation of Logistic Regression w.r.t Sigmoid Function

A logistic regression model applies the sigmoid to the familiar linear regression model as shown below:
$$ f\_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b ) $$

where

$g(z) = \frac{1}{1 + e^{-x}}$

### Decision Boundary

#### What is a Decision Boundary?

- **The Dividing Line:** In classification problems, the decision boundary is the line (or hyperplane in higher dimensions) that separates the different classes our model is trying to predict. Think of it as the fence that our model builds to distinguish between "yes" and "no", "buy" and "don't buy", or any other class labels.
- **Decision Time:** When we have a new data point, its position relative to the decision boundary determines which class the logistic regression model will predict it belongs to.

### Decision Boundary in Logistic Regression

- **Sigmoid's Role:** Sigmoid function plays a crucial role in defining the decision boundary in logistic regression. Since logistic regression outputs a probability between 0 and 1, the standard decision threshold is 0.5:
  - Probability >= 0.5: Predicted as one class
  - Probability < 0.5: Predicted as the other class
- **Linear at Heart:** In the simplest case (two features), the decision boundary in a logistic regression model will be a straight line. This is because the underlying linear combination of features is ultimately transformed by the sigmoid function.
- **Hyperplanes:** With more features, the decision boundary becomes a hyperplane in higher dimensions. This is harder to visualize, but the concept remains the same – a dividing "surface" between our classes.

#### Things to Note

- **Not Always Linear:** While the core concept in logistic regression often leads to linear decision boundaries, more complex models or datasets can create curvy and nonlinear boundaries.
- **Training Process:** The logistic regression model learns the ideal placement of this decision boundary during the training process by adjusting its coefficients.
- **Visualization:** Plotting decision boundaries is a useful way to understand the behavior of your classification model (especially in 2D cases).

## Cost Function for Logistic Regression

### What is a Cost Function?

In machine learning, a cost function is a way to measure how wrong your model is. It calculates the difference between the predicted values from your model and the actual true values (the labels).<br>
The goal of any machine learning algorithm (including logistic regression) is to minimize this cost function.

### Squared Error For Logistic Regression

For **Linear** Regression we have used the **squared error cost function**:
The equation for the squared error cost with one variable is:
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$

where
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$

### Why a Special Cost Function for Logistic Regression?

Logistic regression is used for classification tasks (e.g., predicting whether an email is spam or not, whether a tumor is benign or malignant). Here's why we can't use simple cost functions like Mean Squared Error (common in linear regression):

1. **Non-linearity:** The output of logistic regression is a probability (between 0 and 1) due to the use of the sigmoid function. This creates a non-linear relationship between the model's prediction and the true labels.
2. **Optimization Issues:** Linear cost functions like Mean Squared Error can lead to non-convex shapes when used with logistic regression. This means the optimization process could get stuck in local minima instead of finding the best possible solution.

### The Solution: Logistic Loss Function

Logistic Regression uses a loss function more suited to the task of categorization where the target is 0 or 1 rather than any number. <br>
**Loss** is a measure of the difference of a single example to its target value while the  
**Cost** is a measure of the losses over the training set

This is defined:

- $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is:

  $$
  \begin{equation}
  loss(f*{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \begin{cases} - \log\left(f*{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=1$}\\ - \log \left( 1 - f\_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=0$}
  \end{cases}
  \end{equation}
  $$

- $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value.

- $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot\mathbf{x}^{(i)}+b)$ where function $g$ is the sigmoid function.

The defining feature of this loss function is the fact that it uses two separate curves. One for the case when the target is zero or ($y=0$) and another for when the target is one ($y=1$). Combined, these curves provide the behavior useful for a loss function, namely, being zero when the prediction matches the target and rapidly increasing in value as the prediction differs from the target.

### How it Works

1. **Penalizes confident incorrect predictions:** If our model predicts 0.9 (very confident) for a positive example (y = 1), the log term log(h(x)) becomes very negative, leading to a high cost.
2. **Rewards confident correct predictions:** If our model predicts 0.9 for a true positive example, the log term log(h(x)) will be close to zero, resulting in a low cost.
3. **Convexity:** The beauty of log loss is that it creates a convex cost function. This means it has a single global minimum, making it easier to find the best solution during the optimization process.

The loss function above can be rewritten to be easier to implement.
$$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$

This is a rather formidable-looking equation. It is less daunting when you consider $y^{(i)}$ can have only two values, 0 and 1. One can then consider the equation in two pieces:  
when $ y^{(i)} = 0$, the left-hand term is eliminated:

$$
\begin{align}
loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 0) &= (-(0) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 0\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \\
&= -\log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$

and when $ y^{(i)} = 1$, the right-hand term is eliminated:

$$
\begin{align}
  loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 1) &=  (-(1) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 1\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\\
  &=  -\log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$

## Gradient Descent for Logistic Regression

1. **Cost Function:** We use the simplified cost function (log loss) explained earlier, as it works perfectly with gradient descent.

2. **Calculating the Gradient:** The gradient involves taking partial derivatives of the cost function with respect to each weight and the bias in our model:

`Gradient = [∂J/∂w1, ∂J/∂w2, ..., ∂J/∂b]`<br>
Where J is the cost function, 'w's are the weights, and 'b' is the bias. The math behind this is a bit involved, but it essentially represents the direction of steepest increase of the cost.

3. **Updating Parameters:** We adjust the weights and bias opposite the gradient:

```
New Weight = Old Weight - (Learning Rate _ Gradient)
New Bias = Old Bias - (Learning Rate _ Gradient)
```

- **Learning Rate:** This is a hyperparameter that controls the size of our steps. Too large, and we might overshoot valleys; too small, and it takes ages to converge.

## The Problem of Overfitting

### What is Overfitting?

- **Too Good to Be True:** An overfit model performs exceptionally well on the training data it's learned from, but fails to generalize well to new, unseen data.
- **Memorization, Not Learning:** The model becomes overly complex and focuses too closely on the specific details and noise present in the training data instead of learning the underlying general patterns.

### Analogy: The Perfectionist Student

Imagine a student who memorizes every word of their textbook for an exam. They might ace the test questions from the book, but if the exam contains new, slightly different questions, their answers will likely be off the mark. The student has overfit to the textbook.

### Causes of Overfitting

- **Complex Models:** Models with many parameters (like deep neural networks) have the flexibility to learn super intricate patterns, making them prone to overfitting.
- **Small Datasets:** Limited training data can prevent the model from getting a representative view of the real-world problem.
- **Noise:** If the training data is messy with errors or outliers, the model may try to account for these anomalies instead of the broader trend.
- **Prolonged Training:** Training a model for too long can lead to it getting hyper-focused on the training data's specifics.

### Recognizing Overfitting

- **Huge Gap:** A vast difference between the performance on training data and validation/test data is a strong indicator of overfitting. The model is great at 'remembering' the training set, but bad at applying its knowledge to new scenarios.

### How to Combat Overfitting

1. **Regularization Techniques:**

   - **L1 and L2 Regularization:** Add penalty terms to the cost function that discourage overly complex models.
   - **Dropout:** Randomly drop neurons during training in neural networks, forcing them to generalize better.

2. **Cross-Validation:** Divide your data into multiple folds. Train on some folds, validate on others, and rotate. This gives a better picture of how the model might perform on unseen data.

3. **Early Stopping:** Monitor performance on a validation set while training. Stop training when validation error starts to increase, preventing overly long training sessions.

4. **Data Augmentation:** Artificially create more training data variations (e.g., rotating images) to help the model learn more robust features.

4; **Simpler Models:** If possible, sometimes using a less complex model (e.g., linear regression instead of a huge neural network) can prevent overfitting from the start.

## Regularisation

The core purpose of regularization is to prevent overfitting. Remember, overfitting is when a model becomes overly complex and super-tuned to the specifics of the training data. This leads to amazing performance on training data, but dismal performance on new, unseen data.

### How Regularization Works

The central idea is to introduce a penalty term to the cost function (the function a model tries to minimize during training). This penalty discourages the model from having overly large or complex coefficients (weights).

### Key Techniques

1. **L1 Regularization (Lasso Regression):**

   - **How it Works:** Adds the absolute value of the coefficients (magnitude) to the cost function.
   - **Effect:** Encourages many coefficients to become zero, essentially performing feature selection (picking only the most important features).

2. **L2 Regularization (Ridge Regression):**

   - **How it Works:** Adds the square of the coefficients (magnitude) to the cost function.
   - **Effect:** Shrinks coefficients towards zero, but they won't become exactly zero. It tends to distribute importance across features.

3. **Elastic Net Regularization:**
   - **How it Works:** Combines the penalties of both L1 and L2.
   - **Effect:** Balances feature selection with coefficient shrinkage.

### Visual Analogy

Imagine our model's parameters like dials on a control panel. Without regularization, the model can turn them all the way up to fit the training data perfectly. Regularization adds friction, making it harder to turn those dials to extreme settings.

#### Hyperparameter: Regularization Strength

Regularization is often controlled by a hyperparameter, usually called lambda (λ) or alpha (α).

- **Large lambda:** Strong regularization, forcing the model to be simpler.
- **Small lambda:** Weak regularization, allowing more complex models.

### Benefits of Regularization

- **Improved Generalization:** Regularized models are usually better at making predictions on new data.
- **Feature Selection (L1):** Can help identify the most important features in your dataset.
- **Handles Multicollinearity:** Can improve model performance when there are highly correlated features.

### Choosing the Right Type

- **L1:** If you suspect only a few features are truly important (sparse models).
- **L2:** If all features might contribute slightly, but you want to prevent any single feature from being too dominant.
- **Elastic Net:** Good when you want the benefits of feature selection with more evenly distributed weights.

### Example

#### Scenario: Building a House Price Prediction Model

Imagine we have a dataset with:

- **Features:** Square footage, number of bedrooms, zip code, age of the house, etc.
- **Target: House price**

#### Problem: Overfitting

Without regularization, a complex model might learn that a house in zip code '12345' should always be worth $5000 more than average, simply because that pattern happened in the training data. This won't generalize well.

#### Regularization in Action

Let's apply L2 regularization (Ridge regression) to this problem:

1. **Cost Function with Regularization:** Our original cost function (like mean squared error) is augmented with a penalty based on the squares of the model's coefficients (weights).

2. **Effect:** During training, the optimization algorithm not only tries to minimize the prediction errors, but also tries to keep the weights small.

#### Example Outcomes

- **Without Regularization:** You might end up with huge, complex coefficients for some features. The model performs great on your training data but likely does poorly on new houses.

- **With Regularization:** The coefficients for various features are shrunken. Features that have less true predictive power get smaller coefficients, reducing their dominance and preventing over-reliance on minor patterns in the training data.

#### Practical Steps

1. **Split Data:** Divide your data into training, validation, and testing sets.

2. **Train with Different Regularization Strengths:** Train the same model (e.g., linear regression) with varying lambda values (the strength of regularization).

3. **Evaluate on Validation Set:** Compare the performance of each model on the validation set. The model with the best validation performance likely has the optimal amount of regularization.

4. **Final Assessment:** Once you've chosen the best regularization strength, evaluate your final model on the hold-out testing set to get an unbiased estimate of its real-world performance.
