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
