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
