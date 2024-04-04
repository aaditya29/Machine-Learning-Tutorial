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
