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
