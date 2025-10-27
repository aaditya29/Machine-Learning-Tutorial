# Important Notes Week 3

## Evaluation and Choosing a Machine Learning Model

### Train/Test Procedure for Classification Problem

The train/test procedure is a fundamental technique used in machine learning to assess the performance of classification models. It involves splitting your dataset into two subsets: one for training the model (the training set) and the other for evaluating its performance (the test set). This approach helps to estimate how well our model is likely to perform on unseen data.<br>

Here's how we can implement this procedure step-by-step for a classification problem:

1. ### **Dataset Splitting:**

- **Splitting the Data:** Start by dividing our dataset into two separate sets: the training set and the test set. Typically, the data is divided randomly, with a certain percentage allocated to each set. Common splits include 70-30% or 80-20% for training and testing, respectively.

2. ### Training Phase:

- **Training the Model:** Use the training set to train your classification model. This involves feeding the model with input features and their corresponding labels (i.e., the target variable). The model learns to identify patterns and relationships in the data during this phase.

3. ### Testing Phase:

- **Evaluating the Model:** Once the model is trained, use the test set to evaluate its performance. The test set contains data that the model has not seen during training, simulating how the model would perform on unseen data.

4. ### Performance Evaluation:

- **Calculating Metrics:** Apply your trained model to the test set to make predictions. Compare these predictions against the true labels (ground truth) from the test set. Use evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix to assess the model's performance.

5. ### Cross-Validation:

- **Cross-Validation:** In addition to a single train/test split, you can perform cross-validation to ensure robustness of your model evaluation. Techniques like k-fold cross-validation split the data into multiple subsets (folds) and iteratively use different subsets for training and testing.

## Model Selection and Training/Cross Validation/Test Sets

Model selection and training, along with the use of cross-validation and test sets, are crucial steps in developing machine learning models. These steps ensure that your model performs well, generalizes to new data, and avoids overfitting.<br>
Let's break down each component:

### Model Selection and Training:

1. **Model Selection:**

- Choose the type of model (e.g., linear regression, decision tree, support vector machine, neural network) based on the nature of our problem (regression, classification, etc.) and the characteristics of our data.
- Consider the complexity of the model and its interpretability.
- Decide on the algorithm and its hyperparameters (e.g., learning rate, number of layers in a neural network) that will be used to train the model.

2. **Training the Model:**

- Split our labeled dataset into two main subsets: training set and test set (holdout set).
- Use the training set to train (fit) the chosen model on our data.
- During training, the model learns the patterns and relationships in the training data.

### Cross-Validation:

Cross-validation is a technique used to assess how well a model will generalize to new, unseen data. It involves splitting the dataset into multiple subsets (folds) and training the model on different combinations of these subsets.

1. **k-Fold Cross-Validation:**

- Split the dataset into k subsets (folds) of approximately equal size.
- Train the model k times, each time using a different fold as the test set and the remaining folds as the training set.
- Compute the average performance metric (e.g., accuracy, F1-score) across all k iterations to evaluate the model's performance.
- Common choices for k are 5 or 10, but this can vary depending on the dataset size and computational resources.

2. **Benefits of Cross-Validation:**

- Provides a more reliable estimate of model performance compared to a single train/test split.
- Helps in tuning hyperparameters (e.g., regularization strength) by selecting the best parameter values that maximize performance across multiple folds.
- Reduces the risk of overfitting to a particular train/test split.

### Test Set:

The test set is a separate portion of your dataset that is not used during model training or hyperparameter tuning. It serves as a final evaluation to estimate the model's performance on unseen data.

1. **Role of the Test Set:**

- Assess the model's performance on new, unseen data to estimate its real-world performance.
- Validate that the model has not overfit the training data.
- Compare the performance of different models to make a final selection.

2. **Best Practices:**

- Keep the test set completely independent until the final evaluation.
- Use the test set sparingly; it should only be used once to avoid bias in model assessment.

## Diagnosing Bias and Variance

Diagnosing bias and variance in machine learning is a fundamental concept that helps in understanding the performance of a learning algorithm and guides improvements to achieve better predictive models.

Let's break down these concepts step by step:

### 1. Understanding Bias and Variance:

- **Bias (Underfitting):** Bias refers to the error introduced by approximating a real-world problem with a simplified model. A high bias model is overly simplistic and fails to capture the underlying patterns of the data. It often leads to underfitting, where the model is unable to learn the complexities of the data and performs poorly on both training and unseen data.
- **Variance (Overfitting):** Variance refers to the model's sensitivity to small fluctuations in the training data. A high variance model fits the training data very well but fails to generalize to new, unseen data. This is known as overfitting, where the model learns noise from the training data rather than the actual signal, resulting in poor performance on new data.

### 2. Bias-Variance Tradeoff:

There's often a tradeoff between bias and variance. Increasing model complexity typically reduces bias but increases variance, and vice versa. The goal is to find the right balance where the model generalizes well to unseen data.

### 3. Diagnosing Bias and Variance:

- **Training Error vs. Validation Error:**

  - **High Bias (Underfitting):** Both training and validation errors are high and similar. This suggests that the model is too simple to capture the underlying patterns in the data.
  - **High Variance (Overfitting):** There is a large gap between the training error (low) and validation error (high). This indicates that the model is fitting too closely to the noise in the training data and not generalizing well.

- **Learning Curves:**

  - **Bias:** Learning curves for both training and validation sets converge at a high error rate.
  - **Variance:** Learning curves show a significant gap between the training and validation error, with the training error much lower than the validation error.

- **Model Complexity:**

  - **Bias:** If a more complex model (e.g., increasing polynomial degree in regression) does not significantly reduce training or validation error, it suggests high bias.
  - **Variance:** If the validation error starts increasing while the training error decreases with increasing model complexity, it indicates high variance.

### 4. Addressing Bias and Variance:

- **High Bias (Underfitting):**
  - **Solution:** Increase model complexity (e.g., use a more powerful model, add more features).
- **High Variance (Overfitting):**
  - **Solution:** Decrease model complexity (e.g., regularization, feature selection), increase training data, or use techniques like dropout (in neural networks) to prevent overfitting.

### 5. Cross-Validation:

Cross-validation techniques like k-fold cross-validation can help in diagnosing bias and variance by providing estimates of both training and validation errors across multiple subsets of the data.

## Regularization and Bias/Variance

Regularization techniques add a penalty term to the model's loss function. The purpose of this penalty is to discourage the model from fitting the training data too closely, thus reducing variance. The type and strength of regularization also impact the model's bias.

### Common Regularization Techniques:

1. **L2 Regularization (Ridge Regression):**

- **Effect on Bias:** L2 regularization adds a penalty proportional to the square of the magnitude of the coefficients (weights) to the loss function. This encourages smaller weights, which can reduce the complexity of the model.
- **Effect on Variance:** By penalizing large weights, L2 regularization helps in reducing variance. It prevents the model from learning intricate details from the noise in the training data.

2. **L1 Regularization (Lasso Regression):**

- **Effect on Bias:** L1 regularization adds a penalty proportional to the absolute value of the coefficients to the loss function. This encourages sparsity in the model (i.e., some weights become exactly zero), which can lead to increased bias.
- **Effect on Variance:** L1 regularization can also reduce variance by preventing the model from overfitting to noisy features. However, it tends to be more aggressive in feature selection compared to L2 regularization.

### Impact on Bias-Variance Tradeoff:

- **Increasing Regularization Strength:**

  - As the regularization strength increases (e.g., by increasing the regularization parameter λ), the model's complexity decreases.
  - **Effect on Bias:** Higher regularization typically leads to higher bias because the model becomes more simplified.
  - **Effect on Variance:** Higher regularization reduces variance by discouraging complex models that fit the training data too closely.

  ### Choosing the Right Regularization Strength:

  The optimal regularization strength (λ) depends on the specific dataset and the complexity of the problem:

  - **Too Low λ:** May lead to overfitting (high variance), especially with complex models.
  - **Too High λ:** May lead to underfitting (high bias) as the model becomes too simplistic.

  ### Practical Implementation:

  - Cross-validation techniques are often used to tune the regularization parameter (λ) and find the optimal balance between bias and variance.
  - Regularization is a crucial tool in model development to create models that generalize well to unseen data and are robust against overfitting.

## Establishing a Baseline Level of Performance

Establishing a baseline level of performance with respect to bias and variance is an important initial step in evaluating and improving machine learning models. This baseline provides a starting point for comparison when experimenting with different algorithms, model architectures, or hyperparameters.

Here's how we can establish a baseline level of performance:

### 1. Choose a Simple Model:

Start with a simple and straightforward model to establish the baseline. For example:

- **Linear Regression:** Use linear regression for regression tasks.
- **Logistic Regression:** Use logistic regression for binary classification tasks.
- **Naive Bayes:** Use Naive Bayes for text classification tasks.
- **Decision Trees:** Use decision trees for classification or regression tasks.

### 2. Split the Data:

Split your dataset into training and testing sets. Common splits are 70-30 or 80-20, where the larger portion is used for training.

### 3. Train the Model:

Train the chosen simple model using the training dataset.

### 4. Evaluate on Training Data:

Evaluate the model's performance on the training dataset. Calculate:

- **Training Error (Bias):** This is the error (e.g., mean squared error for regression, cross-entropy loss for classification) obtained when the model predicts on the training data itself. A high training error indicates high bias (underfitting).

### 5. Evaluate on Testing Data:

Evaluate the model's performance on the testing dataset (which the model has not seen during training). Calculate:

- **Testing Error (Variance):** This is the error obtained when the model predicts on the testing data. A significant difference between the training error and testing error indicates high variance (overfitting).

### 6. Analyze Bias and Variance:

Compare the training error (bias) and testing error (variance):

- **High Training Error, High Testing Error:** Indicates high bias (underfitting).
- **Low Training Error, High Testing Error:** Indicates high variance (overfitting).

## 7. Record Baseline Metrics:

Record key metrics such as:

- Training Error
- Testing Error
- Accuracy (for classification tasks)
- Mean Squared Error (for regression tasks)

### 8. Iterative Improvement:

Use the baseline performance as a reference point to iterate and improve the model:

- If bias is high (underfitting), consider using a more complex model, adding more features, or tuning hyperparameters.
- If variance is high (overfitting), try regularization techniques, feature selection, or increasing training data.
