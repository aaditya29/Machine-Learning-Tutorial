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
