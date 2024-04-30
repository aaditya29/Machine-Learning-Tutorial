# AdaBoost Method

Adaboost (Adaptive Boosting) is a popular ensemble learning algorithm used for classification problems. It combines the predictions of several weak classifiers (learners that perform slightly better than random guessing) to create a strong classifier. The key idea behind Adaboost is to iteratively train these weak classifiers while giving more weight to previously misclassified samples, thereby focusing subsequent learners on the hard-to-classify examples.

## Concept of Decision Stumps in AdaBoost

In the context of the Adaboost algorithm, a "stump" refers to a specific type of weak learner that is often used as the base classifier in Adaboost. Specifically, a stump is a decision tree classifier with a single decision node (also called a decision stump) that splits the feature space into just two regions based on a single feature.

Here are the key characteristics of decision stumps (or stumps) in Adaboost:

1. **Depth of 1:** A decision stump is a very shallow decision tree with a maximum depth of 1. This means that it only makes one decision based on one feature to classify the data.
2. **Binary Split:** The decision stump performs a binary split on a single feature of the dataset. It chooses a threshold value for this feature and classifies instances based on whether they fall above or below this threshold.
3. **Simple Decision Rule:** Because of its shallow nature, a decision stump is a simple classifier that implements a basic decision rule. For example, it might classify instances based on whether a specific feature value is greater than or less than a certain threshold.
4. **Efficiency:** Decision stumps are computationally efficient to train and use. They are quick to build and can be applied to large datasets without requiring extensive computational resources.

### Advantages of Using Stumps in Adaboost:

- **Efficiency:** Decision stumps are computationally efficient to train and use, making them suitable for boosting algorithms that require multiple iterations.
- **Interpretability:** Decision stumps are easy to interpret because they make decisions based on a single feature, making it clear which feature is most informative for classification.
- **Less Prone to Overfitting:** Decision stumps are simple models with low variance, which reduces the risk of overfitting especially when combined with boosting.

## Three Main Ideas Behind AdaBoost

1. **AdaBoost** combines a lot of _weak learners_ to make classifications. The weak learners are always aways stumps.
2. Some stumps get more say in the classification than others.
3. Each **stump** is made by taking the previous stump's mistake into account.
