# Important Notes Week 4

## Decision Trees

### Decision Trees Learning

A decision tree is a popular machine learning algorithm used for both classification and regression tasks. It models decisions and their possible consequences in a tree-like structure. The learning process involves constructing a decision tree from a given dataset, where the tree is built recursively by splitting the data into subsets based on the values of different attributes.

Here’s a step-by-step breakdown of the learning process:

1. **Selecting the Root Node:**

The learning process starts by selecting the best attribute to use as the root node of the tree. This is typically done by calculating a metric like information gain (for classification tasks) or variance reduction (for regression tasks) for each attribute. The attribute with the highest information gain or variance reduction becomes the root node of the tree.

2. **Splitting the Dataset:**

Once the root node is selected, the dataset is split into subsets based on the values of the chosen attribute. Each subset corresponds to a different value of the attribute.

3. **Recursive Splitting:**

The process then recursively repeats on each subset created from the previous step. For each subset, a new attribute is selected to split on. This selection is again based on criteria like information gain or variance reduction.<br>
This recursive splitting continues until one of the stopping criteria is met (e.g., a maximum tree depth is reached, no further gain is achieved by splitting, or all instances in a node belong to the same class or have similar target values).

4. **Stopping Criteria:**

The decision tree construction stops when a stopping criterion is met. Common stopping criteria include:

- Maximum depth of the tree.
- Minimum number of samples required to split a node.
- Minimum number of samples in a leaf node.
- No further information gain or variance reduction achieved by splitting.

5. **Handling Categorical and Numerical Attributes:**

Decision trees can handle both categorical and numerical attributes. For categorical attributes, the dataset is split into subsets based on discrete values of the attribute. For numerical attributes, the dataset is split based on a threshold value.

6. **Tree Pruning (Optional):**

After the tree is fully grown, it may be pruned to reduce overfitting. Pruning involves removing parts of the tree that do not provide additional predictive power on validation data. This helps in improving the generalization ability of the tree.

7. **Prediction:**

Once the tree is constructed, it can be used for prediction. To predict the target value for a new instance, you traverse the decision tree from the root node down to a leaf node based on the attribute values of the instance. The predicted target value is typically the majority class (for classification) or the mean value (for regression) of the training instances in the leaf node.
