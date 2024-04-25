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

## Measuring Purity in Decision Tree Learning

In decision tree learning, purity refers to the homogeneity of the target variable within a subset of data. The goal is to create splits (or nodes) in the decision tree that maximize this homogeneity, leading to more accurate and reliable predictions. There are several common measures used to quantify the purity of a node in decision trees. The choice of purity measure depends on whether the task is classification or regression.

## Entropy

Entropy is a concept used in decision tree algorithms as a measure of impurity or uncertainty within a dataset. In the context of decision trees for classification tasks, entropy is a way to quantify the randomness or disorder of the classes in a dataset. The goal of using entropy is to find splits in the data that result in subsets with the least amount of disorder, which leads to more homogeneous (pure) subsets.

### Understanding Entropy:

Entropy in the context of decision trees is based on information theory. It measures the average amount of information (or uncertainty) contained in each instance of the dataset.

For a given dataset $S$ with $n$ instances belonging to different classes ${C_1, C_2, C_3, C_4, .., C_n}$, the $Entropy(S)$ is calculated as:

$$
\text{Entropy}(t) = -\sum_{i=1}^c p(i \mid t) \log_2(p(i \mid t))
\$
$$

Where:

- $c$ is the number of distinct classes.
- $p_i$ is the proportion of instances belonging to class $C_i$ in dataset $S$.

### Interpreting Entropy:

- If a dataset $S$ is perfectly homogeneous (i.e., all instances belong to the same class), then entropy is $0$ because $p_i = 1$
  ​for one class and $p_j=0$ for all other classes.
- Higher entropy indicates higher disorder or uncertainty, meaning the classes are more mixed or evenly distributed among the instances.

## Choosing a Split: Information Gain

Information gain is a fundamental concept used in decision tree learning to quantify the effectiveness of a split in reducing uncertainty (or entropy) within a dataset. It helps in selecting the best attribute to split the data at each node of the decision tree, aiming to create more homogeneous (pure) child nodes.

### Understanding Information Gain:

Information gain measures the reduction in entropy (or increase in purity) achieved by splitting a dataset based on a particular attribute. The attribute that results in the highest information gain is chosen as the splitting criterion.

### Steps to Calculate Information Gain:

1. **Calculate Entropy before Splitting (Entropy(S)):**

   Compute the entropy of the dataset $S$ before any split. Entropy measures the impurity or randomness in the dataset.

2. **Split the Dataset based on an Attribute:**

   Split the dataset $S$ into subsets based on the values of a selected attribute $A$. Let ${v_1, v_2,..,v_n}$ be the possible values of attribute $A$, and $Sv_j$ be the subset of $S$ where attribute $A$ takes the value $v_j$.

3. **Calculate Entropy after Splitting $(Entropy Split(S, A))$.**

4. **Calculate Information Gain:**

   Finally, calculate the information gain $IG(S,A)$ for attribute $A$ as the difference between the entropy before splitting and the entropy after splitting:

   $IG(S,A) = Entropy(S) - Entropy_{split}(S,A)$

### Interpretation of Information Gain:

- Higher information gain indicates that splitting the dataset based on the selected attribute $A$ results in more homogeneous (pure) child nodes compared to before the split.

- The attribute $A$ with the highest information gain is chosen as the splitting criterion at a node in the decision tree because it maximally reduces the uncertainty (entropy) in the dataset.

## Random Forest Algorithm

The random forest algorithm is a powerful ensemble learning method used for both classification and regression tasks. It belongs to the family of tree-based ensemble methods and is particularly effective in improving the accuracy and robustness of predictive models.

### What is Random Forest?

Random Forest is an ensemble learning technique that combines multiple decision trees to create a more robust and accurate model. Instead of relying on a single decision tree, random forest leverages the wisdom of crowds by aggregating predictions from multiple individual trees.

### Key Concepts and Features of Random Forest:

1. **Decision Trees as Base Learners:**

   Random forest uses decision trees as its base learners. Each tree is trained independently on a random subset of the training data and a random subset of the features.

2. **Bagging (Bootstrap Aggregating):**

   Random forest employs a technique called bagging, where multiple trees are trained on different random samples (with replacement) drawn from the training dataset. This helps in reducing overfitting and improving the generalization of the model.

3. **Random Feature Selection:**
   In addition to training each tree on a random subset of the data, random forest also randomly selects a subset of features at each node of the tree. This introduces further randomness and diversity among the trees, which leads to more robust predictions.

4. **Voting or Averaging for Predictions:**

   For classification tasks, random forest aggregates the predictions of individual trees by using either a majority voting (for discrete classes) or averaging (for continuous predictions).<br>
   For regression tasks, the predictions of individual trees are averaged to obtain the final prediction.
