# Important Notes Week 2

## Neural Network Training

### Training a Network in TensorFlow

1. **Create The Model**

```Python
import tensorflow as tf
#importing the Sequential model type from the Keras library
#Sequential models are a standard way to build linear stacks of neural network layers
from tensorflow.keras import Sequential
#This imports the Dense layer type from Keras
#A Dense layer is a fully-connected layer, meaning all neurons in one layer are connected to all neurons in the next layer
from tensorflow.keras.layers import Dense

#defining a Sequential model in Keras
model = ([
    #creates the first Dense layer with 25, 15 and 1 neurons
    Dense(units = 25, activation = 'sigmoid'),
    Dense(units = 15, activation = 'sigmoid'),
    Dense(units = 1, activation = 'sigmoid'),
    ])
```

2. **Loss And Cost Functions**

```Python
#This line specifically imports the BinaryCrossentropy loss function from the tensorflow.keras.losses module.
#Binary Crossentropy is particularly suited for binary classification problems
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss = BinaryCrossentropy)# This line compiles our previously defined Keras model
"""
Here we initiate the training process of our neural network.
X: Your input data (features).
Y: Your target labels (the true values we want to predict).
epochs=100: The model will train for 100 epochs.
"""
model.fit(X,Y, epochs=100)
```

## Activation Functions

### What is an Activation Function?

In artificial neural networks, an activation function determines the output of a neuron. It introduces non-linearity, allowing the network to learn complex patterns in data.

### What is ReLU?

The Rectified Linear Unit is one of the most popular activation functions. It's incredibly simple:

- If the input (x) is negative, ReLU outputs 0.
- If the input (x) is positive, ReLU outputs the input directly (x).
- **Mathematical Expression:** `ReLU(x) = max(0, x)`

### Why is ReLU so Popular?

- **Computationally Efficient:** ReLU is very simple to calculate, making training of neural networks faster.
- **Helps with Vanishing Gradients:** Traditional activation functions like sigmoid and tanh can suffer from "vanishing gradients" during training, especially in deep networks. This means updates to the network's weights become very small, hindering learning. ReLU's gradient is either 0 or 1, mitigating this problem.
- **Sparsity:** ReLU can output true zeros, leading to a sparse network representation. This can sometimes provide a degree of computational efficiency.

### Choosing Activation Functions

1. **Problem Type:**

   - **Regression:** Linear activation functions are often suitable for the output layer in regression problems.
   - **Classification:**
     - Binary Classification: Sigmoid (output layer).
     - Multi-class Classification: Softmax (output layer).
     - Complex Patterns: ReLU and its variants are popular for hidden layers for learning deeper and more intricate representations.

### Common Practices

- **Hidden Layers:**

  - **Start with ReLU:** It's generally an excellent first choice due to its mitigation of vanishing gradients.
  - **For Sparsity:** If sparse representations are desired, ReLU can aid in this.
  - **Addressing Dying ReLU:** Experiment with leaky ReLU or PReLU if you encounter the dying ReLU problem.

- **Output Layers:**

  - **Regression:** Often linear (no activation) for unconstrained output.
  - **Binary Classification:** Sigmoid (values between 0 and 1, like probabilities).
  - **Multi-Class Classification:** Softmax (produces probabilities for each class).

**For Example:**<br>

```Python
from tensorflow.keras.layers import Dense

model = ([
    #creates the first Dense layer with 25, 15 and 1 neurons
    #with relu and sigmoid activations
    Dense(units = 25, activation = 'relu'),
    Dense(units = 15, activation = 'relu'),
    Dense(units = 1, activation = 'sigmoid'),
    ])
```

### Need Of Activation Functions

Activation functions are a critical component in neural networks because they introduce non-linearity. This is essential for a few reasons:

- **Complex learning:** Without activation functions, a neural network would simply be a stack of linear regression models. This limits the network's ability to learn complex relationships between inputs and outputs, which is what makes neural networks so powerful for tasks like image recognition and speech translation.

- **Real-world data:** Real-world data often has non-linear relationships. For example, as the temperature increases, the number of ice creams sold might not increase at a constant rate, but rather accelerate at a certain point. Activation functions allow neural networks to capture these non-linear patterns.

- **Learning from errors:** During training, neural networks adjust their internal weights based on the difference between their predictions and the actual outputs. Activation functions help ensure these adjustments can propagate backward through the network (backpropagation) efficiently, allowing the network to learn from its mistakes.

## Softmax Regression

Softmax is a mathematical function often used in multiclass classification tasks to convert a vector of raw scores (also called logits) into a probability distribution. It is particularly useful when you want to model the probabilities of each class as an output of a neural network or a machine learning model.

### Understanding Softmax Function:

1. **Input to Softmax:**
   Suppose you have a vector of raw scores or logits $z = (z_1, z_2, z_3,..,z_n)$ where $z_i$ is the score/logit corresponding to class $i.$ These logits can be the output of the last layer of a neural network before applying the softmax.

2. **Softmax Calculation:**
   Softmax function computes the probability $p_i$ for each class $i$ as follows: $\text{Probability of class } i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$

In this formula:

- $z_i$ represents the score or output for class $i$
- $e$ is the base of the natural logarithm (Euler's number),
- **K** is the total number of classes.

3. **Interpretations:**

- The softmax function ensures that each output $p_i$ (probability of class $i$) is between 0 and 1.
- The denominator of $\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$ normalizes the scores, turning them into probabilities that sum up to 1.

### Benefits of Softmax in Multiclass Classification:

- **Interpretability:** Softmax converts raw scores into probabilities, making the model's predictions more interpretable.

- **Training Objective:** Softmax is often used in conjunction with the cross-entropy loss function, which is suitable for training models in multiclass classification tasks.

### Softmax Regression in Layman Terms

Imagine we have a big box of colored marbles, and we want to figure out which color is the most popular. We have marbles of different colors like red, blue, green, and yellow. Now, each color represents a different type of thing we're trying to identify, like different types of fruits or animals.<br>

Softmax is like a magic math trick that helps us decide which type of thing (or marble color) is the most likely for each item we're looking at. Let's break it down step by step:

1. **Assigning Numbers:** First, we count how many marbles we have for each color. For example, we count 5 red, 3 blue, 2 green, and 4 yellow marbles.

2. **Making Numbers Easier to Compare:** Softmax helps to turn these numbers into something called probabilities, which are easier to compare. Probabilities tells us how likely each color (or type of thing) is.

3. **Softmax Formula:** We use a special formula called softmax to do this. It's like a machine that takes the number of marbles for each color and spits out the probabilities.

Here's the formula for one color (like red):
$$\text{Probability of Red } = \frac{e^{NumberOfRedMarbles}}{e^{NumberOfRedMarbles} + e^{NumberOfBlueMarbles} + e^{NumberOfGreenMarbles} + e^{NumberOfYellowNumbers}}$$

4. **Getting the Results:** After we use softmax, we get the probabilities for each color. This tells us which color (or type of thing) is the most likely for each item.

Hence, softmax is like a magical way to turn counts of marbles (or things) into nice, understandable probabilities.

### Neural Network with Softmax Output:

In a neural network for multi-class classification using softmax:

1. **Forward Propagation:**

- The input data is propagated through the network from the input layer, through the hidden layers, to the output layer.
- Each node in the network applies a linear transformation (weighted sum of inputs) followed by a non-linear activation function (like ReLU, sigmoid, etc., in hidden layers).
- The output layer uses the softmax activation function to produce a probability distribution across all classes. The output of the softmax layer represents the predicted probabilities for each class.

2. **Loss Calculation:**

Typically, the loss function used with softmax output is the categorical cross-entropy loss. This loss measures the difference between the predicted probability distribution and the actual distribution (one-hot encoded vector representing the true class).<br>
The loss function is defined as:
$-{\sum_{i=1}^{K}y_i \log(\hat{y}_i)}$

$Where$

- y is the true distribution (one-hot encoded vector of the actual class).
- $\hat{y}$ the predicted distribution (output of the softmax layer).
