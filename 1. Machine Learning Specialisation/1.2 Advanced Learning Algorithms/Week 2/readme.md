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
