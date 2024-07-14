# Important Notes Week 1

## Introduction to Neural Networks

Neural networks are a class of machine learning models inspired by the human brain's structure and function. They consist of interconnected layers of nodes (or neurons) that process data in a way that can learn and make decisions or predictions.

### Basic Concepts

1. **Neuron:**

   - The fundamental unit of a neural network.
   - Each neuron receives one or more inputs, processes them, and produces an output.
   - The processing typically involves summing the inputs and passing them through an activation function.

2. **Layers:**

   - **Input Layer:** The first layer that receives the initial data.
   - **Hidden Layers:** Layers between the input and output layers where computations are performed.
   - **Output Layer:** The final layer that produces the network's output.

3. **Weights and Biases:**

   - **Weights:** Parameters that transform input data within the neuron.
   - **Biases:** Additional parameters to adjust the output along with weights.

4. **Activation Function:**
   - A function applied to the neuron's input to produce the output.
   - Common activation functions include Sigmoid, Tanh, and ReLU (Rectified Linear Unit).

### How Neural Networks Work

1. **Forward Propagation:**

   - Data is fed into the input layer and passed through the network layer by layer.
   - At each neuron, the weighted sum of inputs is calculated, and the activation function is applied.
   - The process continues until the data reaches the output layer.

2. **Loss Function:**

   - Measures the difference between the network's prediction and the actual target.
   - Common loss functions include Mean Squared Error (MSE) for regression and Cross-Entropy Loss for classification.

3. **Backpropagation:**

   - A method to update weights and biases to minimize the loss.
   - Involves calculating the gradient of the loss function with respect to each weight and bias, then adjusting them in the opposite direction of the gradient.

4. **Optimization Algorithm:**
   - Algorithms like Gradient Descent, Adam, and RMSprop are used to update the weights and biases during training.

#### Example: Simple Neural Network

Imagine we want to build a neural network to recognize handwritten digits (0-9) from images. Here's a simplified example:

1. **Input Layer:**
   - Each image is 28x28 pixels, so the input layer has 784 neurons (28\*28).
2. **Hidden Layers:**
   - Let's say we use one hidden layer with 128 neurons.
3. **Output Layer:**

   - Since we have 10 digits, the output layer has 10 neurons.

4. **Activation Functions:**

   - Use ReLU for the hidden layer and Softmax for the output layer (to get probabilities for each digit).

5. **Forward Propagation:**

   - The input image is flattened into a vector of 784 values.
   - It is passed through the hidden layer, where weights and biases are applied, and ReLU is used.
   - The result is then passed to the output layer, where Softmax provides the probability distribution over the 10 digits.

6. **Loss Function:**

   - Use Cross-Entropy Loss to measure the error between the predicted and actual digits.

7. **Backpropagation and Optimization:**
   - Calculate gradients and update weights using an optimizer like Adam.

### Training the Neural Network

1. **Dataset:**
   - Use a dataset like MNIST, which contains labeled images of handwritten digits.
2. **Training Loop:**
   - Split the data into training and validation sets.
   - For each epoch (iteration over the entire dataset):
     - Perform forward propagation on the training data.
     - Compute the loss.
     - Perform backpropagation to update weights.
     - Validate the model on the validation set to monitor performance.
