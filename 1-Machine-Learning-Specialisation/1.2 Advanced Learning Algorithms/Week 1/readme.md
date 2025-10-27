# Important Notes Week 1

## Neural Network Intuition

**Biological Neurons: The Blueprint**

- **Dendrites:** Imagine these as the input branches of a neuron. They receive signals from other neurons.
- **Soma (Cell Body):** This is the core of the neuron. It processes the incoming signals it receives.
- **Axon:** Think of this as the output line. If a neuron gets sufficiently excited by the input signals, the axon transmits its own electrical impulse.
- **Synapses:** These are the junctions where neurons connect. Signals are transmitted across the synapse using chemicals called neurotransmitters.

**Artificial Neural Networks: The Mathematical Model**

Inspired by the structure of the brain, artificial neural networks (ANNs) are comprised of these key elements:

- **Inputs:** These represent the data that enters the network (like pixels in an image).
- **Neurons (Nodes):** These are the processing units. They take inputs, multiply them by weights (which simulate the strength of a connection, similar to synapses), and apply an activation function to determine the neuron's output.
- **Outputs:** These are the final results produced by the network (e.g., the classification of an image).
- **Layers:** ANNs are often organized into layers. Information flows from input, through potentially many "hidden" layers, finally reaching the output layer.

**Key Similarities**

- **Interconnection:** Both systems are massively interconnected. Biological neurons form networks through synapses; artificial neurons connect in layers.
- **Decision-Making:** Biological neurons "fire" or don't based on the signals they receive. Artificial neurons have activation functions (like sigmoid or ReLU) that decide whether to strongly pass on a signal or not.
- **Learning:** Both systems adapt. Biological neural networks change through synaptic plasticity (strengthening or weakening connections); artificial neural networks change through adjusting weights during training.

**Key Differences**

- **Complexity:** The human brain is vastly more complex than any ANN, with billions of neurons and trillions of connections. ANNs are highly simplified in comparison.
- **Signal Type:** Biological neurons use electrical impulses and chemical neurotransmitters; ANNs primarily use numerical calculations.
- **Hardware:** The brain is organic and biological; ANNs are (currently) simulations run on digital computers.

**Intuition Building**

Think of a neural network like a giant decision-making system with many layers. Imagine each neuron as a tiny worker:

1. **Workers Receive Information:** Each gets information from the workers in the layer before it (like sensory data about the world).
2. **Workers Have Opinions:** Each worker has a 'weight' representing how important they feel their information is.
3. **Excited Workers Shout:** Each worker does a little calculation and, if excited enough, yells their opinion to the workers in the next layer.
4. **Final Decision:** The last layer's workers combine their input, like a final vote, to create the neural network's output.
5. **Learning:** If the output is wrong, all the workers are told to adjust their opinion weights slightly (backpropagation), so hopefully, they make a better decision next time.

**Key Points of the Analogy**

- **Simplified Model:** Artificial neurons are vastly simplified versions of biological neurons. They capture the basic idea of input, processing, and output but ignore many biological complexities.
- **Connectivity:** The power of both biological and artificial neural networks comes from their interconnectedness. A single neuron means little, but a network creates complex decision-making abilities.
- **Learning Mechanism:** The brain changes through synaptic plasticity; artificial neural networks learn by adjusting the weights between artificial neurons.

## Neural Network Model

### Neural Networks: The Basics

- **Artificial Neurons:** Neural networks are inspired by the biological brain. The fundamental building block is an artificial neuron, a mathematical unit that processes inputs and produces an output.
- **Layers:** Artificial neurons are organized into layers:
  - **Input Layer:** Receives the raw data (features) for the problem.
  - **Hidden Layers:** The heart of the network, performing calculations and transformations on the data, learning complex patterns. There can be one or many hidden layers.
  - **Output Layer:** Produces the final prediction or classification.

### Notations

Let's introduce standard notations to describe the elements within layers:

- **Inputs:**

  - `x_i`: The ith input feature of a single data sample.
  - `X`: Often a vector or matrix representing multiple input features

- **Weights:**

  - `w^(l)_{ij}`: The weight connecting the jth neuron in layer l-1 to the ith neuron in layer l.
  - `W^(l)`: The matrix of all weights connecting layer l-1 to layer l.

- **Biases:**

  - `b^(l)_i`: The bias associated with the ith neuron in layer l.
  - `b^(l)`: The vector of biases for layer l.

- **Activations:**

  - `z^(l)_i`: The weighted sum of inputs and bias for the ith neuron in layer l, before applying the activation function.
  - `a^(l)_i`: The output of the ith neuron in layer l after applying the activation function.

### A Simple Example: Two-Layer Network

1. **Layer 1: (Hidden Layer)**

   - **Calculations:**
     - `z^(1)_1 = w^(1)_{11} _ x*1 + w^(1)*{12} _ x_2 + ... + b^(1)_1`
     - `a^(1)\_1 = g(z^(1)_1)` (where `g` is an activation function like ReLU or sigmoid)
       ... (Similar for other neurons in the hidden layer)

2. **Layer 2: (Output Layer)**

   - **Calculations:**
     - `z^(2)_1 = w^(2)_{11} _ a^(1)*1 + w^(2)*{12} _ a^(1)_2 + ... + b^(2)_1`
     - `a^(2)_1 = g(z^(2)_1)` (This might be the final output)

### Forward Propagation Example w.r.t. Notations(Basic)

#### What is Forward Propagation?

In a neural network, forward propagation is the process of calculating the output of the network, given an input. It involves data flowing in one direction: from the input layer, through the hidden layers, and finally to the output layer.

**Steps Involved**<br>

Here's a step-by-step explanation:

1.  **Input Data:** The process starts by feeding an input data point (e.g., an image, a sentence, etc.) into the input layer.

2.  **Weighted Sum (Pre-activation):** For each neuron in the first hidden layer:

    - Multiply each input feature by its corresponding weight.
    - Sum up all the weighted inputs.
    - Add a bias term.
    - This result is called the pre-activation value (`z`).

3.  **Activation Function:** The pre-activation value (`z`) is passed through a non-linear activation function (e.g., sigmoid, ReLU, tanh). This function introduces non-linearity and allows the network to learn complex patterns. The output of the activation function is the neuron's final output (`a`).

4.  **Subsequent Hidden Layers:** The outputs of neurons in the previous layer become the inputs to the next hidden layer. Steps 2 and 3 are repeated within each hidden layer.

5.  **Output Layer:** The final hidden layer feeds into the output layer. The calculations are similar, but the output layer may use a different activation function depending on the task (e.g., softmax for classification, linear for regression).
