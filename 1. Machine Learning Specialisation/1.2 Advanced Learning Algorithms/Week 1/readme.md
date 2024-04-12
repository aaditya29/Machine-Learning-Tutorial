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
