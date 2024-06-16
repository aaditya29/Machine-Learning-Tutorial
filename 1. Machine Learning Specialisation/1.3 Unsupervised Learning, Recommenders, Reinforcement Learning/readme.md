# Course 3- Unsupervised Learning, Recommenders, Reinforcement Learning

## Reinforcement Learning Introduction

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize some notion of cumulative reward. Here's a detailed breakdown of the key concepts and components in reinforcement learning:

### Key Concepts

1. **Agent**: The learner or decision-maker.
2. **Environment**: Everything the agent interacts with.
3. **State (s)**: A representation of the current situation of the agent.
4. **Action (a)**: A decision or move made by the agent.
5. **Reward (r)**: Immediate return received by the agent after performing an action.
6. **Policy (π)**: A strategy used by the agent to determine the next action based on the current state.
7. **Value Function (V)**: A function that estimates the expected cumulative reward of being in a state.
8. **Action-Value Function (Q)**: A function that estimates the expected cumulative reward of taking an action in a state.

### The Learning Process

The process of reinforcement learning involves the following steps:

1. **Initialization**: The agent starts with an initial policy and value function, often with arbitrary values.
2. **Interaction**: The agent interacts with the environment by taking actions.
3. **Observation**: The agent observes the resulting state and receives a reward.
4. **Learning**: The agent updates its policy and value functions based on the observed reward and state transition.

### Types of Reinforcement Learning

1. **Model-Free RL**: The agent learns directly from interactions with the environment without a model of the environment. Common algorithms include:

   - **Q-Learning**: Learns the value of actions directly.
   - **SARSA (State-Action-Reward-State-Action)**: Similar to Q-learning but updates the action-value function using the action actually taken by the agent.
   - **Deep Q-Networks (DQN)**: Uses neural networks to approximate the Q-values for complex state spaces.

2. **Model-Based RL**: The agent builds a model of the environment and uses it to make decisions. This can involve planning and learning.
   - **Dynamic Programming**: Uses a known model of the environment to solve for the optimal policy.
   - **Monte Carlo Methods**: Learn from complete episodes of experience without needing a model of the environment.

### Exploration vs. Exploitation

A fundamental challenge in reinforcement learning is balancing exploration (trying new actions to discover their effects) and exploitation (using known actions that yield high rewards). Common strategies include:

- **ε-Greedy**: With probability ε, choose a random action (exploration); otherwise, choose the best-known action (exploitation).
- **Softmax**: Select actions probabilistically, with actions having higher estimated rewards being chosen more often.

### Reward Signal

The reward signal is crucial as it guides the agent's learning. Designing the right reward signal is essential for effective learning. Rewards can be sparse or dense, and they can significantly impact the agent's behavior and learning efficiency.

### Applications

Reinforcement learning has a wide range of applications, including:

- **Game Playing**: AlphaGo, AlphaZero
- **Robotics**: Autonomous robots learning to navigate or manipulate objects.
- **Recommendation Systems**: Personalizing content recommendations.
- **Finance**: Automated trading systems.

### Mars Rover Example

Let's use the example of a Mars rover to illustrate the concepts of reinforcement learning (RL). Imagine a Mars rover that needs to navigate the Martian terrain to collect samples and return to its base station. The goal is for the rover to learn how to achieve this task efficiently through reinforcement learning.

### Key Components in the Mars Rover Example

1. **Agent**: The Mars rover.
2. **Environment**: The Martian terrain, including obstacles, sample locations, and the base station.
3. **State (s)**: The current situation of the rover, which can include its position, remaining battery life, and the locations of nearby samples and obstacles.
4. **Action (a)**: Decisions the rover can make, such as moving forward, turning left or right, stopping to collect a sample, or returning to the base station.
5. **Reward (r)**: Immediate feedback received after performing an action, such as a positive reward for collecting a sample or returning to base, and a negative reward for hitting an obstacle or running out of battery.
6. **Policy (π)**: The strategy the rover uses to decide its next action based on the current state.
7. **Value Function (V)**: The expected cumulative reward of being in a state, helping the rover evaluate long-term benefits.
8. **Action-Value Function (Q)**: The expected cumulative reward of taking a specific action in a specific state.

### The Learning Process

1. **Initialization**: The rover starts with an initial policy and value function, which might be random or based on prior knowledge.
2. **Interaction**: The rover explores the Martian terrain by taking actions.
3. **Observation**: After each action, the rover observes the new state (e.g., new position) and receives a reward (e.g., +10 for collecting a sample, -5 for hitting an obstacle).
4. **Learning**: The rover updates its policy and value functions based on the observed rewards and state transitions.

### Example Scenario

1. **Initial State**: The rover is at the base station with full battery life and no samples collected.
2. **Actions**:

   - Move forward to explore the terrain.
   - Turn left or right to navigate around obstacles.
   - Stop to collect samples when near a sample location.
   - Return to the base station to drop off collected samples and recharge.

3. **Rewards**:
   - +10 for successfully collecting a sample.
   - +5 for returning to the base station with collected samples.
   - -5 for hitting an obstacle.
   - -10 for running out of battery far from the base station.

### Exploration vs. Exploitation

The rover needs to balance exploration (trying new paths to find more samples) and exploitation (following known paths that have been successful). Strategies include:

- **ε-Greedy**: With probability ε, the rover chooses a random action to explore; otherwise, it chooses the best-known action based on its policy.
- **Softmax**: The rover probabilistically selects actions, favoring those with higher estimated rewards but still occasionally exploring.

### Updating the Policy

Using a method like Q-learning, the rover updates its action-value function $( Q(s, a) )$:

$[ Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] ]$

Where:

- $( \alpha )$ is the learning rate.
- $( \gamma )$ is the discount factor.
- $( s' )$ is the new state after taking action $( a )$.
- $( a' )$ is the next action.

### Example Episode

1. **Initial State**: Rover is at the base station.
2. **Action**: Move forward.
3. **New State**: Rover moves to a new position.
4. **Reward**: No immediate reward.
5. **Action**: Move forward and finds a sample.
6. **New State**: Near a sample.
7. **Reward**: +10 for collecting the sample.
8. **Action**: Return to the base station.
9. **New State**: At the base station.
10. **Reward**: +5 for returning with samples.

The rover updates its Q-values based on these interactions, learning the best paths and strategies over many episodes.
