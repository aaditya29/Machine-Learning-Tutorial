# Important Notes Week 3

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

### Return in Reinforcement Learning

In reinforcement learning (RL), the concept of **Return** is central to understanding how agents evaluate and optimize their behavior over time. The Return is essentially the total accumulated reward that an agent aims to maximize, often considering future rewards with a certain discount to reflect their delayed nature. Let's break down the concept:

### Definition of Return

The Return, typically denoted as $( G_t )$, is the total reward the agent receives from time step $( t )$ onwards. There are two main forms of Return:

1. **Finite Horizon Return**: When considering a finite number of future steps $(horizon ( T ))$, the Return is the sum of rewards up to that horizon.
2. **Infinite Horizon Return**: When considering an infinite number of future steps, the Return includes rewards extending indefinitely into the future.

### Mathematical Formulation

The Return at time step $( t )$, $( G*t )$, is defined as:<br>

$[ G_t = R*{t+1} + R*{t+2} + R*{t+3} + \dots ]$

#### Discounted Return

To handle the practical issue that future rewards are often less certain or less valuable than immediate rewards, a discount factor $( \gamma ) (where ( 0 \leq \gamma \leq 1 ))$ is introduced.
The discounted Return is given by:

$[ G*t = R*{t+1} + \gamma R*{t+2} + \gamma^2 R*{t+3} + \dots ]$

In a more compact form:
$[ G*t = \sum*{k=0}^{\infty} \gamma^k R\_{t+k+1} ]$

Here:

- $( R\_{t+k+1} )$ is the reward received $( k+1 )$ time steps after time $( t )$.
- $( \gamma )$ is the discount factor, which reduces the weight of future rewards.

### Importance of Return in RL

1. **Evaluating Policies**: The Return is used to evaluate how good a particular policy (a strategy for choosing actions) is. A policy that leads to higher Returns is considered better.
2. **Training Objectives**: Many RL algorithms aim to maximize the expected Return. For example, Q-learning updates the action-value function based on expected future Returns.

### Value Functions and Return

1. **State-Value Function ($( V(s) )$)**: The expected Return when starting from state $( s )$ and following policy $( \pi )$.

   $[ V^\pi(s) = \mathbb{E}\_\pi [G_t \mid S_t = s] ]$

2. **Action-Value Function ($( Q(s, a) )$)**: The expected Return when starting from state $( s )$, taking action $( a )$, and then following policy $( \pi )$.

   $[ Q^\pi(s, a) = \mathbb{E}\_\pi [G_t \mid S_t = s, A_t = a] ]$

### Example: Mars Rover

Let's revisit the Mars rover example to illustrate the concept of Return:

1. **Immediate Rewards**:

   - +10 for collecting a sample.
   - -5 for hitting an obstacle.
   - +5 for returning to the base station with samples.

2. **Discounted Return Calculation**:

   Suppose the rover is at state $(s*t)$ and the sequence of rewards is $(R*{t+1} = 10)$, $( R*{t+2} = -5 )$, $( R*{t+3} = 5 )$.<br>
   With a discount factor $( \gamma = 0.9 )$, the Return $(G_t)$ would be:

   $[
   G_t = 10 + 0.9 \times (-5) + 0.9^2 \times 5
   = 10 - 4.5 + 4.05
   = 9.55
   ]$

#### Summary

The Return in reinforcement learning is a measure of the total accumulated reward from a given time step onward. Discounted Return helps to prioritize immediate rewards over distant ones, reflecting the uncertainty or lesser value of future rewards. Understanding and maximizing the Return is crucial for training effective RL agents, as it directly relates to the long-term success of their policies.

## State-Action Value Function

The state value function is a fundamental concept in reinforcement learning (RL) that quantifies the expected future rewards an agent can obtain from a given state. Here's an overview of the concept:

### 1. **State Value Function (V)**

The state value function, often denoted as $( V(s) )$, represents the expected return (cumulative future rewards) when starting from state $( s )$ and following a particular policy $( \pi )$.

#### Definition:

$[ V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_t = s \right] ]$

- $( \mathbb{E}_\pi )$ denotes the expected value given that the agent follows policy $( \pi )$.
- $( \gamma )$ (gamma) is the discount factor, $( 0 \leq \gamma \leq 1 )$, which determines the importance of future rewards.
- $( R\_{t+1} )$ is the reward received at time step $( t+1 )$.
- $( S_t )$ is the state at time $( t )$.

### 2. **Key Components**

- **State $((s)$)**: A specific situation or configuration in the environment.
- **Policy $((\pi))$**: A strategy or rule that defines the behavior of the agent, mapping states to actions.
- **Return**: The total accumulated reward an agent receives, usually discounted over time.
- **Discount Factor ($(\gamma)$)**: A factor used to reduce the weight of future rewards compared to immediate rewards. A lower $( \gamma )$ makes the agent more myopic, focusing on immediate rewards.

### 3. **Importance of the State Value Function**

- **Decision Making**: Helps the agent evaluate the desirability of different states. A higher value indicates a more favorable state.
- **Policy Evaluation**: Used to assess how good a given policy is in terms of the expected returns it can yield from different states.
- **Policy Improvement**: Guides the agent in improving its policy by comparing values of different states and choosing actions that lead to higher value states.

### 4. **Calculating the State Value Function**

The state value function can be estimated using various methods:

- **Dynamic Programming (DP)**: Requires a model of the environment (i.e., transition probabilities and rewards). Examples include:
  - **Value Iteration**: Iteratively updates state values based on the Bellman equation.
  - **Policy Iteration**: Alternates between policy evaluation (calculating $( V^\pi )$) and policy improvement (updating the policy based on $( V^\pi )$).
- **Monte Carlo Methods**: Uses sampled episodes from the environment to estimate \( V(s) \) by averaging the returns obtained from visiting state $( s )$.
- **Temporal Difference (TD) Learning**: Combines ideas from DP and Monte Carlo methods. TD learning updates value estimates based on observed rewards and estimated values of subsequent states (e.g., TD(0), SARSA, Q-learning).

### 5. **Bellman Equation for $( V(s) )$**

The Bellman equation provides a recursive definition for the state value function:

$[ V^\pi(s) = \sum*{a} \pi(a \mid s) \sum*{s', r} P(s', r \mid s, a) \left[ r + \gamma V^\pi(s') \right] ]$

- $( \pi(a \mid s) )$ is the probability of taking action $( a )$ in state $( s )$ under policy $( \pi )$.
- $( P(s', r \mid s, a) )$ is the probability of transitioning to state $( s' )$ and receiving reward $( r )$ given state $( s )$ and action $( a )$.

### 6. **Optimal State Value Function $(( V^* ))$**

The optimal state value function represents the maximum expected return obtainable from state $( s )$ by following the best possible policy $( \pi^* )$:

$[ V^*(s) = \max\_\pi V^\pi(s) ]$

The Bellman optimality equation for $( V^*)$ is:

$[ V^*(s) = \max*a \sum*{s', r} P(s', r \mid s, a) \left[ r + \gamma V^*(s') \right] ]$
