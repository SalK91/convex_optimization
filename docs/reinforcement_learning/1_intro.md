## 1. What is Reinforcement Learning (RL)?
Reinforcement Learning is defined as learning through experience and data to make good decisions under uncertainty . It is considered an essential part of intelligence .

The field builds strongly on theory and ideas starting in the 1950s with Richard Bellman . However, there has been a huge increase in interest and success in the last decade .

### Notable Recent Successes
* Games: Achieving superhuman performance in the board game Go .
* Science:
    * Learning plasma control for fusion science .
    * AI achieving silver-medal standards on International Mathematical Olympiad problems .
* Healthcare/Policy: "Eva," a system for efficient and targeted COVID-19 border testing allocation .
* LLMs: Using RL (specifically PPO) for training models like ChatGPT via steps involving prompt collection, reward modeling, and policy optimization .


## 2. Key Characteristics of RL
RL differs from other learning types because it involves four specific challenges :

1.  Optimization: The goal is to find an optimal way to make decisions that yield the best outcomes (utility) .
2.  Delayed Consequences: Decisions made now can impact things much later .
    * *Challenge:* Temporal credit assignment is hard—it is difficult to determine which past action caused a later high or low reward .
3.  Exploration: The agent acts as a scientist, learning about the world by trying actions .
    * *Challenge:* You only get a reward for the decision made; you do not know what would have happened if you chose differently .
4.  Generalization: The policy is a learned mapping from past experience to action, rather than a pre-programmed rule set .

### When is RL particularly powerful?
1.  No Examples: When there are no examples of desired behavior (e.g., aiming for superhuman performance or no existing data) .
2.  Complex Search: When solving enormous search or optimization problems with delayed outcomes (e.g., Matrix multiplication optimization in AlphaTensor) .


## 3. Sequential Decision Making
The core of RL is the interaction loop between the Agent and the World .

### The Interaction Loop (Discrete Time)
At each time step $t$ :
1.  The Agent takes an action $a_t$.
2.  The World updates given $a_t$ and emits an observation $o_t$ and reward $r_t$.
3.  The Agent receives observation $o_t$ and reward $r_t$.

Goal: Select actions to maximize total expected future reward. May require balancing immediate and long term rewards.

### History and State
* History ($h_t$): The sequence of past observations, actions, and rewards .
    $$h_t = (a_1, o_1, r_1, \dots, a_t, o_t, r_t)$$
* State ($s_t$): Information assumed to determine what happens next. It is a function of history ($s_t = f(h_t)$) .

### The Markov Assumption
A state $s_t$ is Markov if and only if the future is independent of the past, given the present :
$$p(s_{t+1}|s_t, a_t) = p(s_{t+1}|h_t, a_t)$$

This assumption is popular because it reduces computational complexity and data requirements .


## 4. Formal Models

### A. Markov Process (MP)
A memoryless random process defined by a tuple $(S, P)$ .
* $S$: A finite set of states ($s \in S$).
* $P$: Dynamics/transition model specifying $P(s_{t+1} = s' | s_t = s)$.
* *Note:* No rewards, no actions.

### B. Markov Reward Process (MRP)
An MRP is a Markov Chain plus rewards, defined by $(S, P, R, \gamma)$ .
* $R$: Reward function, where $R(s_t=s) = \mathbb{E}[r_t | s_t=s]$.
* $\gamma$: Discount factor, $\gamma \in [0, 1]$ .

#### Value and Return
* Return ($G_t$): The discounted sum of rewards from time $t$ to horizon $H$ .
    $$G_t = r_t + r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{H-1} r_{t+H-1}$$
* State Value Function ($V(s)$): The expected return starting from state $s$ .
    $$V(s) = \mathbb{E}[G_t | s_t = s]$$

#### The Bellman Equation for MRPs
The value of a state can be decomposed into immediate reward plus the discounted value of the next state:

$$V(s) = R(s) + \gamma \sum_{s' \in S} P(s'|s)V(s')$$

### C. Markov Decision Process (MDP)
An MDP adds actions to the model .
* Transition Model: $P(s_{t+1} = s' | s_t = s, a_t = a)$.
* Reward Model: $r(s, a) = \mathbb{E}[r_t | s_t = s, a_t = a]$.
* Policy ($\pi$): A mapping from states to actions ($\pi: S \rightarrow A$). Policy determined how the agent choses actions. 


## 5. Goals in RL
There are two main tasks :
1.  Evaluation: Estimate/predict the expected rewards from following a given policy.
2.  Control: Find the best policy (Optimization).