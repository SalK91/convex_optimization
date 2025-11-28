Reinforcement Learning (RL) is fundamentally the science of sequential decision-making under uncertainty. When an agent possesses a complete mathematical description (a model) of its environment's rules and rewards, this complex problem is rigorously formalized by the Markov Decision Process (MDP). Understanding the MDP hierarchy—from simple state transitions to optimal policy discovery—is the bedrock of modern RL.

> Reinforcement Learning (RL) is about selecting actions over time to maximize long-term reward.

## I. The Markovian Hierarchy

The RL framework is built upon three foundational models, each adding complexity and agency.

### 1. The Markov Process (MP)
A Markov Process, or Markov Chain, is the simplest model, concerned only with the flow of states. It is defined by the set of States ($S$) and the Transition Model ($P(s' \mid s)$).

The defining characteristic is the Markov Property: the next state is independent of the past states, given only the current state.
$$
P(s_{t+1} \mid s_t, s_{t-1}, \ldots) = P(s_{t+1} \mid s_t)
$$

>  The future is conditionally independent of the past given the present/

> *Intuition: MPs describe what happens but do not assign any value to these events.*

### 2. The Markov Reward Process (MRP)
An MRP introduces the concept of value by adding Rewards ($R(s)$) and the Discount Factor ($\gamma \in [0,1]$). The central concept here is the Return ($G_t$), the sum of all future rewards, discounted exponentially:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

The goal of an MRP is to calculate the Value Function ($V(s)$), the expected return from state $s$: $V(s) = \mathbb{E}[G_t \mid s_t = s]$.

This gives rise to the Bellman Equation for MRPs: a linear system where the value of a state is the sum of its immediate reward and the discounted, expected value of its successor states.

$$
V(s) = R(s) + \gamma \sum_{s'} P(s' \mid s) V(s')
$$

In matrix form (for solving via direct inversion or iteration):

$$
V = R + \gamma P V \implies V = (I - \gamma P)^{-1}R
$$

### 3. The Markov Decision Process (MDP)
An MDP introduces agency. Defined by the tuple $(S, A, P, R, \gamma)$, it extends the MRP by giving the agent a set of Actions ($A$) to choose from.
* Action-Dependent Transition: $P(s' \mid s, a)$
* Action-Dependent Reward: $R(s, a)$

The agent's strategy is described by a Policy ($\pi(a \mid s)$), the probability of selecting action $a$ in state $s$. A key insight is that fixing any policy $\pi$ reduces an MDP back into an MRP, allowing all tools developed for MRPs to be applied to the MDP.


$$
R_\pi(s) = \sum_a \pi(a|s) R(s,a)
$$

$$
P_\pi(s'|s) = \sum_a \pi(a|s) P(s'|s,a)
$$

## II. Value Functions and Expectation

To evaluate a fixed policy $\pi$, we define two inter-related value functions based on the Bellman Expectation Equations.

### 1. State Value Function ($V^\pi(s)$)
$V^\pi(s)$ quantifies the long-term expected return starting from state $s$ and strictly following policy $\pi$.
$$
V^\pi(s) = \mathbb{E}[G_t \mid s_t = s, \pi]
$$
> How much total reward should I expect if I start in state s and follow policy $\pi$: forever?

### 2. State-Action Value Function ($Q^\pi(s,a)$)
$Q^\pi(s,a)$ is a more granular measure, quantifying the expected return if the agent takes action $a$ in state $s$ first, and *then* follows policy $\pi$.
$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')
$$
> *Intuition:* The $Q$-function is the value of doing a specific action; the $V$-function is the value of being in a state (the weighted average of the $Q$-values offered by the policy $\pi$ in that state):
$$
V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s,a)
$$

The Bellman Expectation Equation for $V^\pi$ links the value of a state to the values of the actions chosen by $\pi$ and the resulting future states:
$$
V^\pi(s) = \sum_a \pi(a \mid s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]
$$

 
## III. Optimal Control: Finding $\pi^*$

The ultimate goal of solving an MDP is to find the optimal policy ($\pi^*$) that maximizes the expected return from every state $s$.

$$
\pi^* = \operatorname*{arg\,max}_{\pi} V^\pi(s) \quad \text{for all } s \in S
$$

This optimal policy is characterized by the Optimal Value Functions ($V^*$ and $Q^*$).

### 1. The Bellman Optimality Equations
These equations are fundamental, describing the unique value functions that arise when acting optimally. Unlike the expectation equations, they contain a $\max$ operator, making them non-linear.

* Optimal State Value ($V^*$): The optimal value of a state equals the maximum expected return achievable from any single action $a$ taken from that state:

    $$
    V^*(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]
    $$

* Optimal Action-Value ($Q^*$): The optimal value of taking action $a$ is the immediate reward plus the discounted value of the optimal subsequent actions ($\max_{a'}$) in the next state $s'$:

    $$
    Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s', a')
    $$

Once $Q^*$ is known, the optimal policy $\pi^*$ is easily extracted by simply choosing the action that maximizes $Q^*(s,a)$ in every state:
$$
\pi^*(s) = \operatorname*{arg\,max}_{a} Q^*(s,a)
$$


## IV. Dynamic Programming Algorithms

For MDPs where the model ($P$ and $R$) is fully known, Dynamic Programming methods are used to solve the Bellman Optimality Equations iteratively.

### 1. Policy Iteration (PI)
Policy Iteration (PI) follows an alternating cycle of Evaluation and Improvement. It takes fewer, but more expensive, iterations to converge.

1.  Policy Evaluation: For the current policy $\pi_k$, compute $V^{\pi_k}$ by iteratively applying the Bellman Expectation Equation until full convergence. This is the computationally intensive step.
    $$
    V^{\pi_k}(s) \leftarrow \text{solve } V^{\pi_k} = R_{\pi_k} + \gamma P_{\pi_k} V^{\pi_k}
    $$
2.  Policy Improvement: Update the policy $\pi_{k+1}$ by choosing an action that is greedy with respect to the fully converged $V^{\pi_k}$.
    $$
    \pi_{k+1}(s) \leftarrow \operatorname*{arg\,max}_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^{\pi_k}(s') \right]
    $$
    
The process repeats until the policy stabilizes ($\pi_{k+1} = \pi_k$), guaranteeing convergence to $\pi^*$.

### 2. Value Iteration (VI)
Value Iteration (VI) is a single, continuous process that combines evaluation and improvement by repeatedly applying the Bellman Optimality Equation. It takes many, but computationally cheap, iterations.

1.  Iterative Update: For every state $s$, update the value function $V_k(s)$ using the $\max$ operation. This immediately incorporates a greedy improvement step into the value update.
    $$
    V_{k+1}(s) \leftarrow \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]
    $$
2.  Convergence: The iterations stop when $V_{k+1}$ is sufficiently close to $V^*$.
3.  Extraction: The optimal policy $\pi^*$ is then extracted greedily from the final $V^*$.

| Feature | Policy Iteration (PI) | Value Iteration (VI) |
| :--- | :--- | :--- |
| Core Idea | Evaluate completely, then improve. | Greedily improve values in every step. |
| Equation | Uses Bellman Expectation (inner loop) | Uses Bellman Optimality (max) |
| Convergence | Few, large policy steps. Policy guaranteed to stabilize faster. | Many, small value steps. Value function converges slowly to $V^*$. |
| Cost | High cost per iteration (due to full evaluation). | Low cost per iteration (due to one-step backup). |

 