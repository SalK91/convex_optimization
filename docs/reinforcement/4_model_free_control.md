# Chapter 4: Model-Free Control and Function Approximation

 Chapters 2 and 3 laid the foundation for solving Reinforcement Learning problems. In Chapter 2, we mastered Dynamic Programming methods (Policy Iteration and Value Iteration), which provide exact solutions for finding the optimal policy $\pi^*$ but require a perfect model of the environment's dynamics ($P$ and $R$). Chapter 3 addressed the lack of a model by introducing Model-Free Policy Evaluation (Monte Carlo and TD(0)), which learned the value of a *fixed* policy $\pi$ solely from experience. This chapter completes the learning framework by tackling Model-Free Control: the challenge of *finding* the optimal policy $\pi^*$ when the model is unknown and the policy must simultaneously be learned and improved upon. We move from estimating state values ($V^\pi(s)$) to estimating action-values ($Q^\pi(s, a)$) and introduce techniques for scaling these methods to real-world problems with vast state spaces.

$\text{Model-Free Control}$ is the process of finding the optimal policy ($\pi^*$) when the agent does not have access to the environment's transition probabilities ($P$) or reward function ($R$). This challenge requires estimating the action-value function ($Q^\pi(s, a)$) directly from experience and ensuring sufficient exploration.


## Model-Free Policy Iteration and the Need for Action-Values

In the model-based setting (Chapter 2), Policy Iteration alternated between Policy Evaluation (computing $V^\pi$) and Policy Improvement (deriving a new, better $\pi'$ greedily from $V^\pi$). In the Model-Free setting, we must shift our focus from state values ($V^\pi(s)$) to action-values ($Q^\pi(s, a)$).

### From State-Value to Action-Value

When the transition dynamics ($P$) are unknown, computing the greedy policy improvement, $\pi_{i+1}(s) = \operatorname*{arg\,max}_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^{\pi_i}(s') \right]$, is impossible. We must instead use the action-value function, which already incorporates the expected reward for taking an action $a$ in state $s$:

$$
\pi_{i+1}(s) = \operatorname*{arg\,max}_{a} Q^{\pi_i}(s,a)
$$

The Model-Free Policy Iteration loop thus becomes:

1.  Policy Evaluation: Compute $Q^{\pi}$ from experience.
2.  Policy Improvement: Update the policy $\pi$ given the estimated $Q^{\pi}$.

### The Problem of Deterministic Policies

A deterministic policy ($\pi(s) = a$) only generates experience for the action it selects. Consequently, if $\pi$ is deterministic, the agent can't compute $Q(s, a)$ for any action $a$ that is not $\pi(s)$. This leads directly to the core challenge of control.

---

## II. Exploration vs. Exploitation

The fundamental dilemma of Model-Free Control is the exploration-exploitation trade-off. The agent must maximize expected future reward, but to do so, it must learn about the environment by trying new actions (explore). However, trying new actions means spending less time taking the actions that current knowledge suggests will yield high reward (exploit).

### $\epsilon$-Greedy Policies

The $\epsilon$-greedy policy is a simple and common solution to balance this trade-off. For a given state-action value function $Q(s, a)$ and a small parameter $\epsilon \in (0, 1]$, the policy $\pi(a|s)$ is defined as:

* Select the greedy action (the one that maximizes $Q(s, a)$) with probability $1 - \epsilon$.
* Select a random action (uniformly from all actions) with probability $\epsilon$.

The formal expression for an $\epsilon$-greedy policy with $|A|$ actions is:

$$
\pi(a|s) =
\begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \operatorname*{arg\,max}_{a'} Q(s, a') \\
\frac{\epsilon}{|A|} & \text{if } a \neq \operatorname*{arg\,max}_{a'} Q(s, a')
\end{cases}
$$

Importantly, policy iteration using $\epsilon$-greedy policies is guaranteed to yield monotonic improvement, meaning the value of the new policy is always greater than or equal to the value of the previous policy: $V^{\pi_{i+1}} \geq V^{\pi_i}$.

---

## III. Monte Carlo Control (Tabular)

Monte Carlo Control adapts the MC policy evaluation methods from Chapter 3 to the control problem by estimating the Action-Value Function $Q(s, a)$ instead of $V(s)$.

### On-Policy MC Control

The simplest approach is On-Policy MC Control (also known as MC Exploring Starts), which follows the generalized policy iteration structure using $\epsilon$-greedy policies for exploration.

* Evaluation: $Q(s, a)$ is updated using the full return ($G_t$) observed after the state-action pair $(s_t, a_t)$ has occurred in an episode. The incremental update uses the formula $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \frac{1}{N(s,a)}(G_{t} - Q(s_t, a_t))$.
* Improvement: The new policy $\pi_{k+1}$ is set to be $\epsilon$-greedy with respect to the updated $Q$ function.

### Greedy in the Limit of Infinite Exploration (GLIE)

For Monte Carlo Control to converge to the optimal action-value function $Q^*(s, a)$, the process must satisfy the Greedy in the Limit of Infinite Exploration (GLIE) conditions:

1.  Infinite Visits: All state-action pairs $(s, a)$ must be visited an infinite number of times ($\lim_{i \rightarrow \infty} N_i(s, a) \rightarrow \infty$).
2.  Converging Greed: The behavior policy (the policy used to act and generate data) must eventually converge to a greedy policy.

A simple strategy to satisfy GLIE is to use an $\epsilon$-greedy policy where $\epsilon$ is decayed over time, such as $\epsilon_i = 1/i$ (where $i$ is the episode number). Under the GLIE conditions, Monte-Carlo control converges to the optimal state-action value function $Q^*(s, a)$.

---

## IV. Temporal Difference (TD) Control (Tabular)

TD control methods combine the $\epsilon$-greedy exploration of Monte Carlo with the bootstrapping (learning from estimates) and online updating of TD learning.

### A. On-Policy TD Control: SARSA

SARSA is an on-policy TD control algorithm. It learns the value of the policy *currently being followed* ($\pi$). Its name is derived from the sequence of steps used in its update rule: State, Action, Reward, State, Action.

The update for the action-value $Q(s_t, a_t)$ uses the value of the *next* state-action pair, $(s_{t+1}, a_{t+1})$, selected by the current policy $\pi$.

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

The TD Target here is $r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$. SARSA learns $Q^{\pi}$ while $\pi$ is improved greedily with respect to $Q^{\pi}$, allowing it to find the optimal policy $\pi^*$.

### B. Off-Policy TD Control: Q-Learning

Q-Learning is the most widely known off-policy TD control algorithm. Off-policy learning means we estimate and evaluate an optimal policy ($\pi^*$, the *target policy*) using experience gathered by a different behavior policy ($\pi_b$).

In Q-Learning, the agent acts using a soft, exploratory $\pi_b$ (like $\epsilon$-greedy) but the value function update is based on the *best* possible action from the next state, effectively estimating $Q^*$.

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

The key difference is the target: Q-Learning uses the value of the max action ($\max_{a'} Q(s_{t+1}, a')$), regardless of what action was actually taken in the next step. This makes it a greedy update towards $Q^*$.

Convergence: Q-Learning converges to the optimal action-value $Q^*(s, a)$, provided the learning rate $\alpha$ satisfies the Robbins-Munro conditions and the behavior policy $\pi_b$ satisfies the GLIE condition (ensuring all state-action pairs are continually visited).

| Feature | SARSA (On-Policy) | Q-Learning (Off-Policy) |
| :--- | :--- | :--- |
| Behavior Policy | Uses $\pi$ to choose $a_t$ and $a_{t+1}$ | Uses $\pi_b$ to choose $a_t$, but ignores $a_{t+1}$ for the update. |
| Target Value | $r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$ | $r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$ |

---

## V. Value Function Approximation (VFA)

All methods discussed so far assume a tabular representation, where a separate entry for $Q(s, a)$ is stored for every state-action pair. This is only feasible for MDPs with small, discrete state and action spaces.

### Motivation for Approximation

For environments with large or continuous state/action spaces (e.g., in robotics or image-based games like Atari), we face three critical issues:

1.  Memory: Explicitly storing every $V$ or $Q$ value is impossible.
2.  Computation: Computing or updating every value is too slow.
3.  Experience: It would take vast amounts of data to visit and learn every single state-action pair.

Value Function Approximation (VFA) addresses this by using a parameterized function (like a linear model or a neural network) to estimate the value function: $\hat{Q}(s, a; \mathbf{w}) \approx Q(s, a)$. The goal shifts from filling a table to finding the parameter vector $\mathbf{w}$ that minimizes the error between the true value and the estimate.

$$
J(\mathbf{w}) = \mathbb{E}_{\pi} \left[ \left( Q^{\pi}(s, a) - \hat{Q}(s, a; \mathbf{w}) \right)^2 \right]
$$

The parameter vector $\mathbf{w}$ is typically updated using Stochastic Gradient Descent (SGD), which uses a single sample to approximate the gradient of the loss function $J(\mathbf{w})$.

### Model-Free Control with VFA

When using function approximation, we substitute the old $Q(s, a)$ in the update rules (MC, SARSA, Q-Learning) with the function approximator $\hat{Q}(s, a; \mathbf{w})$.

* MC with VFA: The return $G_t$ is used as the target in an SGD update: $\Delta \mathbf{w} \propto \alpha (G_t - \hat{Q}(s_t, a_t; \mathbf{w})) \nabla_{\mathbf{w}} \hat{Q}(s_t, a_t; \mathbf{w})$.
* SARSA with VFA: The TD target is $r + \gamma \hat{Q}(s', a'; \mathbf{w})$, leveraging the current function approximation.
* Q-Learning with VFA: The TD target is $r + \gamma \max_{a'} \hat{Q}(s', a'; \mathbf{w})$.

### Deep Q-Networks (DQN)

The most prominent example of VFA for control is Deep Q-Learning, or Deep Q-Networks (DQN), where the action-value function $\hat{Q}(s, a; \mathbf{w})$ is approximated by a deep neural network. DQN successfully solved control problems directly from raw sensory input (e.g., pixels from Atari games).

DQN stabilizes the non-linear learning process using two critical techniques:

1.  Experience Replay (ER): Transitions $(s_t, a_t, r_t, s_{t+1})$ are stored in a replay buffer ($\mathcal{D}$). Instead of learning from sequential, correlated experiences, the algorithm samples a random mini-batch of past transitions from $\mathcal{D}$ for the update. This breaks correlations, making the data samples closer to i.i.d (independent and identically distributed).
2.  Fixed Q-Targets: The Q-Learning update requires a target value $y_i = r_i + \gamma \max_{a'} \hat{Q}(s_{i+1}, a'; \mathbf{w})$. To prevent the estimate $\hat{Q}(s, a; \mathbf{w})$ from chasing its own rapidly changing target, the parameters $\mathbf{w}^{-}$ used to compute the target are fixed for a period of time, then synchronized with the current parameters $\mathbf{w}$. This provides a stable target $y_i = r_i + \gamma \max_{a'} \hat{Q}(s_{i+1}, a'; \mathbf{w}^{-})$.