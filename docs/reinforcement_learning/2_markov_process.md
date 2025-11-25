# From MRPs to MDPs and Optimal Control

Reinforcement Learning (RL) is about selecting actions over time to maximize long-term reward. When the agent has a full model of the environment’s dynamics and rewards, the problem is formalized as a Markov Decision Process (MDP).

# 1. Markov Processes, MRPs, and MDPs

## 1.1 Markov Process (MP) — State Transitions Only

A Markov Process is a discrete-time stochastic system defined solely by:

- States $S$
- Transition model $P(s' \mid s)$

Its defining property is the Markov property:

$$
P(s_{t+1} \mid s_t, s_{t-1}, \ldots) = P(s_{t+1} \mid s_t)
$$

> The future is conditionally independent of the past given the present/

But there are no rewards yet. MPs describe what happens, not how good it is.

## 1.2 Markov Reward Process (MRP) — State Transitions + Rewards

An MRP adds a reward signal:
- State Reward $R(s)$
- Discount factor $\gamma \in [0,1]$

The return from time t is:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

The value function is the expected return starting from state $s$:

$$
V(s) = \mathbb{E}[G_t \mid s_t = s]
$$

This gives rise to the Bellman equation for MRPs:

$$
V(s) = R(s) + \gamma \sum_{s'} P(s' \mid s) V(s')
$$


This is a linear system and can be written compactly as:
$$
V = R + \gamma P V
$$

$$
V = (I - \gamma P)^{-1}R
$$

or solved iteratively via dynamic programming.

 
## 1.3 Markov Decision Process (MDP)

An MDP extends an MRP by letting an agent choose actions:

$$
(S, A, P, R, \gamma)
$$

where:

- A: action set
- $P(s'|s,a)$ — transition model now depends on actions
- $R(s,a)$ — reward from doing action $a$ in state $s$

The agent’s behavior is described by a policy:

$$
\pi(a \mid s)
$$

## 1.4 Link Between MRP and MDP

When we fix a policy $\pi$, an MDP reduces to an MRP::

$$
R_\pi(s) = \sum_a \pi(a|s) R(s,a)
$$

$$
P_\pi(s'|s) = \sum_a \pi(a|s) P(s'|s,a)
$$

Thus every policy induces a Markov Reward Process, making the tools for MRPs directly applicable to MDPs.

# 2. Value Functions and Q-Functions

## 2.1 State Value Function $V^\pi(s)$

Meaning:
> How much total reward should I expect if I start in state s and follow policy $\pi$: forever?


Defination:
$$
V^\pi(s) = \mathbb{E}[G_t \mid s_t = s, \pi]
$$

It is a measure of long-term goodness of a state under policy $\pi$.

## 2.2 State-Action Value Function $Q^\pi(s,a)$
> If I take action a in state s, and then follow $\pi$ forever after, how good is that?

Expected return if you take action $a$ in state $s$ and then follow $\pi$:

$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')
$$

if $V^\pi(s)$ describes how good states are,
$Q^\pi(s,a)$  describes how good actions are contextually.

Intuition:

- $V(s)$ — how good is this state - value of being
- $Q(s,a)$ — how good is this action here -value of doing
- $S$ — where you are
- $A$ — what you can do
- $R$ — what you get immediately
- $\gamma$ — how far into the future you care



# 3. Policy Evaluation: Computing $V^\pi(s)$

Compute $V^\pi$ for a fixed policy $\pi$:

$$
V_{k+1}^\pi(s) =
\sum_a \pi(a|s)
\left[
R(s,a) + \gamma \sum_{s'} P(s'|s,a)V_k^\pi(s')
\right]
$$

This is iterative application of the Bellman operator $B^\pi$.



# 4. Policy Improvement: Making a Policy Better

Given $V^\pi$, compute state-action values:

$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a)V^\pi(s')
$$

Then greedily improve:
$$
\pi'(s) = \arg\max_a Q^\pi(s,a)
$$

This new policy cannot be worse than π.


# 5. Monotonic Improvement Theorem

Key result: Greedifying with respect to $V^\pi$ always produces a policy $\pi'(s)$′ such that

$$
V^{\pi'}(s) \ge V^\pi(s)
$$

And strict inequality if $\pi$ is not optimal.

### Why

Because:

$$
V^\pi(s) = Q^\pi(s,\pi(s))
$$

and

$$
\max_a Q^\pi(s,a) \ge Q^\pi(s,\pi(s))
$$

choosing the maximizing action cannot reduce value.

# 6. Policy Iteration (PI): Evaluate + Improve

Algorithm:
1. Initialize $\pi_0$
2. Policy Evaluation: compute $V^{\pi_i}$
3. Policy Improvement: compute $\pi_{i+1}$
4. Stop when $\pi_{i+1} = \pi_i$

Properties:
- Guaranteed convergence
- Monotonic improvement
- At most $|A|^{|S|}$ iterations

This is often very fast: policies stabilize in few iterations.

# 7. Value Iteration (VI): Optimality by Repeated Bellman Backups

Instead of fully evaluating each policy, VI applies Bellman optimality operator:

$$
V_{k+1}(s) =
\max_a \left[
R(s,a) + \gamma \sum_{s'} P(s'|s,a)V_k(s')
\right]
$$

This combines:

* policy improvement (max over a)
* partial policy evaluation (use $V_k$ instead of full $V^{\pi_k}$


Optimal policy after convergence:

$$
\pi^*(s) =
\arg\max_a \left[
R(s,a) + \gamma \sum_{s'} P(s'|s,a)V^*(s')
\right]
$$



# 8. Contraction Property

The Bellman optimality operator $B$ satisfies:

$$
\|BV - BW\|_\infty \le \gamma \|V - W\|_\infty
$$

Thus $V_k$ converges to a unique fixed point $V^*$.


# 9. Finite Horizon Value Iteration

Let $V_0(s)=0$.

For $k = 1 \ldots H$:

$$
V_{k+1}(s) =
\max_a \left[
R(s,a) + \gamma \sum_{s'}P(s'|s,a)V_k(s')
\right]
$$

Policy for horizon $k+1$:

$$
\pi_{k+1}(s) =
\arg\max_a \left[
R(s,a) + \gamma \sum_{s'}P(s'|s,a)V_k(s')
\right]
$$

Finite horizon optimal policies are generally non-stationary.

 
# 10. Monte Carlo Evaluation
If you can simulate but don’t know the model:

1. Simulate episodes
2. Compute returns
3. Average them

Does *not* require Markov assumptions.
But MRP/MDP dynamic programming does require the Markov property.
---

# 11. Summary Intuition

| Symbol | Meaning | Intuition |
|--------|---------|-----------|
| $S$ | states | where you are |
| $A$ | actions | what you can do |
| $R$ | reward | what you get now |
| $\gamma$ | discount factor | how far the future matters |
| $\pi$ | policy | your strategy |
| $V^\pi$ | value | goodness of a state |
| $Q^\pi$ | state–action value | goodness of doing an action in a state |

What is optimality?

An optimal policy maximizes long-term expected discounted reward from all states.

There is:
* one unique optimal value function $V^*$
* possibly many optimal policies

What do the algorithms do?
* Policy Evaluation: compute value of a given policy
* Policy Improvement: greedify a policy
* Policy Iteration: loop eval + improve
* Value Iteration: compute optimal values directly via Bellman optimality backups


Policy iteration and value iteration are two different ways of solving an MDP, and the real difference lies in how they improve the estimate of the best decision-making strategy. Policy iteration works by first committing to a complete plan (a policy), then fully evaluating how good that plan is by computing accurate long-term values for every state, and finally improving the plan by choosing better actions wherever possible. It repeats this cycle—evaluate completely, then improve—**until the policy stops changing**. Each iteration is computationally expensive because it requires solving for the exact value of the policy, but the number of iterations is typically small since each improvement is large. 

In contrast, value iteration does not commit to any full policy up front. Instead, it repeatedly updates the value of each state using a Bellman optimality backup, which assumes the agent takes the best action next step. This means value iteration mixes evaluation and improvement at every update: each iteration only makes a small, approximate improvement, but the iterations are cheap and eventually converge to the **optimal value function**. Once values converge, a final greedy step extracts the optimal policy. 

In simple terms, policy iteration takes fewer big steps—each iteration is heavy but decisive—while value iteration takes many small steps, gradually refining values until the optimal solution emerges.