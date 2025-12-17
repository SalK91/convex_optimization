# Chapter 11: Data-Efficient Reinforcement Learning — Bandit Foundations


In real-world applications of Reinforcement Learning (RL), data is expensive, time-consuming, or risky to collect. This necessitates data-efficient RL: designing agents that learn effectively from limited interaction. Bandits provide a foundational setting to study such principles. In this chapter, we explore multi-armed banditsas the prototypical framework for understanding the exploration-exploitation tradeoff, and examine several algorithmic approaches and regret-based evaluation criteria.


## The Multi-Armed Bandit Model

A multi-armed bandit is defined as a tuple $(\mathcal{A}, \mathcal{R})$, where:

- $\mathcal{A} = \{a_1, \dots, a_m\}$ is a known, finite set of actions (arms),
- $R_a(r) = \mathbb{P}[r \mid a]$ is an unknown probability distribution over rewards for each action.
- there is no "state".

At each timestep $t$, the agent:

1. Chooses an action $a_t \in \mathcal{A}$,
2. Receives a stochastic reward $r_t \sim R_{a_t}$.

Goal: Maximize cumulative reward:  
$$
\sum_{t=1}^{T} r_t
$$

This simple model embodies the core RL challenges—particularly exploration vs. exploitation—in an isolated setting.


### Evaluating Algorithms: Regret Framework

Regret: 

- $Q(a) = \mathbb{E}[r \mid a]$ be the expected reward for action $a$,
- $a^* = \arg\max_{a \in \mathcal{A}} Q(a)$,
- Optimal Value $V^* = Q(a^*)$

Then regret is the opportunity loss for one step:
$$
\ell_t = \mathbb{E}[V^* - Q(a_t)]
$$

Total Regret is the total opportunity loss: Total regret over $T$ timesteps

$$
L_T = \sum_{t=1}^T \ell_t = \sum_{a \in \mathcal{A}} \mathbb{E}[N_T(a)] \cdot \Delta_a
$$
Where:

- $N_T(a)$: Number of times arm $a$ is selected by time $T$,
- $\Delta_a = V^* - Q(a)$: Suboptimality gap.


> Maximize cumulative reward <=> minimize total regret





## Baseline Approaches and Their Regret

### Greedy Algorithm

$$
\hat{Q}_t(a) = \frac{1}{N_t(a)} \sum_{\tau=1}^t r_\tau \cdot \mathbb{1}(a_\tau = a)
$$

$$
a_t = \arg\max_{a \in \mathcal{A}} \hat{Q}_t(a)
$$

#### Key Insight:

- Exploits current estimates.
- May lock onto suboptimal arms due to early bad luck.
- Linear regret in expectation.

### Example:
If $Q(a_1) = 0.95, Q(a_2) = 0.90, Q(a_3) = 0.1$, and the first sample of $a_1$ yields 0, the greedy agent may ignore it indefinitely.

### $\varepsilon$-Greedy Algorithm

At each timestep:

- With probability $1 - \varepsilon$: exploit ($\arg\max \hat{Q}_t(a)$),
- With probability $\varepsilon$: explore uniformly at random.

#### Performance:
- Guarantees exploration.
- Linear regret unless $\varepsilon$ decays over time.

### Decaying $\varepsilon$-Greedy

Allows $\varepsilon_t \to 0$ as $t \to \infty$, enabling convergence.


## Optimism in the Face of Uncertainty
Prefer actions with uncertain but potentially high value:

Why? Two possible outcomes:

1. Getting a high reward:    If the arm really has a high mean reward.

2. Learning something : If the arm really has a lower mean reward, pulling it will (in expectation) reduce its average reward estimate and the uncertainty over its value.


Algorithm: 

* Estimate an upper confidence bound $U_t(a)$ for each action value, such that   $Q(a) \le U_t(a)$ with high probability.

* This depends on the number of times $N_t(a)$ action $a$ has been selected.

* Select the action maximizing the Upper Confidence Bound (UCB):

$$a_t = \arg\max_{a \in \mathcal{A}} \left[ U_t(a) \right]$$


> Hoeffding Bound Justification
> Given i.i.d. bounded rewards $X_i \in [0,1]$,
> $$\mathbb{P}\left[ \mathbb{E}[X] > \bar{X}_n + u \right] \le \exp(-2nu^2)$$





$$
a_t = \arg\max_{a \in \mathcal{A}} \left[ \hat{Q}_t(a) + \text{UCB}_t(a) \right]
$$


### UCB1 Algorithm

$$
\text{UCB}_t(a) = \hat{Q}_t(a) + \sqrt{\frac{2 \log t}{N_t(a)}}
$$

- Provable sublinear regret.
- Balances estimated value and exploration bonus.


---

## 11.5 Lower Bounds and Problem Hardness

### Lai & Robbins Lower Bound

No algorithm can do better (asymptotically) than:
$$
\lim_{T \to \infty} L_T \ge \log T \sum_{a: \Delta_a > 0} \frac{\Delta_a}{\text{KL}(R_a || R_{a^*})}
$$

Where $\text{KL}(R_a || R_{a^*})$ is the Kullback-Leibler divergence between reward distributions.  
Hard problems have similar distributions across arms.

---

## 11.6 Bayesian Regret and Thompson Sampling

Instead of upper bounds, use posterior distributions over arm values.

### Thompson Sampling:

For each $a$:
1. Sample $\theta_a \sim P(\theta_a \mid \mathcal{D})$,
2. Select $a_t = \arg\max \theta_a$.

Probabilistic matching between belief and action. Known to achieve near-optimal regret bounds both empirically and theoretically.

---

## 11.7 Summary of Approaches

| Algorithm          | Exploration | Regret Behavior  | Notes                               |
|-------------------|-------------|------------------|--------------------------------------|
| Greedy            | None        | Linear           | May miss optimal arm                 |
| $\varepsilon$-Greedy | Fixed       | Linear           | Unless $\varepsilon$ decays          |
| UCB1              | Optimistic  | $O(\log T)$      | Theoretical guarantees               |
| Thompson Sampling | Posterior   | $O(\log T)$      | Bayesian, flexible                   |

---

## 11.8 Key Takeaways

- Bandits abstract the exploration challenge and help benchmark RL algorithms.
- Regret is a key theoretical tool for assessing efficiency.
- Optimism and probability matching offer two distinct yet powerful paths to exploration.
- These principles extend beyond bandits, forming the backbone of data-efficient reinforcement learning in general.
