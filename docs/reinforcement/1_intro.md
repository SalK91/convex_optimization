Reinforcement Learning (RL) is defined as learning through experience and data to make good decisions under uncertainty. RL differs from other learning types because it involves four specific challenges:

1.  Optimization: The goal is to find an optimal way to make decisions that yield the best outcomes (utility).
2.  Delayed Consequences: Decisions made now can impact things much later.
    * $\star$ *Challenge:* Temporal credit assignment is hard—it is difficult to determine which past action caused a later high or low reward.
3.  Exploration: The agent acts as a scientist, learning about the world by trying actions.
    * $\star$ *Challenge:* You only get a reward for the decision made; you do not know what would have happened if you chose differently.
4.  Generalization: The policy ($\pi$) is a learned mapping from past experience to action, rather than a pre-programmed rule set.

RL particularly powerful:

1.  No Examples: When there are no examples of desired behavior (e.g., aiming for superhuman performance or no existing data).
2.  Complex Search: When solving enormous search or optimization problems with delayed outcomes (e.g., Matrix multiplication optimization in AlphaTensor).

---

Sequential Decision Making:  The core of RL is the interaction loop between the Agent and the World.

At each time step $t$:

1.  The Agent takes an action $a_t$.
2.  The World updates given $a_t$ and emits an observation $o_t$ and reward $r_t$.
3.  The Agent receives observation $o_t$ and reward $r_t$.

Goal: Select actions to maximize total expected future reward. May require balancing immediate and long term rewards.


* History ($h_t$): The sequence of past observations, actions, and rewards.
    $$h_t = (a_1, o_1, r_1, \dots, a_t, o_t, r_t)$$
* State ($s_t$): Information assumed to determine what happens next. It is a function of history:
    $$s_t = f(h_t)$$

A state $s_t$ is Markov if and only if the future is independent of the past, given the present:

$$p(s_{t+1}|s_t, a_t) = p(s_{t+1}|h_t, a_t)$$

This assumption is popular because it reduces computational complexity and data requirements.