# Chapter 6: Advanced Policy Gradient Methods

Policy Gradient methods are powerful but face several practical challenges:

* High variance in gradient estimates  
* Poor sample efficiency (data discarded after one update)  
* Sensitive to step-size — large updates may collapse the policy  
* Policy updates occur in parameter space, not true policy space  
* Reusing old data (off-policy) is unstable  

This chapter introduces techniques to address these issues:
1. Baselines for variance reduction  
2. Advantage estimation (TD, n-step, GAE)  
3. Actor-Critic methods  
4. Performance guarantees and KL-divergence  
5. Proximal Policy Optimization (PPO)


## Policy Gradient Recap

In the previous chapter, we derived an expression for the gradient of the policy objective:

$$
\nabla_\theta V(\theta) \approx
\frac{1}{m} \sum_{i=1}^{m}
R(\tau^{(i)}) 
\sum_{t=0}^{T-1}
\nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)})
$$

This Monte Carlo policy gradient estimator is unbiased, meaning that on average it gives the correct gradient direction. However, it suffers from high variance, because the return $R(\tau)$ depends on the entire trajectory, which can vary significantly between episodes. As a result, updates can be noisy, unstable, and slow to converge.

In reinforcement learning, data used for learning is obtained through actual interactions with the environment.  
This means we must make decisions using the current policy—even when it is still poor or suboptimal. Such interactions may lead to inefficient or costly behavior, but they are necessary for learning.


Goal:  Learn an effective policy with as few interactions (samples or episodes) as possible,  
and converge as quickly and reliably as possible to a good (local) optimum.



## Variance Reduction with Baseline

To reduce variance of the policy gradient estimator, we introduce a baseline function $b(s_t)$:

$$
\nabla_\theta \mathbb{E}_\tau [R] \;=\;
\mathbb{E}_\tau \left[
\sum_{t=0}^{T-1}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\;
\left(\sum_{t'=t}^{T-1} r_{t'} - b(s_t)\right)
\right]
$$

For any choice of baseline $b(s)$ (as long as it depends only on the state and not on the action), the gradient estimator remains unbiased — it still points in the correct direction on average.


### What is a good baseline?

A near-optimal choice for $b(s_t)$ is the expected return from state $s_t$, i.e., the value function:

$$
b(s_t) \approx \mathbb{E}[r_t + r_{t+1} + \dots + r_{T-1}]
= V_\pi(s_t)
$$

Intuition: The baseline represents the expected return from a state. So the gradient update becomes proportional to how much better (or worse) the action performed compared to expectation.

> Increase $\log \pi_\theta(a_t|s_t)$ if the action did better than expected (positive advantage), Decrease it if the action did worse than expected (negative advantage).

This keeps the estimator unbiased but significantly reduces variance, leading to more stable updates.



This leads to the advantage function:

$$
A(s_t, a_t) = Q(s_t,a_t) - V(s_t)
$$

> ### What is the Advantage Function?

>  The advantage function measures how much better (or worse) an action is compared to the average action the policy would normally take in that state.  Formally, it is defined as:
>   $$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$
> This decomposition has an intuitive interpretation:
> - $Q(s_t, a_t)$ is the expected return if we take action $a_t$ in state $s_t$ and follow the policy afterward.  
> - $V(s_t)$ is the expected return from $s_t$ *on average* under the current policy, without committing to any specific action.
> Therefore:
>
>- If $A(s_t,a_t) > 0$, the action performed *better than expected*.  
>- If $A(s_t,a_t) < 0$, the action performed *worse than expected*.  
>- If $A(s_t,a_t) = 0$, the action was exactly as good as the policy's average behavior.
> ### Why Advantage Is Useful
> Advantage isolates the *incremental contribution* of the chosen action.   It tells the policy gradient update *how much credit or blame* the action deserves.
> Using the advantage function helps because:
> 1. Reduces variance: subtracting the baseline $V(s)$ removes large parts of the return that don't depend on the specific action.  
> 2. Focuses updates on decisions that matter: only deviations from expected performance influence learning.  
> 3. Keeps estimator unbiased: since $V(s)$ does not depend on $a_t$, the expectation remains unchanged.



### Algorithm: Policy Gradient with Baseline (Advantage Estimation)

1: Initialize policy parameter $\theta$, baseline $b(s)$  
2: for iteration = 1, 2, ... do  
3: $\quad$ Collect a set of trajectories by executing the current policy $\pi_\theta$  
4: $\quad$ for each trajectory $\tau^{(i)}$ and each timestep $t$ do  
5: $\quad\quad$ Compute return:  
$\quad\quad\quad G_t^{(i)} = \sum_{t'=t}^{T-1} r_{t'}^{(i)}$  
6: $\quad\quad$ Compute advantage estimate:  
$\quad\quad\quad \hat{A}_t^{(i)} = G_t^{(i)} - b(s_t^{(i)})$  
7: $\quad$ end for  
8: $\quad$ Re-fit baseline by minimizing:  
$\quad\quad \sum_i \sum_t \left(b(s_t^{(i)}) - G_t^{(i)}\right)^2$  
9: $\quad$ Update policy parameters using gradient estimate:  
$\quad\quad \theta \leftarrow \theta + \alpha \sum_{i,t} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)})\, \hat{A}_t^{(i)}$  
10: $\quad$ (Plug into SGD or Adam optimizer)  
11: end for  


## Actor-Critic: Combining Policy and Value Function

The key idea behind Actor–Critic methods is to reduce the high variance of Monte Carlo policy gradient estimates. Instead of estimating the return from a single full rollout (the Monte Carlo return $G_t^{(i)}$), the critic uses bootstrapping and function approximation—just like the improvement from Monte Carlo to TD learning.

The result is a lower-variance, more sample-efficient estimate of the advantage.


### Components

Actor: The policy parameterization $\pi_\theta(a|s)$, which selects actions and updates the policy parameters $\theta$.

Critic: A learned value function (either $V(s)$ or $Q(s,a)$) with parameters $w$, used to estimate the advantage and provide lower-variance gradients.

Actor Update: The actor adjusts the policy in proportion to the estimated advantage:

$$
\theta \leftarrow 
\theta + \alpha \, \nabla_\theta \log \pi_\theta(a_t|s_t) \, A_t
$$

Here, the policy becomes more likely to choose actions that the critic believes have positive advantage.

Critic Update: The critic learns to approximate the value function using TD learning:

$$
w \leftarrow 
w + \beta \, \delta_t \, \nabla_w V(s_t; w)
$$

where the TD error is

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

This TD error measures how much better or worse the outcome was compared to the critic’s current prediction.

### Intuition
- The actor collects data by running the current policy and choosing actions.  
- The critic evaluates those actions by estimating how good the resulting states and returns were.

We call it *critic* because it provides an explicit evaluation of the policy’s performance. The actor proposes actions, and the critic judges whether they were better or worse than expected,  
guiding the policy updates accordingly.

This interaction mirrors the TD vs. MC trade-off:

- Monte Carlo (REINFORCE): unbiased but high variance  
- Actor–Critic (TD-based advantage): biased but lower variance and far more sample-efficient

Thus, Actor–Critic methods offer the best of both worlds: faster learning, lower variance, and the ability to scale to large or continuous environments.

## Target Estimation: TD, n-Step, and Monte Carlo

So far, we have seen two extremes for estimating returns:

- Monte Carlo return: Roll out the episode until termination and use the full return  
  $$
  G_t = \sum_{k=t}^{T-1} r_{k+1}
  $$
  This estimator is unbiased but typically has very high variance.

- One-step TD target: Use a bootstrapped estimate based on the critic:  
  $$
  r_t + \gamma V(s_{t+1})
  $$
  This estimator is low variance but biased, because it relies on an approximate value function.

These two approaches represent opposite ends of a spectrum. However, the critic does not need to choose only one extreme. Instead, it can combine information from both Monte Carlo and bootstrapping. This leads to a family of n-step return estimators, which interpolate smoothly between the TD target and the full Monte Carlo return.

In general, the critic can use any mixture of TD and Monte Carlo signals. n-step returns provide a flexible trade-off:

- Shorter n: more bias, lower variance (more like TD)  
- Larger n: less bias, higher variance (more like Monte Carlo)

This family of estimators allows the critic to balance bootstrapping with empirical returns, improving stability and sample efficiency.

 
### One-Step TD Target

The 1-step return uses the immediate reward and a bootstrapped value estimate:

$$
\hat{R}_t^{(1)} = r_t + \gamma V(s_{t+1})
$$

Advantage estimate:

$$
\hat{A}_t^{(1)} = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

This is the lowest variance estimator, but also the most biased.

 
### n-Step Return

The n-step return blends real rewards for $n$ steps with a bootstrapped estimate:

$$
\hat{R}_t^{(n)} =
r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1}
+ \gamma^n V(s_{t+n})
$$

Corresponding advantage:

$$
\hat{A}_t^{(n)} = \hat{R}_t^{(n)} - V(s_t)
$$

By choosing $n$, we control the bias–variance trade-off:

- Small $n$ → low variance, higher bias  
- Large $n$ → high variance, lower bias  

 
### Monte Carlo Return (∞-Step)

In the limit as $n \to \infty$, the n-step return becomes the Monte Carlo return:

$$
\hat{R}_t^{(\infty)} = G_t
$$

Advantage:

$$
\hat{A}_t^{(\infty)} = G_t - V(s_t)
$$

This is unbiased but has very high variance, especially in long or stochastic episodes.

 
## Problems with Vanilla Policy Gradient Methods

Although policy gradient methods provide a clean and direct way to optimize policies, they suffer from several fundamental issues. These limitations motivate the need for more advanced algorithms such as Actor–Critic, TRPO, and PPO.


### Sample Inefficiency

Vanilla policy gradient methods are highly sample-inefficient:

- Each batch of collected trajectories is used for only a single gradient step.
- After the update, the entire batch is discarded.
- To obtain another unbiased gradient estimate, the agent must collect fresh trajectories using the *new* policy.

This happens because the policy gradient is an on-policy expectation:

$$
\nabla_\theta J(\theta) = 
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) A_t
\right]
$$

Using data from old policies introduces bias. Thus, vanilla PG wastes data and requires many interactions with the environment — a major problem in real-world RL.



### Using Data for Multiple Gradient Steps

A natural question arises:Can we reuse old trajectories to take multiple gradient steps?

This is highly desirable for sample efficiency. However, because the gradient expectation is explicitly over the current policy, reusing old samples quickly becomes biased and unstable.

Two approaches exist:

1. Gather trajectories from the current policy (stable, on-policy)  
2. Use old trajectories with importance sampling (unstable, high variance)

This motivates trust-region and proximal methods that safely reuse data.


### Choosing a Step Size

Policy gradients perform stochastic gradient ascent:

$$
\theta_{k+1} = \theta_k + \alpha_k \hat{g}_k
$$

The step size $\alpha_k$ is critical:

- If the step is too large, the policy can change drastically →  
  performance collapse (even catastrophic).
- If the step is too small, learning becomes extremely slow.
- The “correct” step size depends heavily on the current parameters and landscape.

Even with adaptive optimizers (Adam, RMSProp) or gradient normalization, a single bad update can cause the agent to fall off a performance cliff.

Once a policy collapses, it may be very hard or impossible for vanilla policy gradient to recover.

### Small Parameter Changes ≠ Small Policy Changes

A deeper issue is that a small step in parameter space may produce a large change in the policy itself.

Example: Consider a two-action policy parameterized by a logistic function:

$$
\pi_\theta(a=1) = \sigma(\theta), \quad 
\pi_\theta(a=2) = 1 - \sigma(\theta)
$$

As $\theta$ changes slightly, the action probabilities may change dramatically:

- At $\theta = 4$, the policy is nearly deterministic.  
- At $\theta = 2$, action probabilities already shift significantly.  
- At $\theta = 0$, both actions become equally likely.

This illustrates an important point: Distance in parameter space does not correspond to distance in policy space.

A small update to $\theta$ can cause:

- Policy to become nearly deterministic  
- Policy to flip preference between actions  
- Large, unintended behavioral changes  
- Massive drops in performance

This is not just a step-size problem. It is a fundamental mismatch between how we update parameters and how policies behave.


Because small parameter updates can produce large changes in the policy, the key question becomes: How do we design a policy update rule that limits how much the policy itself changes?

This leads to trust-region ideas:

- Constrain the policy to not deviate too far from the previous version
- Measure distance between policies, not parameters
- Update safely using a surrogate objective
- Form the foundation for TRPO, PPO, and other modern policy-gradient RL algorithms


## 7. Policy Performance Difference Lemma

For two policies $\pi$ and $\pi'$:

$$
J(\pi') - J(\pi) = \frac{1}{1-\gamma}
\;\mathbb{E}_{s \sim d_{\pi'}, a \sim \pi'} [A_\pi(s,a)]
$$

where

$$
d_{\pi}(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t P(s_t = s \mid \pi)
$$

---

## 8. KL Divergence — Measuring Policy Change

For two policies:

$$
D_{KL}(\pi' || \pi)[s] =
\sum_{a} \pi'(a|s) \log \frac{\pi'(a|s)}{\pi(a|s)}
$$

If KL is small, the policy changed only slightly → safe trust-region update.

---

## 9. Monotonic Policy Improvement Bound

$$
J(\pi') - J(\pi) \ge
L_\pi(\pi') -
C \cdot \mathbb{E}_{s \sim d_\pi}[D_{KL}(\pi'||\pi)[s]]
$$

Where surrogate loss:

$$
L_\pi(\pi') =
\frac{1}{1-\gamma}
\mathbb{E}_{s,a\sim\pi}
\left[
\frac{\pi'(a|s)}{\pi(a|s)} A_\pi(s,a)
\right]
$$

Goal: Maximize $L_\pi(\pi')$ while keeping KL small.

---

## 10. Proximal Policy Optimization (PPO)

PPO improves vanilla policy gradient by restricting updates to avoid policy collapse.

### PPO Variant 1: KL Penalty Objective

$$
\theta_{k+1} =
\arg\max_\theta
\Big[
L_{\theta_k}(\theta) - \beta_k \bar{D}_{KL}(\theta||\theta_k)
\Big]
$$

Where:

$$
\bar{D}_{KL}(\theta || \theta_k) =
\mathbb{E}_{s \sim d_{\pi_k}} \big[ D_{KL}(\pi_\theta(\cdot|s) || \pi_{\theta_k}(\cdot|s)) \big]
$$

$\beta_k$ is adjusted dynamically to enforce KL trust region.

---

### PPO Variant 2: Clipped Surrogate Objective

Define probability ratio:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}
$$

Clipped objective:

$$
L^{CLIP}(\theta) = \mathbb{E}
\left[
\min
\left(
r_t(\theta) A_t,\;
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
$$

Intuition:  
Prevent large destructive policy updates by clipping probability ratios.

---

### PPO (Clipped) Algorithm

1. Collect trajectories using old policy $\pi_{\theta_k}$  
2. Estimate advantages $\hat{A}_t$  
3. Optimize $L^{CLIP}(\theta)$ using SGD for multiple epochs  
4. Update policy parameters  
5. Repeat  

---

## Final Summary

| Method | Key Idea | Pros | Cons |
|--------|----------|------|------|
| REINFORCE | MC return-based update | Simple, unbiased | Very high variance |
| Actor-Critic | TD baseline value | More efficient | Needs critic |
| Advantage Actor-Critic | Uses $A(s,a)$ | Best trade-off | Requires value estimate |
| PPO | Trust-region clipped objective | Stable, efficient | More complex |
| TRPO | Exact trust region | Strong theory | Hard to implement |

