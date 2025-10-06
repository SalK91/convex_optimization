# Subgradient Method: Derivation and Convergence

We aim to minimize a **convex** function $f$, which may be **nonsmooth**:

$$
\min_{x \in \mathcal{X}} f(x),
$$

where $f$ is convex but may not be differentiable everywhere.


## 1. Subgradients

For nonsmooth convex functions, we use a **subgradient** $g_t \in \partial f(x_t)$ at iteration $t$, where $\partial f(x_t)$ is the **subdifferential** at $x_t$.

- For differentiable $f$, $\partial f(x_t) = \{\nabla f(x_t)\}$.
- For nonsmooth $f$, $\partial f(x_t)$ is a set of vectors satisfying:

$$
f(y) \ge f(x_t) + \langle g_t, y - x_t \rangle, \quad \forall y \in \mathcal{X}.
$$

> **Intuition**: A subgradient generalizes the concept of gradient for nonsmooth convex functions. It points in a direction that does not decrease the function.

  

## 2. Subgradient Update Rule

The **projected subgradient method** updates:

$$
x_{t+1} = \Pi_{\mathcal{X}} \big( x_t - \eta_t g_t \big),
$$

where:

- $g_t \in \partial f(x_t)$ is a subgradient,  
- $\eta_t > 0$ is the step size (may vary with $t$),  
- $\Pi_{\mathcal{X}}$ denotes the projection onto the feasible set $\mathcal{X}$.

If $\mathcal{X} = \mathbb{R}^n$ (unconstrained), this reduces to:

$$
x_{t+1} = x_t - \eta_t g_t.
$$

 
## 3. Distance Recurrence

Let $x^\star$ be an optimal solution. Consider the squared distance to the optimum:

$$
\|x_{t+1} - x^\star\|^2 = \|x_t - \eta_t g_t - x^\star\|^2
$$

Expanding:

$$
\|x_{t+1} - x^\star\|^2 = \|x_t - x^\star\|^2 - 2\eta_t \langle g_t, x_t - x^\star \rangle + \eta_t^2 \|g_t\|^2
$$

Since $g_t$ is a subgradient of convex $f$:

$$
f(x_t) - f(x^\star) \le \langle g_t, x_t - x^\star \rangle
$$

Substitute into the distance expansion:

$$
\|x_{t+1} - x^\star\|^2 \le \|x_t - x^\star\|^2 - 2\eta_t \big(f(x_t) - f(x^\star)\big) + \eta_t^2 \|g_t\|^2
$$

 
## 4. Rearranging for Function Suboptimality

Rewriting:

$$
f(x_t) - f(x^\star) \le \frac{\|x_t - x^\star\|^2 - \|x_{t+1} - x^\star\|^2}{2 \eta_t} + \frac{\eta_t}{2} \|g_t\|^2
$$

> **Intuition**: The suboptimality is bounded by:

> 1. The decrease in squared distance to the optimum.  
> 2. A term depending on step size and the subgradient norm.



## 5. Summing Over Iterations

Sum over $t = 0, \dots, T-1$:

$$
\sum_{t=0}^{T-1} \big(f(x_t) - f(x^\star)\big) \le \frac{\|x_0 - x^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=0}^{T-1} \|g_t\|^2
$$

Assume $\|g_t\| \le G$ and **constant step size** $\eta$:

$$
\sum_{t=0}^{T-1} \big(f(x_t) - f(x^\star)\big) \le \frac{\|x_0 - x^\star\|^2}{2\eta} + \frac{\eta G^2 T}{2}
$$

Divide by $T$ to bound the **average iterate** $\bar{x}_T = \frac{1}{T} \sum_{t=0}^{T-1} x_t$:

$$
f(\bar{x}_T) - f(x^\star) \le \frac{\|x_0 - x^\star\|^2}{2 \eta T} + \frac{\eta G^2}{2}
$$

 

## 6. Step Size Choice and Convergence Rate

Choosing a **diminishing step size**:

$$
\eta_t = \frac{R}{G \sqrt{T}}, \quad R = \|x_0 - x^\star\|
$$

gives the classic subgradient **sublinear convergence rate**:

$$
f(\bar{x}_T) - f(x^\star) \le \frac{R G}{\sqrt{T}}
$$

- $x^\star$ is an optimal solution.  
- $\bar{x}_T = \frac{1}{T}\sum_{t=0}^{T-1} x_t$ is the average iterate.  
- $R = \|x_0 - x^\star\|$, distance to optimum.  
- $G$ bounds the subgradients: $\|g_t\| \le G$.

> **Implication**: The convergence rate is **sublinear**, $O(1/\sqrt{T})$.  
> Unlike gradient descent, subgradient method **cannot achieve linear convergence** without additional assumptions (like strong convexity and smoothness).

 

## 7. Practical Remarks

1. **Step size selection** is crucial:
   - Diminishing step sizes ensure convergence.  
   - Constant step size may lead to oscillations near optimum.

2. **Averaging iterates** ($\bar{x}_T$) improves convergence guarantees.  

3. **Robustness**: Works for nonsmooth convex functions where gradient does not exist.

4. **Slower than gradient descent**: $O(1/\sqrt{T})$ vs $O(1/T)$ or linear for smooth strongly convex functions.

