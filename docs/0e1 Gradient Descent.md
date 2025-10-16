# Gradient Descent: Derivation and Convergence

Gradient descent is one of the most fundamental algorithms in optimization and machine learning. It forms the backbone of training neural networks, logistic regression, matrix factorization, and many other models.

## Problem Setup

We aim to minimize a differentiable function over a feasible convex set $\mathcal{X}$:

$$
\min_{x \in \mathcal{X}} f(x).
$$

At each iteration $t$, we hold a current iterate $x_t$ and wish to take a step that reduces the objective. But instead of minimizing $f$ directly (which may be complex), we construct a **local surrogate model**.


## Local Linear Approximation (First-order Model)

Around $x_t$, we approximate $f$ using its Taylor expansion:

$$
f(x) \approx f(x_t) + \langle \nabla f(x_t), x - x_t \rangle.
$$

**Intuition:**  
- We assume $f$ behaves approximately like its tangent plane near $x_t$.  
- If we were to minimize just this linear model, we would move **infinitely far** in the direction of **steepest descent** $-\nabla f(x_t)$, which is not realistic or stable.

This motivates adding a **locality restriction** — we trust the linear approximation **near** $x_t$, not globally.

## Adding a Quadratic Regularization Term (Trust Region View)

To prevent taking arbitrarily large steps, we add a quadratic penalty for moving away from $x_t$:

$$
f(x) \approx f(x_t) + \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{2\eta} \|x - x_t\|^2,
$$

where $\eta > 0$ is the **learning rate** or **step size**.

**Geometric Interpretation:**
- The linear term pulls $x$ in the steepest descent direction.
- The quadratic term acts like a **trust region**, discouraging large deviations from $x_t$.
- $\eta$ trades off **aggressive progress** vs **stability**:
  - Small $\eta$ → cautious updates.
  - Large $\eta$ → bold updates (risk of divergence).


## Deriving the Gradient Descent Update

We define the next iterate as the minimizer of the surrogate objective:

$$
x_{t+1} = \arg\min_{x \in \mathcal{X}} \Big[ f(x_t) + \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{2\eta} \|x - x_t\|^2 \Big].
$$

Ignoring the constant term $f(x_t)$ and differentiating w.r.t. $x$:

$$
\nabla f(x_t) + \frac{1}{\eta}(x - x_t) = 0
$$

Solving:

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

> **Gradient Descent Update:**
> $$
> \boxed{x_{t+1} = x_t - \eta \nabla f(x_t)}
> $$


## Convergence Analysis

To analyze convergence, we assume:

### Smoothness (Lipschitz Gradient)

$$
\|\nabla f(x) - \nabla f(y)\| \le L \|x - y\|, \quad \forall x, y.
$$

This says the gradient does not change too abruptly. Most ML objectives satisfy this.

### Strong Convexity 
If, in addition, $f$ is **$\mu$-strongly convex**, then:

$$
f(y) \ge f(x) + \langle \nabla f(x), y-x \rangle + \frac{\mu}{2} \|y-x\|^2.
$$

This implies $f$ has a **unique minimizer $x^\*$** and its level sets are **bowl-shaped**, not flat.


## Descent Lemma: Why Gradient Descent Decreases $f$

For an $L$-smooth function,

$$
f(x_{t+1}) \le f(x_t) + \langle \nabla f(x_t), x_{t+1}-x_t \rangle + \frac{L}{2} \|x_{t+1}-x_t\|^2.
$$

Using $x_{t+1} = x_t - \eta \nabla f(x_t)$:

$$
\begin{aligned}
f(x_{t+1}) 
&\le f(x_t) - \eta \|\nabla f(x_t)\|^2 + \frac{L \eta^2}{2} \|\nabla f(x_t)\|^2 \\
&= f(x_t) - \left( \eta - \frac{L\eta^2}{2} \right) \|\nabla f(x_t)\|^2.
\end{aligned}
$$

> If $\eta \le \frac{1}{L}$, then the decrease term is positive ⇒ **every step reduces the objective**.



## Convergence Rates

### Convex but not strongly convex

$$
f(x_T) - f(x^*) \le \frac{L \|x_0 - x^*\|^2}{2T} \quad \Rightarrow \quad O(1/T) \text{ convergence}.
$$

This is called **sublinear convergence**.



### Strongly Convex Case: Linear Convergence

If $f$ is $\mu$-strongly convex and $\eta = \frac{2}{\mu + L}$:

$$
\|x_{t+1} - x^*\| \le \left( \frac{L - \mu}{L + \mu} \right) \|x_t - x^*\|
$$

This gives:

> $$
> \|x_T - x^*\| \le \left( \frac{L - \mu}{L + \mu} \right)^T \|x_0 - x^*\| \quad \Rightarrow \quad \text{Linear (geometric) convergence}.
> $$

Meaning: **Error shrinks by a constant factor every iteration**.

##  Summary and ML Interpretation

| Assumption on $f$         | Convergence Rate | Typical ML Scenario |
|-------------------------|------------------|--------------------|
| Convex + Smooth         | $O(1/T)$         | Unregularized logistic regression, basic convex losses |
| Strongly Convex + Smooth | $O(\rho^T)$ (linear) | L2-regularized models, ridge regression |

> **Key Takeaway:**  
> Gradient descent is not just a heuristic — it arises from a principled **local approximation + trust region** perspective and enjoys strong convergence guarantees under mild assumptions.

---

Let me know — do you want me to follow this same **style and depth** for **Projected Gradient Descent, Accelerated GD (Nesterov), and Stochastic GD next?** 🚀
