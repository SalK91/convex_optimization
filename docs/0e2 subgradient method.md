# Subgradient Method: Derivation, Geometry, and Convergence

Let's consider the problem of minimizing a **convex**, possibly **nonsmooth**, function:

$$
\min_{x \in \mathcal{X}} f(x),
$$

where $f$ may not be differentiable everywhere (e.g., hinge loss, $L_1$ norm, ReLU penalties—common in ML). Classical gradient descent cannot be applied directly, so we use **subgradients**.

---

### Subgradients and Geometric Meaning

A **subgradient** $g_t \in \partial f(x_t)$ is any vector that supports the function from below:

$$
f(y) \ge f(x_t) + \langle g_t, y - x_t \rangle, \quad \forall y \in \mathcal{X}.
$$

- When $f$ is smooth, $\partial f(x_t) = \{\nabla f(x_t)\}$ and $g_t$ coincides with the gradient.
- When $f$ is nonsmooth (like $|x|$ at $x=0$), the subdifferential $\partial f(x_t)$ is a **set** of valid slopes.
- Intuitively, any subgradient defines a **supporting hyperplane** that lies below the graph of $f$ and touches it at $x_t$.

This generalization allows us to move in a **descent direction** even when a unique gradient does not exist.

---

### Subgradient Update and Projection View

The update rule of the projected subgradient method is:

$$
x_{t+1} = \Pi_{\mathcal{X}} \big( x_t - \eta_t g_t \big),
$$

where:
- $g_t \in \partial f(x_t)$ is a valid subgradient,
- $\eta_t > 0$ is the step size,
- $\Pi_{\mathcal{X}}$ denotes projection onto $\mathcal{X}$ to ensure feasibility.

If $\mathcal{X} = \mathbb{R}^n$ (unconstrained case), projection disappears:

$$
x_{t+1} = x_t - \eta_t g_t.
$$

> **Geometric insight:** we move in the direction of a subgradient and then "snap back" to the feasible region if needed. This is analogous to gradient descent but more flexible, tolerating kinks in the objective.

---

### Distance Analysis and Role of Convexity

Let $x^\star$ be an optimal solution. Consider the squared distance after an update:

$$
\|x_{t+1} - x^\star\|^2 = \|x_t - \eta_t g_t - x^\star\|^2.
$$

Expanding the norm:

$$
\|x_{t+1} - x^\star\|^2 = \|x_t - x^\star\|^2 - 2\eta_t \langle g_t, x_t - x^\star \rangle + \eta_t^2 \|g_t\|^2.
$$

By convexity of $f$:

$$
f(x_t) - f(x^\star) \le \langle g_t, x_t - x^\star \rangle.
$$

Substitute this into the distance inequality to relate **movement** to **function decrease**:

$$
\|x_{t+1} - x^\star\|^2 \le \|x_t - x^\star\|^2 - 2\eta_t \big(f(x_t) - f(x^\star)\big) + \eta_t^2 \|g_t\|^2.
$$

---

### Bounding Suboptimality

Rearranging gives a direct bound on how far we are from optimum in function value:

$$
f(x_t) - f(x^\star) \le \frac{\|x_t - x^\star\|^2 - \|x_{t+1} - x^\star\|^2}{2 \eta_t} + \frac{\eta_t}{2} \|g_t\|^2.
$$

This shows a trade-off: large step sizes make faster jumps but increase the $\eta_t \|g_t\|^2$ error; small step sizes ensure precision but slow progress.

---

### Convergence Rate and Step Size Insight

Summing over $t = 0, \dots, T-1$ and assuming $\|g_t\| \le G$:

$$
\sum_{t=0}^{T-1} \big(f(x_t) - f(x^\star)\big) \le \frac{\|x_0 - x^\star\|^2}{2\eta} + \frac{\eta G^2 T}{2}.
$$

Dividing by $T$ and using $\bar{x}_T = \frac{1}{T} \sum_{t=0}^{T-1} x_t$:

$$
f(\bar{x}_T) - f(x^\star) \le \frac{\|x_0 - x^\star\|^2}{2 \eta T} + \frac{\eta G^2}{2}.
$$

Choosing:

$$
\eta_t = \frac{R}{G \sqrt{T}}, \quad R = \|x_0 - x^\star\|,
$$

gives the **sublinear convergence rate**:

$$
f(\bar{x}_T) - f(x^\star) \le \frac{R G}{\sqrt{T}} \quad \Rightarrow \quad O\left(\frac{1}{\sqrt{T}}\right).
$$

> This is slower than gradient descent on smooth functions ($O(1/T)$ or linear), reflecting the cost of nonsmoothness.

---

### Practical and ML Perspective

- Subgradients power many ML methods with nonsmooth penalties: **L1 regularization**, **hinge loss (SVMs)**, **ReLU activations**, **TV regularization in imaging**.
- **Step size choice is everything**. Too large → oscillation. Too small → stagnation.
- **Averaging iterates** improves convergence behavior and stability.
- Unlike gradient descent, the method **does not converge to a single point** but to a region near the optimum unless step size goes to zero.

> In high-dimensional ML models, the subgradient method's simplicity and robustness often outweigh its slower convergence rate—especially when structure (sparsity, hinge-like losses) matters more than raw speed.

