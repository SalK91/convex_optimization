# Newton’s Method 

Optimization algorithms can be broadly classified based on the type of information they use. **Gradient Descent (GD)** uses only **first-order information** (the gradient), while **Newton’s Method** incorporates **second-order information** via the **Hessian matrix**, allowing it to adaptively rescale updates based on local curvature.


## 1. Taylor Expansions and Local Models

### 1.1 Gradient Descent — Linear Approximation

Gradient Descent is based on a **first-order Taylor approximation** of $f$ around $x$:

$$
f(x + d) \approx f(x) + \nabla f(x)^\top d
$$

To prevent uncontrolled steps, we regularize with a quadratic trust term:

$$
\min_d \; \nabla f(x)^\top d + \frac{1}{2\eta}\|d\|^2
$$

This yields the solution:

$$
d = -\eta \nabla f(x), \quad x_{t+1} = x_t - \eta \nabla f(x_t)
$$

Thus, GD follows the **steepest descent direction** with respect to the **Euclidean metric**.

 
### 1.2 Newton’s Method — Quadratic Approximation

Newton's Method instead constructs a **second-order Taylor approximation**:

$$
f(x + d) \approx f(x) + \nabla f(x)^\top d + \frac{1}{2} d^\top \nabla^2 f(x) d
$$

Minimizing this quadratic model:

$$
\min_d \; \nabla f(x)^\top d + \frac{1}{2} d^\top H(x) d \quad \text{with} \quad H(x) = \nabla^2 f(x)
$$

Setting derivative to zero gives:

$$
H(x)d = -\nabla f(x) \quad \Rightarrow \quad d = -H(x)^{-1} \nabla f(x)
$$

Update rule:

$$
x_{t+1} = x_t - H(x_t)^{-1} \nabla f(x_t)
$$

Newton’s step directly targets the minimizer of the local quadratic model, adjusting the direction based on curvature.

 

## 2. Smoothness and Strong Convexity Assumptions

Assume:

- $f$ is **$\beta$-smooth**: $\|\nabla f(x) - \nabla f(y)\| \le \beta \|x - y\|$  
- $f$ is **$\alpha$-strongly convex**: $f(y) \ge f(x) + \nabla f(x)^\top (y-x) + \frac{\alpha}{2}\|y-x\|^2$

These assumptions enable convergence analysis for both GD and Newton’s method.

 
## 3. Convergence Rates

### 3.1 Gradient Descent — Linear Convergence

For smooth and strongly convex $f$:

$$
f(x_t) - f(x^*) \le \left(1 - \frac{\alpha}{\beta}\right)^t (f(x_0) - f(x^*))
$$

- Convergence is **linear**.
- Each iteration reduces the error by a **constant factor**.

 
### 3.2 Newton’s Method — Quadratic Convergence

Assuming $f$ has **Lipschitz-continuous Hessian** and $x_t$ is sufficiently close to $x^*$:

$$
\|x_{t+1} - x^*\| \le C \|x_t - x^*\|^2
$$

- Convergence is **quadratic**.
- The number of **correct digits doubles after each iteration**.

 
## 4. Damped vs. Non-Damped Newton Method

### 4.1 Classical (Non-Damped) Newton

Uses the raw step:

$$
x_{t+1} = x_t - H(x_t)^{-1} \nabla f(x_t)
$$

- Extremely fast near optimum.
- However, if $x_t$ is far from $x^*$, the quadratic model may be inaccurate, causing divergence.

### 4.2 Damped Newton Method

To improve global stability, introduce a **damping factor** $\lambda_t \in (0,1]$:

$$
x_{t+1} = x_t - \lambda_t H(x_t)^{-1} \nabla f(x_t)
$$

- $\lambda_t$ is often chosen via **line search** to ensure sufficient decrease.
- Damping makes Newton’s method **globally convergent**, transitioning to **full Newton steps** when close to optimum.

**Insight:**  
- Non-damped Newton is extremely fast but potentially unstable.
- Damped Newton trades some speed for **robustness during early iterations**.

---

## 5. Affine Transformation Perspective — Geometry Matters

A powerful perspective is to analyze both methods under an affine change of coordinates:

$$
x = Ay + b, \quad A \in \mathbb{R}^{d \times d} \text{ invertible}
$$

### Effect on Gradient Descent

- Gradient transforms as $\nabla_y f = A^\top \nabla_x f$.
- Update becomes **dependent on the coordinate system**.
- GD is **not affine invariant** — performance heavily depends on scaling and conditioning.

### Effect on Newton’s Method

- Hessian transforms as $H_y = A^\top H_x A$.
- Update:

$$
y_{t+1} = y_t - (A^\top H_x A)^{-1} (A^\top \nabla_x f)
$$

which simplifies to the **same geometric step** as in original coordinates.

> **Newton’s Method is affine invariant** — it adapts to the local curvature, effectively removing ill-conditioning.

**Interpretation:**

- Gradient Descent uses **isotropic steps** — same metric in every direction.
- Newton Method uses the **Hessian-induced metric**, effectively **rescaling space** so level sets become spherical.

 
## 6. Computational Trade-Off Summary

| Method | Local Model | Metric Used | Per-Step Cost | Convergence | Affine Invariance |
|--------|------------|------------|--------------|------------|------------------|
| GD | Linear | Euclidean ($I$) | $O(d)$ | Linear | No |
| Newton | Quadratic | Hessian-inverse ($H^{-1}$) | $O(d^3)$ | Quadratic | Yes |
| Damped Newton | Quadratic + Line Search | Hessian-inverse | $O(d^3)$ + line search | Quadratic near optimum, stable globally | Yes |

 
## 7. Strategic Use

- Use **GD/SGD** for large-scale problems or as a **warm start**.
- Switch to **(damped) Newton or quasi-Newton methods** when close to optimal region.
- The Hessian captures **intrinsic geometry**, removing conditioning issues that slow down GD.
