# Proximal Gradient Algorithm

Many optimization problems involve **composite objectives** of the form:

$$
\min_{x \in \mathbb{R}^n} F(x) := f(x) + g(x)
$$

where:  

- $f(x)$ is **convex and differentiable** with a **Lipschitz continuous gradient** ($\nabla f$ exists and is $L$-Lipschitz).  
- $g(x)$ is **convex but possibly non-differentiable** (e.g., $\ell_1$-norm, indicator of a constraint set).  

This structure appears in many applications: LASSO ($f = \text{least squares}, g = \lambda \|x\|_1$), elastic net, constrained optimization, etc.

---

## 1. Motivation

- Standard gradient descent cannot handle $g(x)$ if it is non-differentiable.  
- Projected gradient descent works only if $g$ is an indicator function of a set.  
- The **proximal gradient method** generalizes both approaches and allows efficient updates even when $g$ is non-smooth.

---

## 2. Proximal Gradient Update

For step size $\eta > 0$, the **proximal gradient update** is:

$$
x_{t+1} = \text{prox}_{\eta g}\big(x_t - \eta \nabla f(x_t)\big)
$$

**Interpretation:**

1. Take a **gradient step** on the smooth part $f$:

$$
y_t = x_t - \eta \nabla f(x_t)
$$

2. Apply the **proximal operator** of $g$ to handle the non-smooth part:

$$
x_{t+1} = \text{prox}_{\eta g}(y_t)
$$

This ensures:

- $f(x)$ decreases via the gradient step.  
- $g(x)$ is accounted for via the proximal step.  

---

## 3. Step Size Selection

For convergence, the step size $\eta$ is typically chosen as:

$$
0 < \eta \le \frac{1}{L}
$$

where $L$ is the Lipschitz constant of $\nabla f$.  

- Smaller $\eta$ → conservative steps.  
- Larger $\eta$ may overshoot and break convergence guarantees.  
- Adaptive strategies (like **backtracking line search**) can also be used.

---

## 4. Algorithm (Proximal Gradient Method / ISTA)

**Input:** $x_0$, step size $\eta > 0$  

**Repeat** for $t = 0, 1, 2, \dots$:  

1. Compute gradient step:

$$
y_t = x_t - \eta \nabla f(x_t)
$$

2. Apply proximal operator:

$$
x_{t+1} = \text{prox}_{\eta g}(y_t)
$$

3. Check convergence (e.g., $\|x_{t+1} - x_t\| < \epsilon$).

---

## 5. Special Cases

| Non-smooth term $g(x)$ | Proximal operator $\text{prox}_{\eta g}(y)$ | Interpretation |
|----------------------------|----------------------------------------------|----------------|
| $\lambda \|x\|_1$       | Soft-thresholding: $\text{sign}(y_i)\max(|y_i| - \eta\lambda, 0)$ | Promotes sparsity |
| Indicator $I_{\mathcal{X}}(x)$ | Projection: $\text{Proj}_{\mathcal{X}}(y)$ | Constrained optimization |
| $\lambda \|x\|_2^2$     | Shrinkage: $y / (1 + 2\eta\lambda)$        | Smooth regularization |

---

## 6. Properties of Proximal Operators

Proximal operators have several useful **mathematical properties**:

#### **Non-expansiveness (Lipschitz continuity):**

$$
\|\text{prox}_{g}(x) - \text{prox}_{g}(y)\|_2 \le \|x - y\|_2, \quad \forall x, y
$$

#### **Firmly non-expansive:**

$$
\|\text{prox}_{g}(x) - \text{prox}_{g}(y)\|_2^2 \le \langle \text{prox}_{g}(x) - \text{prox}_{g}(y), x - y \rangle
$$

#### **Fixed point characterization:**

$$
x^\star = \text{prox}_{g}(x^\star - \eta \nabla f(x^\star)) \quad \Longleftrightarrow \quad 0 \in \nabla f(x^\star) + \partial g(x^\star)
$$

This shows that proximal gradient fixed points correspond to **optimality conditions** for composite convex functions.

#### **Translation property:**

$$
\text{prox}_{g}(x + c) = \text{prox}_{g(\cdot - c)}(x) + c
$$

#### **Separable for sums over coordinates:**

If $g(x) = \sum_i g_i(x_i)$, then

$$
\text{prox}_{g}(x) = \big( \text{prox}_{g_1}(x_1), \dots, \text{prox}_{g_n}(x_n) \big)
$$

This is why soft-thresholding works coordinate-wise.

---

## 7. Why Proximal Gradient Works

- The proximal gradient method **splits the objective** into smooth and non-smooth parts.  
- The gradient step moves toward minimizing $f(x)$ (smooth).  
- The proximal step moves toward minimizing $g(x)$ (structure or constraints).  
- Geometrically, the proximal operator finds a point **close to the gradient update** but also **reduces the non-smooth term**, ensuring convergence under convexity and Lipschitz continuity.  

- If $g = 0$, it reduces to **gradient descent**.  
- If $g$ is an indicator function, it reduces to **projected gradient descent**.  

---

## 8. Convergence

For convex $f$ and $g$:

$$
F(x_t) - F(x^\star) = \mathcal{O}\Big(\frac{1}{t}\Big)
$$

- Accelerated variants (like **FISTA**) improve the rate to $\mathcal{O}(1/t^2)$.  
- Requires convexity and Lipschitz continuity of $\nabla f$.
 