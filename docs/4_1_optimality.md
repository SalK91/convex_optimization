# D.3 First-Order Optimality Conditions in Convex Optimization

Convex optimization enjoys a powerful guarantee: **every local minimum is a global minimum**. This section provides a unified framework for checking optimality in both **unconstrained** and **constrained** settings — whether the function is smooth or nonsmooth.

These optimality criteria are especially useful when:

- Gradients don't exist (e.g., hinge loss, $\ell_1$ norm)
- You're working with constraints (e.g., regularization, feasible regions)
- You want a geometric understanding of **what makes a point optimal**


---

## Unconstrained Convex Problems

### 1. Differentiable Objective

Let $f:\mathbb{R}^n \to \mathbb{R}$ be convex and differentiable. The unconstrained problem:

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

has **optimal solution** $\hat{x}$ if and only if:

$$
\nabla f(\hat{x}) = 0
$$

📌 This is the **first-order condition**: the gradient vanishes at the minimum.

**Examples**:

- *Quadratic*: $f(x) = x^2 - 4x + 7$  
  $\nabla f(x) = 2x - 4$ ⇒ $\hat{x} = 2$

- *Least squares*:  
  $f(x) = \sum_i (x - y_i)^2$ ⇒ $\hat{x} = \frac{1}{n}\sum y_i$ (the **mean**)

---

### 2. Nondifferentiable Objective

Let $f$ be convex but not necessarily differentiable (e.g., $f(x) = |x|$).

Then the optimality condition becomes:

$$
0 \in \partial f(\hat{x})
$$

where $\partial f(x)$ is the **subdifferential** (see [A.7: Subgradients](#a7-subgradients)).

**Examples**:

- *Absolute loss*: $f(x) = \sum_i |x - y_i|$ ⇒ optimal $x$ is the **median**  
- *Max function*: $f(x) = \max(x - 1, 2 - x)$  
  Subdifferential:  
  $$
  \partial f(x) = \begin{cases}
  \{-1\} & x < 1.5 \\\
  [-1, 1] & x = 1.5 \\\
  \{1\} & x > 1.5
  \end{cases}
  $$  
  Minimum occurs at $x = 1.5$

---

## Constrained Convex Problems

Let $f: \mathbb{R}^n \to \mathbb{R}$ be convex, and let $\mathcal{X} \subseteq \mathbb{R}^n$ be a **convex feasible set**. Consider:

$$
\min_{x \in \mathcal{X}} f(x)
$$

Now, optimality depends on where $\hat{x}$ lies relative to $\mathcal{X}$.

---

### 1. Interior Points

If $\hat{x}$ lies **strictly inside** the feasible set:

$$
\hat{x} \in \text{int}(\mathcal{X})
$$

Then the constraint has no effect locally. Optimality is:

$$
0 \in \partial f(\hat{x})
$$

Same as the unconstrained case.

---

### 2. Boundary Points

When $\hat{x}$ lies on the **boundary** of $\mathcal{X}$, the optimality condition changes. You **cannot move in all directions**, only within the feasible set.

A point $\hat{x}$ is optimal if and only if:

$$
- \nabla f(\hat{x}) \in N_{\mathcal{X}}(\hat{x})
$$

Where $N_{\mathcal{X}}(\hat{x})$ is the **normal cone** (see [A.6: Projections](#a6-projections)) — the set of vectors pointing **outward from** $\mathcal{X}$ at $\hat{x}$.

**Geometric version**: For all feasible directions $d \in T_{\mathcal{X}}(\hat{x})$ (the **tangent cone**):

$$
\nabla f(\hat{x})^\top d \ge 0
$$

Interpretation: The gradient points **away from** the feasible region. Any feasible move will not decrease $f$.

---

### 3. Unified Compact Form

The condition below covers **both** interior and boundary cases:

$$
0 \in \partial f(\hat{x}) + N_{\mathcal{X}}(\hat{x})
$$

This is the **general optimality condition** for constrained convex problems — including nonsmooth cases.

---

## Intuition: Tangent vs Normal Cones

Imagine standing on the boundary of a convex set $\mathcal{X}$. You can only step along certain directions — these form the **tangent cone**. The **normal cone** consists of vectors that oppose all those directions.

At an optimum, the gradient is aligned with the **normal cone**:  
Any movement within $\mathcal{X}$ increases or keeps the objective the same.

> **Interior optimality:** gradient is zero  
> **Boundary optimality:** gradient pushes you *outward*, and any move into $\mathcal{X}$ would worsen the objective

---

## Example: Quadratic with Constraint

Consider:

$$
\min f(x) = x^2 \quad \text{s.t. } x \ge 1
$$

- Feasible set: $\mathcal{X} = [1, \infty)$  
- Gradient: $\nabla f(x) = 2x$

Check:

- Unconstrained minimizer is $x = 0$ → infeasible  
- Try $x = 1$:
  $$
  -\nabla f(1) = -2 \in \mathbb{R}_+ = N_{\mathcal{X}}(1)
  $$
  ✔️ Satisfied — $x = 1$ is optimal

---

## Summary

| Case | Optimality Condition |
|------|----------------------|
| Unconstrained, smooth | $\nabla f(\hat{x}) = 0$ |
| Unconstrained, nonsmooth | $0 \in \partial f(\hat{x})$ |
| Constrained | $0 \in \partial f(\hat{x}) + N_{\mathcal{X}}(\hat{x})$ |

These conditions appear throughout convex optimization — in projection methods (see [G.2](#g2-proximal-and-projected-descent)), duality theory (Section D), and in practical solvers.

---

📚 **Next:** We extend this understanding to *Fenchel duality*, *conjugate functions*, and structured dual problems in **D.4 and D.5**.
