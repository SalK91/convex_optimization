# Chapter 6: Nonsmooth Convex Optimization – Subgradients

Many of the most important convex functions are not differentiable everywhere:

- $\|x\|_1 = \sum_i |x_i|$ has corners at $x_i = 0$,
- $f(x) = \max\{a_1^\top x + b_1, \dots, a_k^\top x + b_k\}$ is piecewise affine,
- the hinge loss $\max\{0, 1 - y w^\top x\}$ (used in SVMs) is not smooth at the kink.

We still want optimality conditions, descent methods, and dual variables. Subgradients give us that (Hiriart-Urruty and Lemaréchal, 2001; Boyd and Vandenberghe, 2004).

---

## 6.1 The problem with $\nabla f$

For a convex but nonsmooth $f$, the usual condition “$\nabla f(x^*) = 0$” may not make sense, because $\nabla f(x^*)$ may not exist.

But geometrically, convex functions still have supporting hyperplanes at every point. That is the key.

---

## 6.2 Subgradients and the subdifferential

Let $f : \mathbb{R}^n \to \mathbb{R}$ be convex. A vector $g \in \mathbb{R}^n$ is called a **subgradient** of $f$ at $x$ if, for all $y$,
$$
f(y) \ge f(x) + g^\top (y - x).
$$

Interpretation:

- The affine function $y \mapsto f(x) + g^\top (y-x)$ is a global underestimator of $f$.
- $g$ defines a supporting hyperplane to the epigraph of $f$ at $(x,f(x))$.

The set of all subgradients of $f$ at $x$ is called the **subdifferential** of $f$ at $x$:
$$
\partial f(x) = \{ g : f(y) \ge f(x) + g^\top (y-x) \ \forall y \}.
$$

If $f$ is differentiable at $x$, then
$$
\partial f(x) = \{ \nabla f(x) \}.
$$

If $f$ is not differentiable at $x$, $\partial f(x)$ is typically a nonempty convex set.

---

## 6.3 Examples

### 6.3.1 Absolute value in 1D
Let $f(t) = |t|$.

- For $t>0$, $\partial f(t) = \{1\}$.
- For $t<0$, $\partial f(t) = \{-1\}$.
- For $t=0$, 
  $$
  \partial f(0) = [-1, 1].
  $$

At the kink, any slope between $-1$ and $1$ is a valid supporting line from below.

### 6.3.2 $\ell_1$ norm
For $f(x) = \|x\|_1 = \sum_i |x_i|$, we have
$$
\partial \|x\|_1 = \{ g \in \mathbb{R}^n : g_i \in \partial |x_i| \}.
$$
So

- if $x_i > 0$, then $g_i = 1$,
- if $x_i < 0$, then $g_i = -1$,
- if $x_i = 0$, then $g_i \in [-1,1]$.

This is exactly what shows up in LASSO optimality conditions in statistics.

### 6.3.3 Pointwise max of affine functions
Let
$$
f(x) = \max_{i=1,\dots,k} (a_i^\top x + b_i).
$$

If a single index $i^*$ achieves the max at $x$, then
$$
\partial f(x) = \{ a_{i^*} \}.
$$

If multiple $i$ are tied at the max, then
$$
\partial f(x) = \mathrm{conv}\{ a_i : i \text{ active at } x \},
$$
the convex hull of all active slopes.

---

## 6.4 Subgradient optimality condition

Suppose we want to solve the unconstrained convex minimisation problem

$$
\min_x f(x),
$$

Then a point $x^*$ is optimal if and only if
$$
0 \in \partial f(x^*).
$$

This is the nonsmooth analogue of $\nabla f(x^*) = 0$ (Boyd and Vandenberghe, 2004). It says:
> at the minimiser, there is no subgradient pointing into a direction that would reduce $f$.

Equivalently: every subgradient "pushes up" from $x^*$.

---

## 6.5 Subgradient calculus (useful rules)

If $f$ and $g$ are convex:

- $\partial (f+g)(x) \subseteq \partial f(x) + \partial g(x)$, i.e.
  $$
  \partial (f+g)(x)
  \subseteq
  \{ u+v : u \in \partial f(x),\ v \in \partial g(x) \}.
  $$

- If $A$ is a matrix and $h(x) = f(Ax)$, then
  $$
  \partial h(x) = A^\top \partial f(Ax).
  $$

- If $f(x) = \max_i f_i(x)$ and each $f_i$ is convex, then
  $$
  \partial f(x) = \mathrm{conv}\{ \partial f_i(x) : i \text{ active at } x \}.
  $$

These rules make it possible to compute subgradients of complicated nonsmooth objectives.

---

## 6.6 Connection to duality and conjugates

For a convex function $f$, its **convex conjugate** (also called Legendre–Fenchel transform) is
$$
f^*(y) = \sup_x \big( y^\top x - f(x) \big).
$$

A deep fact (Hiriart-Urruty and Lemaréchal, 2001) is:
$$
y \in \partial f(x)
\quad \Longleftrightarrow \quad
x \in \partial f^*(y).
$$

This symmetry is the algebraic heart of Lagrangian duality, and it reappears when we discuss Fenchel duality in Chapter 8 and Appendix B.

---

## 6.7 Takeaways

1. Subgradients generalise gradients to nondifferentiable convex functions.
2. Optimality in convex, nonsmooth problems is: $0 \in \partial f(x^*)$.
3. Subdifferentials are convex sets; at kinks they become intervals or polytopes.
4. Subgradients are geometrically supporting hyperplanes to the epigraph.

In the next chapter, we apply all of this to constrained optimisation: KKT.


<!-- # Subgradients and Subdifferentials

Subgradients extend the notion of gradients to **nonsmooth convex functions** – allowing us to optimize functions that are not differentiable. This is crucial in machine learning, where many popular models (e.g., Lasso, SVMs) involve nonsmooth objectives like the $\ell_1$ norm or hinge loss. Subgradients provide the foundation for:

- First-order methods in nonsmooth convex optimization (e.g., subgradient descent, proximal methods)  
- Optimality conditions in convex analysis  
- Connecting primal and dual formulations via Fenchel conjugacy  
- Modeling sparsity and structured regularization via nonsmooth functions  

Subgradients generalize the concept of “directional slope” – they represent **supporting hyperplanes** that touch a convex function from below. Even when no unique gradient exists, the subgradient tells us a direction along which the function does not decrease.


## Formal Definition and Geometric Interpretation

Let $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ be a convex function. A vector $g \in \mathbb{R}^n$ is called a **subgradient** of $f$ at point $x$ if:

$$
f(y) \ge f(x) + \langle g, y - x \rangle \quad \forall y \in \mathbb{R}^n
$$

This inequality defines a **supporting hyperplane** to the graph of $f$ at $x$: the linear function $y \mapsto f(x) + \langle g, y - x \rangle$ underestimates $f$ globally. The set of all such subgradients is called the **subdifferential** of $f$ at $x$:

$$
\partial f(x) = \{ g \in \mathbb{R}^n \mid f(y) \ge f(x) + \langle g, y - x \rangle \;\; \forall y \}
$$

Key geometric interpretation:
- The subdifferential $\partial f(x)$ is a **convex set**.
- At differentiable points, $\partial f(x) = \{\nabla f(x)\}$ (singleton).
- At nondifferentiable points, $\partial f(x)$ typically contains multiple vectors, each defining a valid “direction of descent.”

 
## Properties of Subgradients

- If $f$ is **convex and differentiable** at $x$, then the gradient is the only subgradient: $\partial f(x) = \{ \nabla f(x) \}$  
- If $f$ is **nondifferentiable** at $x$, then $\partial f(x)$ is a **nonempty, convex, closed set**
- **Optimality condition**: $0 \in \partial f(x)$ ⇔ $x$ is a global minimizer of $f$
- Subgradients play a central role in **duality**: $g \in \partial f(x) \iff x \in \partial f^*(g)$ (Fenchel duality)

 
## Practical Usage and Algorithms

Subgradients enable optimization over nondifferentiable functions. They serve in:

### 1. **Subgradient Descent Algorithm**
For minimizing convex (possibly nonsmooth) functions:
$$
x_{k+1} = x_k - \alpha_k g_k \quad \text{where } g_k \in \partial f(x_k)
$$
- Step sizes $\alpha_k$ must typically decrease (e.g. $\alpha_k = \frac{1}{k}$) to ensure convergence.
- Subgradient descent does **not require differentiability**.
- Used widely in training ML models with $\ell_1$, hinge, or piecewise-linear losses.

### 2. **Checking Optimality**
- A point $x^\star$ is **optimal** if and only if $0 \in \partial f(x^\star)$.
- In constrained problems, optimality involves subdifferentials of the objective and constraint indicator functions.

### 3. **Duality and Fenchel Conjugates**
- Subgradients relate the primal and dual: if $f$ and $f^*$ are Fenchel conjugates, then:
  $$
  g \in \partial f(x) \iff x \in \partial f^*(g)
  $$
- This is foundational in dual decomposition, Lagrangian methods, and proximal algorithms.

 
## Examples and Intuition

### **Example 1: Absolute Value Function**
Let $f(x) = |x|$

- Differentiable for $x\ne 0$, not at $x=0$
- Subgradients:
  $$
  \partial f(x) =
  \begin{cases}
  \{1\} & x > 0 \\
  \{-1\} & x < 0 \\
  [-1, 1] & x = 0
  \end{cases}
  $$
- Geometric picture: any line with slope in $[-1,1]$ touches the graph of $|x|$ at $x=0$ from below.

---

### **Example 2: $\ell_1$ Norm**
Let $f(x) = \|x\|_1 = \sum_i |x_i|$

- Subdifferential:
  $$
  \partial \|x\|_1 =
  \left\{ g \in \mathbb{R}^n \;\middle|\;
  g_i = 
  \begin{cases}
  \text{sign}(x_i), & x_i \ne 0 \\
  \in [-1, 1], & x_i = 0
  \end{cases}
  \right\}
  $$
- Useful for modeling **sparsity**: at $x_i = 0$, multiple subgradients exist → corresponds to **soft-thresholding** in proximal operators.

---

### **Example 3: Indicator Function of a Convex Set**

Let $f(x) = \delta_C(x)$ (0 if $x \in C$, $+\infty$ otherwise), where $C$ is convex.

- Subdifferential:
  $$
  \partial \delta_C(x) =
  \begin{cases}
  \{ g \mid \langle g, y - x \rangle \le 0 \quad \forall y \in C \} & \text{if } x \in C \\
  \emptyset & \text{otherwise}
  \end{cases}
  $$
- This is the **normal cone** to $C$ at $x$.
- Important in constrained optimization – subgradients of constraint sets encode geometry of feasible directions.

---

## Applications in Machine Learning

- **Nonsmooth Optimization**:  
  Subgradient descent and proximal algorithms solve problems with nonsmooth losses (hinge, absolute error, $\ell_1$ penalties).  
  Examples:
  - Lasso ($\ell_1$ regularization): $\min \|Xw - y\|_2^2 + \lambda \|w\|_1$
  - SVM (hinge loss): $\min \frac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i w^T x_i)$

- **Regularization and Sparsity**:  
  Subgradients characterize the behavior of norms at the origin – leading to sparsity in optimal solutions (especially $\ell_1$ norms).

- **Duality and Proximal Methods**:  
  Many primal-dual algorithms rely on subgradients for deriving dual objectives and implementing updates via proximal operators.

- **Composite Optimization**:  
  Problems of the form $f(x) + g(x)$, where $f$ is smooth and $g$ is nonsmooth (e.g. Lasso), rely on subgradients of $g$.

---

## Key Takeaways for Practitioners

- Subgradients generalize gradients to **nondifferentiable convex functions**
- The **subdifferential** $\partial f(x)$ contains all valid descent directions
- If $0 \in \partial f(x)$, then $x$ is a **global minimizer**
- Subgradients allow us to:
  - Implement **subgradient descent**
  - Analyze **optimality** in nonsmooth problems
  - Handle **constraints** via indicator function subgradients
  - Relate **primal and dual variables** using conjugate functions
- They are especially critical for machine learning models involving $\ell_1$ regularization, hinge losses, robust formulations, and more -->
