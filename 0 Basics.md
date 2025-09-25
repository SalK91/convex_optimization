---
title: Introduction
nav_order: 2
---

# Introduction
This is the intro chapter.

# Convex Optimisation: The Basics

Convex optimisation is one of the central pillars of modern applied mathematics, machine learning, and artificial intelligence. It provides a rich framework in which we can model problems, design algorithms, and guarantee performance. Unlike general non-convex optimisation, convex optimisation problems enjoy the key property that any local solution is also global. This makes them especially important in practice, where reliability, interpretability, and theoretical guarantees are valued.

## Mathematical Prerequisites

Before we can build convex optimisation tools, we need to review some core mathematical concepts from linear algebra and real analysis.

### Linear Algebra Essentials

- **Vector spaces and norms**: We work primarily in $\mathbb{R}^n$, the $n$-dimensional Euclidean space. The Euclidean norm is $\|x\|_2 = \sqrt{x^T x}$, but other norms, such as $\|x\|_1$ or $\|x\|_\infty$, are also important.
- **Inner products**: An inner product in $\mathbb{R}^n$ is $\langle x, y \rangle = x^T y$.  
- **Affine sets**: A set of the form $\{x \in \mathbb{R}^n : Ax = b\}$, where $A$ is a matrix and $b$ is a vector. Affine sets are the natural generalisation of lines and planes.
- **Positive semidefinite matrices**: A symmetric matrix $Q$ is positive semidefinite (PSD) if $x^T Q x \geq 0$ for all $x$. Quadratic forms with PSD matrices define convex functions.

### Real Analysis Essentials

- **Continuity and differentiability**: Convex functions are continuous on the interior of their domain. Differentiability gives access to gradient-based methods.
- **Convexity of sets**: A set $C$ is convex if for any $x_1, x_2 \in C$ and $\theta \in [0,1]$, we have $\theta x_1 + (1-\theta) x_2 \in C$.
- **Closed sets**: A set is closed if it contains all its limit points. The closure of a set is the smallest closed set containing it.
- **Extreme points**: A point in a convex set is extreme if it cannot be expressed as a convex combination of two other distinct points in the set. For polyhedra, extreme points correspond to vertices.


## Convex Sets and Geometry

### Convex Combination

A convex combination of $x_1, \dots, x_k$ is
$$
x = \sum_{i=1}^k \theta_i x_i, \quad \theta_i \geq 0, \quad \sum_{i=1}^k \theta_i = 1.
$$

This is simply a weighted average where weights are nonnegative and sum to 1.

### Convex Hull

The convex hull of a set $S$ is the collection of all convex combinations of points in $S$. It is the smallest convex set containing $S$.

**Geometric intuition**: Imagine stretching a rubber band around the points; the enclosed region is the convex hull.

### Hyperplanes and Half-spaces

- A **hyperplane** is the solution set of $a^T x = b$.
- A **half-space** is one side of a hyperplane, defined as $a^T x \leq b$ or $a^T x \geq b$.

These objects are convex and serve as building blocks in constraints.

### Separation and Supporting Hyperplanes

One of the most powerful results in convex geometry is the **separating hyperplane theorem**: two disjoint convex sets can be separated by a hyperplane. For a convex set $C$ and a point $x \notin C$, there exists a hyperplane that separates $x$ from $C$. This underpins duality theory in optimisation.

A supporting hyperplane touches a convex set at one or more points but does not cut through it.


## Convex Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if
$$
f(\theta x_1 + (1-\theta)x_2) \leq \theta f(x_1) + (1-\theta) f(x_2), \quad \forall x_1, x_2, \theta \in [0,1].
$$

### Properties

- **First-order condition**: $f$ is convex if and only if its domain is convex and
  $$
  f(y) \geq f(x) + \nabla f(x)^T (y - x), \quad \forall x, y.
  $$
- **Second-order condition**: If $f$ is twice differentiable, it is convex if and only if its Hessian $\nabla^2 f(x)$ is positive semidefinite for all $x$ in its domain.

### Examples

- Quadratic functions with PSD matrices.
- Norms: $\|x\|_p$ for $p \geq 1$.
- Exponential function.
- Negative logarithm on $(0, \infty)$.


## Convex Optimisation Problems

A convex optimisation problem has the form:
$$
\begin{aligned}
& \min_x \quad & f_0(x) \\
& \text{s.t.} \quad & f_i(x) \leq 0, \quad i=1, \dots, m \\
& & h_j(x) = 0, \quad j=1, \dots, p,
\end{aligned}
$$
where $f_0$ and $f_i$ are convex functions, and $h_j$ are affine.

The feasible set is convex, and any local minimum is a global minimum.


## Duality and Optimality

### Lagrangian and Dual Problem

The Lagrangian is
$$
L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x),
$$
where $\lambda \geq 0$ are dual variables. The dual problem is to maximise the infimum of $L(x, \lambda, \nu)$ over $x$.

### Karush-Kuhn-Tucker (KKT) Conditions

For convex problems, the KKT conditions are necessary and sufficient for optimality under mild constraint qualifications:

1. Primal feasibility: $f_i(x^*) \leq 0$, $h_j(x^*) = 0$.
2. Dual feasibility: $\lambda_i^* \geq 0$.
3. Complementary slackness: $\lambda_i^* f_i(x^*) = 0$.
4. Stationarity: $\nabla f_0(x^*) + \sum_i \lambda_i^* \nabla f_i(x^*) + \sum_j \nu_j^* \nabla h_j(x^*) = 0$.



## Algorithms for Convex Optimisation
Convex optimization algorithms exploit the geometry of convex sets and functions.  
Because every local minimum is global, algorithms can **converge reliably** without worrying about bad local minima.

We divide algorithms into three main families:

1. **First-order methods** – use gradients (scalable, but slower convergence).  
2. **Second-order methods** – use Hessians (faster convergence, more expensive).  
3. **Interior-point methods** – general-purpose, highly accurate solvers.


###  Gradient Descent (First-Order Method)

#### Algorithm
For step size $\alpha > 0$:
$$
x^{k+1} = x^k - \alpha \nabla f(x^k)
$$

#### Convergence
- If $f$ is convex and $\nabla f$ is Lipschitz continuous with constant $L$:
  - With fixed step $\alpha \le \tfrac{1}{L}$, we have:
    $$
    f(x^k) - f(x^*) = \mathcal{O}\left(\tfrac{1}{k}\right)
    $$
- If $f$ is **$\mu$-strongly convex**:
    $$
    f(x^k) - f(x^*) \le \left(1 - \tfrac{\mu}{L}\right)^k \big(f(x^0) - f(x^*)\big)
    $$
    (linear convergence).

#### Pros & Cons
- ✅ Simple, scalable to very high dimensions.  
- ❌ Slow convergence compared to higher-order methods.


### Accelerated Gradient Methods

- **Nesterov’s Accelerated Gradient (NAG):** Improves convergence rate from $\mathcal{O}(1/k)$ to $\mathcal{O}(1/k^2)$.  
- Widely used in **machine learning** (e.g., training deep neural networks).  

### Newton’s Method (Second-Order Method)

#### Algorithm
Update rule:
$$
x^{k+1} = x^k - [\nabla^2 f(x^k)]^{-1} \nabla f(x^k)
$$

#### Convergence
- Quadratic near optimum:  
  $$
  \|x^{k+1} - x^*\| \approx \mathcal{O}(\|x^k - x^*\|^2)
  $$
- Very fast, but requires Hessian and solving linear systems.

#### Damped Newton
To maintain global convergence:
$$
x^{k+1} = x^k - \alpha_k [\nabla^2 f(x^k)]^{-1} \nabla f(x^k), \quad \alpha_k \in (0,1]
$$


### Quasi-Newton Methods

Approximate the Hessian to reduce cost.

- **BFGS** and **L-BFGS** (limited memory version).  
- Used in large-scale optimization (e.g., machine learning, statistics).  
- Convergence: superlinear.  


### Subgradient Methods

For **nondifferentiable convex functions** (e.g., $f(x) = \|x\|_1$).

Update rule:
$$
x^{k+1} = x^k - \alpha_k g^k, \quad g^k \in \partial f(x^k)
$$

- $\partial f(x)$: subdifferential (set of all subgradients).  
- Convergence: $\mathcal{O}(1/\sqrt{k})$ with diminishing step sizes.  
- Useful in **large-scale, nonsmooth optimization**.

### Proximal Methods

For composite problems:  
$$
\min_x f(x) + g(x)
$$
where $f$ is smooth convex, $g$ convex but possibly nonsmooth.

**Proximal operator:**
$$
\text{prox}_g(v) = \arg\min_x \left( g(x) + \tfrac{1}{2}\|x-v\|_2^2 \right)
$$

- **Proximal gradient descent:**  
  $$
  x^{k+1} = \text{prox}_{\alpha g}(x^k - \alpha \nabla f(x^k))
  $$
- Widely used in sparse optimization (e.g., Lasso).


###  Interior-Point Methods
Transform constrained problem into a sequence of unconstrained problems using **barrier functions**.

For constraint $g_i(x) \le 0$, replace with barrier:
$$
\phi(x) = -\sum_{i=1}^m \log(-g_i(x))
$$

Solve:
$$
\min_x \; f(x) + \tfrac{1}{t} \phi(x)
$$
for increasing $t$.

#### Properties
- Polynomial-time complexity for convex problems.  
- Extremely accurate solutions.  
- Basis of general-purpose solvers (e.g., CVX, MOSEK, Gurobi).  


### Coordinate Descent

At each iteration, optimize w.r.t. one coordinate (or block of coordinates):

$$
x_i^{k+1} = \arg\min_{z} f(x_1^k, \dots, x_{i-1}^k, z, x_{i+1}^k, \dots, x_n^k)
$$

- Works well for high-dimensional problems.  
- Used in Lasso, logistic regression, and large-scale ML problems.


### Primal-Dual and Splitting Methods

- **ADMM (Alternating Direction Method of Multipliers):**  
  Splits problem into subproblems, solves in parallel.  
  Popular in distributed optimization and ML.  

- **Primal-dual interior-point methods:**  
  Solve both primal and dual simultaneously.  

---

### Summary of Convergence Rates

| Method                  | Smooth Convex | Strongly Convex |
|-------------------------|---------------|-----------------|
| Gradient Descent        | $\mathcal{O}(1/k)$ | Linear |
| Accelerated Gradient    | $\mathcal{O}(1/k^2)$ | Linear |
| Subgradient             | $\mathcal{O}(1/\sqrt{k})$ | – |
| Newton’s Method         | Quadratic (local) | Quadratic |
| Interior-Point          | Polynomial-time | Polynomial-time |

---

### Choosing an Algorithm

- **Small problems, high accuracy:** Newton, Interior-point.  
- **Large-scale smooth problems:** Gradient descent, Nesterov acceleration, L-BFGS.  
- **Large-scale nonsmooth problems:** Subgradient, Proximal, ADMM.  
- **Sparse / structured constraints:** Coordinate descent, Proximal methods.  

## Log-concavity and Log-convexity

A function $f: \mathbb{R}^n \to \mathbb{R}_{++}$ is:

- **Log-concave** if $\log f(x)$ is concave.
- **Log-convex** if $\log f(x)$ is convex.

### Relevance

- Log-concave functions appear in probability (many common distributions have log-concave densities, such as Gaussian, exponential, and uniform). This ensures tractability of maximum likelihood estimation.
- Log-convexity is useful in geometric programming, where monomials and posynomials are log-convex.

### Examples

- Gaussian density is log-concave.
- Exponential function is log-convex.


## Geometric Programming

Geometric programming (GP) is a class of problems of the form:
$$
\begin{aligned}
& \min_x \quad & f_0(x) \\
& \text{s.t.} \quad & f_i(x) \leq 1, \quad i=1, \dots, m,
\end{aligned}
$$
where each $f_i$ is a posynomial.

- **Monomial**: $f(x) = c x_1^{a_1} x_2^{a_2} \dots x_n^{a_n}$, with $c > 0$, exponents real.
- **Posynomial**: Sum of monomials.

By applying the log transformation $y_i = \log x_i$, the problem becomes convex.

---

## Applications in AI and ML

- **Deep learning**: Convex surrogates such as hinge loss and logistic loss. Convex relaxation of non-convex objectives. Initialisation and analysis of non-convex landscapes.
- **Probabilistic modelling**: Maximum likelihood estimation often leads to convex optimisation problems when distributions are log-concave.
- **Optimisation engineering**: Resource allocation, scheduling, control design often rely on convex formulations.
- **Regularisation**: Convex penalties such as $\ell_1$ (lasso) and nuclear norm.

---

## References

- S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.
- D. Bertsekas, *Nonlinear Programming*, Athena Scientific, 1999.
- A. Ben-Tal and A. Nemirovski, *Lectures on Modern Convex Optimization*, SIAM, 2001.
- Relevant research articles on log-concavity, geometric programming, and proximal methods.

---
