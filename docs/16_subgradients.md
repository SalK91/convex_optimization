# Chapter 6: Nonsmooth Convex Optimization – Subgradients

Many of the most important convex functions are not differentiable everywhere:

- $\|x\|_1 = \sum_i |x_i|$ has corners at $x_i = 0$,
- $f(x) = \max\{a_1^\top x + b_1, \dots, a_k^\top x + b_k\}$ is piecewise affine,
- the hinge loss $\max\{0, 1 - y w^\top x\}$ (used in SVMs) is not smooth at the kink.

 
For a convex but nonsmooth $f$, the usual condition “$\nabla f(x^*) = 0$” may not make sense, because $\nabla f(x^*)$ may not exist. But geometrically, convex functions still have supporting hyperplanes at every point. That is the key.


## 6.1 Subgradients and the subdifferential

Let $f : \mathbb{R}^n \to \mathbb{R}$ be convex. A vector $g \in \mathbb{R}^n$ is called a subgradient of $f$ at $x$ if, for all $y$,
$$
f(y) \ge f(x) + g^\top (y - x).
$$

Interpretation:

- The affine function $y \mapsto f(x) + g^\top (y-x)$ is a global underestimator of $f$.
- $g$ defines a supporting hyperplane to the epigraph of $f$ at $(x,f(x))$.

The set of all subgradients of $f$ at $x$ is called the subdifferential of $f$ at $x$:
$$
\partial f(x) = \{ g : f(y) \ge f(x) + g^\top (y-x) \ \forall y \}.
$$

If $f$ is differentiable at $x$, then
$$
\partial f(x) = \{ \nabla f(x) \}.
$$

If $f$ is not differentiable at $x$, $\partial f(x)$ is typically a nonempty convex set.



## 6.2 Examples

###  Absolute value in 1D
Let $f(t) = |t|$.

- For $t>0$, $\partial f(t) = \{1\}$.
- For $t<0$, $\partial f(t) = \{-1\}$.
- For $t=0$, 
  $$
  \partial f(0) = [-1, 1].
  $$

At the kink, any slope between $-1$ and $1$ is a valid supporting line from below.

###  $\ell_1$ norm
For $f(x) = \|x\|_1 = \sum_i |x_i|$, we have
$$
\partial \|x\|_1 = \{ g \in \mathbb{R}^n : g_i \in \partial |x_i| \}.
$$
So

- if $x_i > 0$, then $g_i = 1$,
- if $x_i < 0$, then $g_i = -1$,
- if $x_i = 0$, then $g_i \in [-1,1]$.

This is exactly what shows up in LASSO optimality conditions in statistics.

###  Pointwise max of affine functions
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



## 6.3 Subgradient optimality condition

Suppose we want to solve the unconstrained convex minimisation problem

$$
\min_x f(x),
$$

Then a point $x^*$ is optimal if and only if

$$
0 \in \partial f(x^*).
$$

> At the minimiser, there is no subgradient pointing into a direction that would reduce $f$.



## 6.4 Subgradient calculus (useful rules)

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
 

## 6.5 Applications in Machine Learning

- Nonsmooth Optimization:  
  Subgradient descent and proximal algorithms solve problems with nonsmooth losses (hinge, absolute error, $\ell_1$ penalties).  
    - Lasso ($\ell_1$ regularization): $\min \|Xw - y\|_2^2 + \lambda \|w\|_1$
    - SVM (hinge loss): $\min \frac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i w^T x_i)$

- Regularization and Sparsity:  
  Subgradients characterize the behavior of norms at the origin – leading to sparsity in optimal solutions (especially $\ell_1$ norms).

- Duality and Proximal Methods:  
  Many primal-dual algorithms rely on subgradients for deriving dual objectives and implementing updates via proximal operators.

- Composite Optimization:  
  Problems of the form $f(x) + g(x)$, where $f$ is smooth and $g$ is nonsmooth (e.g. Lasso), rely on subgradients of $g$.
 
 