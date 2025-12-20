# Chapter 6: Nonsmooth Convex Optimization – Subgradients

Many important convex objectives in machine learning are not differentiable everywhere. Examples include:

- the $ \ell_1 $ norm $ \|x\|_1 = \sum_i |x_i| $ (nondifferentiable at zero),
- pointwise-max functions such as $ f(x) = \max_i (a_i^\top x + b_i) $,
- the hinge loss $ \max\{0,\, 1 - y w^\top x\} $ used in SVMs,
- regularisers like total variation or indicator functions of convex sets.

Although these functions have “kinks”, they remain convex—and convexity guarantees the existence of supporting hyperplanes at every point.

## Subgradients and the Subdifferential

Let $f : \mathbb{R}^n \to \mathbb{R}$ be convex.  A vector $g \in \mathbb{R}^n$ is a subgradient of $f$ at $x$ if

$$
f(y) \ge f(x) + g^\top (y - x) \quad \text{for all } y.
$$

Geometric interpretation:

- The affine function $y \mapsto f(x) + g^\top(y-x)$ is a global underestimator of $f$.
- Each subgradient defines a supporting hyperplane touching the epigraph of $f$ at $(x, f(x))$.
- At smooth points, this supporting hyperplane is unique (the tangent plane).
- At kinks, there may be infinitely many supporting hyperplanes.

The subdifferential of $f$ at $x$ is the set
$$
\partial f(x)
=
\{ g : f(y) \ge f(x) + g^\top(y-x) \ \forall y \}.
$$

Properties:

- $ \partial f(x) $ is always a nonempty convex set (if $x$ is in the interior of the domain).
- If $f$ is differentiable at $x$, then  
  $$
  \partial f(x) = \{\nabla f(x)\}.
  $$
- If $f$ is strictly convex, the subdifferential is a singleton except at boundary/kink points.

Thus, subgradients generalise gradients to nonsmooth convex functions, preserving the same geometric meaning.


## Examples

### Absolute value in 1D

Let $f(t) = |t|$.  
Then:

- If $t > 0$,  $\partial f(t) = \{1\}$.
- If $t < 0$,  $\partial f(t) = \{-1\}$.
- If $t = 0$,  
  $$
  \partial f(0) = [-1,\, 1].
  $$

At the kink, any slope between $-1$ and $1$ supports the graph from below.

### The $ \ell_1 $ norm

For $f(x) = \|x\|_1 = \sum_i |x_i|$:

$$
g \in \partial \|x\|_1
\quad\Longleftrightarrow\quad
g_i \in \partial |x_i|.
$$

Thus:

- if $x_i > 0$, then $g_i = 1$,
- if $x_i < 0$, then $g_i = -1$,
- if $x_i = 0$, then $g_i \in [-1,1]$.

This structure appears directly in LASSO and compressed sensing optimality conditions.

### Pointwise maximum of affine functions

Let  
$$
f(x) = \max_{i=1,\dots,k} (a_i^\top x + b_i).
$$

- If only one index $i^\star$ achieves the maximum at $x$, then  
  $$
  \partial f(x) = \{ a_{i^\star} \}.
  $$

- If multiple indices are tied, then  
  $$
  \partial f(x)
  = \mathrm{conv}\{ a_i : i \text{ active at } x \},
  $$
  the convex hull of the active slopes.

This structure underlies SVM hinge loss and ReLU-type functions.


## Subgradient Optimality Condition

For the unconstrained convex minimisation problem

$$
\min_x f(x),
$$

a point $x^\star$ is optimal if and only if

$$
0 \in \partial f(x^\star).
$$

Interpretation:

- At optimality, no subgradient points to a direction that would decrease $f$.
- Geometrically, the supporting hyperplane at $x^\star$ is horizontal, forming the flat bottom of the convex bowl.
- This generalises the smooth condition $ \nabla f(x^\star)=0 $.

## Subgradient Calculus

Subgradients satisfy powerful calculus rules that allow us to work with complex functions. Let $f$ and $g$ be convex.

### Sum rule
$$
\partial(f+g)(x) \subseteq \partial f(x) + \partial g(x)
=
\{ u+v : u \in \partial f(x),\ v \in \partial g(x) \}.
$$

Equality holds under mild regularity conditions (e.g., if both functions are closed).

### Affine composition
If $h(x) = f(Ax + b)$, then
$$
\partial h(x) = A^\top \partial f(Ax+b).
$$

This rule is heavily used in machine learning models, where losses depend on linear predictions $Ax$.

### Maximum of convex functions
If $f(x) = \max_i f_i(x)$, then
$$
\partial f(x)
= \mathrm{conv}\{ \partial f_i(x) : i \text{ active at } x \}.
$$

This supports models based on hinge losses, margin-maximisation, and piecewise-linear architectures.



  

   