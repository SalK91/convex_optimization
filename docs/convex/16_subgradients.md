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
 

## 6.5 Subgradient Methods

Even though nonsmooth functions lack gradients, we can still minimize them using subgradient descent. Given a subgradient $g_k \in \partial f(x_k)$, the iteration

$$
x_{k+1} = x_k - \alpha_k g_k
$$

moves in the direction of the negative subgradient. Unlike in smooth optimization, the step sizes $\alpha_k$ typically decrease with $k$ (for example, $\alpha_k = c / \sqrt{k}$) to guarantee convergence. Subgradient descent converges to the global minimum for convex $f$, though at a slower rate than smooth gradient descent. While smooth convex functions enjoy $\mathcal{O}(1/k^2)$ or linear convergence under strong convexity, nonsmooth convex functions converge at rate $\mathcal{O}(1/\sqrt{k})$. In practice, many machine learning algorithms—such as SVM training with hinge loss, $\ell_1$-regularized models, and even certain deep learning optimizers—operate as subgradient methods in disguise. Their stability and robustness stem from convexity rather than smoothness.

