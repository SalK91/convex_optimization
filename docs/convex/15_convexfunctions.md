# Chapter 5: Convex Functions

Convex functions are the objectives we minimise. Understanding them is essential, because convexity of the objective is what turns an optimisation problem from "possibly impossible" to "provably solvable".


## 5.1 Definitions of convexity
A function $f : \mathbb{R}^n \to \mathbb{R}$ is convex if for all $x,y$ in its domain and all $\theta \in [0,1]$,

$$
f(\theta x + (1-\theta) y)
\le
\theta f(x) + (1-\theta) f(y)~.
$$

If the inequality is strict whenever $x \ne y$ and $\theta \in (0,1)$, then $f$ is strictly convex.

Geometrically, the chord between two points on the graph of f lies above the graph itself. This means that f never “bends downward” — it has a single valley rather than multiple dips.  

Equivalently, the **epigraph** of f,

$$
\mathrm{epi}(f) = \{ (x, t) \in \mathbb{R}^n \times \mathbb{R} : f(x) \le t \}.
$$

is a convex set.

## 5.2 First-order characterisation

If $f$ is differentiable, then $f$ is convex if and only if

$$
f(y) \ge f(x) + \nabla f(x)^\top (y - x)
\quad \text{for all } x,y.
$$

Interpretation:

- The first-order Taylor approximation is always a global underestimator.
- The gradient at $x$ defines a supporting hyperplane to the epigraph of $f$ at $(x, f(x))$.

This inequality is sometimes called the first-order condition for convexity.


> This is a powerful characterization: it says the tangent hyperplane at any point $x$ lies below the graph of $f$ everywhere. In other words, the gradient at $x$ provides a global underestimator of $f$ (supporting hyperplane to epigraph). Geometrically, this means no tangent line ever goes above the function. 

> For a convex differentiable $f$, we have $f(y) - f(x) \ge \nabla f(x)^T (y-x)$, so moving from $x$ in any direction, the actual increase in $f$ is at least as large as the linear prediction by $\nabla f(x)$ (since the function bends upward or straight). At optimum $\hat{x}$, a necessary and sufficient condition (for convex differentiable $f$) is $\nabla f(\hat{x}) = 0$. This ties to optimality: $\nabla f(\hat{x})=0$ means $f(y)\ge f(\hat{x}) + \nabla f(\hat{x})^T (y-\hat{x}) = f(\hat{x})$ for all $y$, so $\hat{x}$ is global minimizer.

> If $f$ is not differentiable, a similar condition holds with subgradients (see next chapter): $f$ is convex iff for all $x,y$ there exists a subgradient $g \in \partial f(x)$ such that $f(y) \ge f(x) + g^T(y-x)$. The set of all subgradients $\partial f(x)$ is a convex set (the subdifferential). At optimum, $0 \in \partial f(\hat{x})$ is the condition. 

## 5.3 Second-order characterisation

If $f$ is twice continuously differentiable, then $f$ is convex if and only if its Hessian is positive semidefinite everywhere:

$$
\nabla^2 f(x) \succeq 0
\quad \text{for all } x~.
$$

If $\nabla^2 f(x) \succ 0$ for all $x$, then $f$ is strictly convex.

 

## 5.4 Examples of convex functions

1. Affine functions:  
   $f(x) = a^\top x + b$.  
   Always convex (and concave).

2. Quadratic functions with PSD Hessian:  
   $f(x) = \tfrac12 x^\top Q x + c^\top x + d$,  
   where $Q \succeq 0$ (symmetric positive semidefinite).  
   Convex because $\nabla^2 f(x) = Q \succeq 0$.

3. Norms:  
   $f(x) = \|x\|$ for any norm.  
   All norms are convex.

4. Maximum of affine functions:  
   $f(x) = \max_i (a_i^\top x + b_i)$.  
   Convex because it is the pointwise maximum of convex functions.

5. Log-sum-exp function:  
   $f(x) = \log \left( \sum_{i=1}^k \exp(a_i^\top x + b_i) \right)$.  
   This function is convex and is ubiquitous in statistics and machine learning (softmax, logistic regression). The convexity follows from Jensen’s inequality and properties of the exponential (Boyd and Vandenberghe, 2004).

## 5.5 Jensen’s inequality

Let $f$ be convex, and let $X$ be a random variable taking values in the domain of $f$. Then
$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]~.
$$

> the function value at the mean is always less than or equal to the mean of function values.

This is Jensen’s inequality. It generalises the definition of convexity from two-point averages to arbitrary expectations. As a special case, for scalars $x_1,\dots,x_n$ and weights $\theta_i \ge 0$ with $\sum_i \theta_i = 1$,
$$
f\!\left(\sum_i \theta_i x_i\right)
\le
\sum_i \theta_i f(x_i).
$$


> Jensen’s inequality has many uses: in machine learning, it justifies algorithms like EM (which use the inequality to create surrogate objectives), and it provides bounds like $\log(\mathbb{E}[e^X]) \ge \mathbb{E}[X]$ (by convexity of $\log$ or $e^x$). As a simple example, taking $f(x)=x^2$ and $X$ uniform in ${-1,1}$, Jensen says $(\mathbb{E}[X])^2 = 0^2 \le \mathbb{E}[X^2] = 1$, which is true. Or $f(x)=\frac{1}{x}$ convex on $(0,\infty)$ implies $\frac{1}{\mathbb{E}[X]} \le \mathbb{E}[\frac{1}{X}]$ for positive $X$. In optimization, Jensen’s inequality often helps in proving convexity of expectations: if you mix some distributions or uncertain inputs, the expected loss is convex if the loss function is convex.

## 5.6 Operations that preserve convexity

If $f$ and $g$ are convex, then:

- $f + g$ is convex.
- $\alpha f$ is convex for any $\alpha \ge 0$.
- $\max\{f,g\}$ is convex.
- Composition with an affine map preserves convexity:  
  If $A$ is a matrix and $b$ a vector, then $x \mapsto f(Ax + b)$ is convex.

If $f$ is convex and nondecreasing in each argument, and each $g_i$ is convex, then the composition $x \mapsto f(g_1(x), \dots, g_k(x))$ is convex. This helps you build new convex functions from known ones.

## 5.7 Level sets of convex functions

For $\alpha \in \mathbb{R}$, define the sublevel set
$$
\{ x : f(x) \le \alpha \}.
$$

> Geometrically, these are the “contour slices” of a convex bowl — cross-sections below certain heights.  


If $f$ is convex, then every sublevel set is convex. This is crucial: constraints of the form $f(x) \le \alpha$ are convex constraints.

For example, the set
$$
\{ x : \|Ax - b\|_2 \le \epsilon \}
$$
is convex because $x \mapsto \|Ax-b\|_2$ is convex.

## 5.8 Strict and strong convexity

- $f$ is strictly convex if
$$
f(\theta x + (1-\theta) y) < \theta f(x) + (1-\theta) f(y)
$$
for all $x \ne y$ and $\theta \in (0,1)$.

> Strict convexity ensures uniqueness of the minimizer: there can be only one bottom to the bowl.


- $f$ is strongly convex with parameter $m>0$ if
$$
f(y) \ge f(x) + \nabla f(x)^\top (y-x) + \frac{m}{2} \|y-x\|_2^2.
$$

Strong convexity implies a unique minimiser and gives quantitative convergence rates for gradient methods.

