A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if its domain $\mathrm{dom}(f) \subseteq \mathbb{R}^n$ is convex and, for all $x_1, x_2 \in \mathrm{dom}(f)$ and all $\theta \in [0,1]$:

$$
f(\theta x_1 + (1-\theta)x_2) \le \theta f(x_1) + (1-\theta) f(x_2)
$$

- **Convex domain:** For any $x_1, x_2 \in \mathrm{dom}(f)$, the line segment connecting them lies entirely in $\mathrm{dom}(f)$.  
- The graph of $f$ lies below or on the straight line connecting any two points on it (“bowl-shaped”).


# First-order condition:

For a convex function $f: \mathbb{R}^n \to \mathbb{R}$ defined on a convex domain $\mathrm{dom}(f)$:

## 1. Differentiable case
If $f$ is differentiable at $x$, the **gradient** $\nabla f(x)$ satisfies, for all $y \in \mathrm{dom}(f)$:

$$
f(y) \ge f(x) + \nabla f(x)^T (y - x)
$$

- Geometric meaning: The tangent hyperplane at $x$ lies below the graph of $f$ at all points.  
- Domain of gradient: $\nabla f(x)$ exists for all $x \in \mathrm{dom}(f)$.


## 2. Non-differentiable case (subgradients)
If $f$ is convex but not differentiable at $x$, a **subgradient** $g \in \mathbb{R}^n$ satisfies:

$$
f(y) \ge f(x) + g^T (y - x), \quad \forall y \in \mathrm{dom}(f)
$$

- The set of all such $g$ is called the **subdifferential** at $x$:  

$$
\partial f(x) = \{ g \in \mathbb{R}^n \mid f(y) \ge f(x) + g^T (y - x), \ \forall y \in \mathrm{dom}(f) \}
$$

- Geometric meaning: Even at a "kink," there exists a hyperplane (with slope $g$) that lies below the graph at all points.  
- If $f$ is differentiable at $x$, $\partial f(x) = \{\nabla f(x)\}$.



# Second-order (Hessian) condition: 
If $f$ is twice differentiable, $f$ is convex if and only if its Hessian matrix $\nabla^2 f(x)$ is positive semidefinite for all $x \in \mathrm{dom}(f)$:

$$
\nabla^2 f(x) \succeq 0
$$

Positive semidefinite Hessian means the function curves upward or is flat in all directions, never downward.  

``` “Curvature is nonnegative in all directions.”  ```
  

# Examples of Convex Functions

1. Quadratic functions: $f(x) = \frac{1}{2} x^T Q x + b^T x + c$, where $Q \succeq 0$ (positive semidefinite).  
2. Norms: $\|x\|_p$ for $p \ge 1$.  
3. Exponential function: $f(x) = e^x$.  
4. Negative logarithm: $f(x) = -\log(x)$ on $x > 0$.  
5. Linear functions: $f(x) = a^T x + b$.  

# Subgradients & Proximal Operators
Modern optimization in machine learning often deals with nonsmooth functions e.g., $L_1$ regularization, hinge loss in SVMs, indicator constraints. Gradients are not always defined at these nonsmooth points, so we need subgradients and proximal operators. For a differentiable convex function $f:\mathbb{R}^n \to \mathbb{R}$, the gradient $\nabla f(x)$ provides the slope for descent. But many convex functions are not differentiable everywhere:

- Absolute value: $f(x) = |x|$    (non-differentiable at $x=0$)  
- Hinge loss: $f(x) = \max(0, 1-x)$  
- $L_1$ norm: $f(x) = \|x\|_1 = \sum_i |x_i|$  

At kinks/corners, derivatives don’t exist.  

A vector $g \in \mathbb{R}^n$ is a subgradient of a convex function $f$ at point $x$ if:

$$f(y) \;\ge\; f(x) + g^\top (y-x), \quad \forall y \in \mathbb{R}^n$$

- Geometric meaning: $g$ defines a supporting hyperplane at $(x, f(x))$ that lies below the function everywhere.  
- The set of all subgradients at $x$ is called the subdifferential, written:
  $$
  \partial f(x) = \{ g \in \mathbb{R}^n \;|\; f(y) \ge f(x) + g^\top (y-x), \;\forall y \}.
  $$



## Example 1: Absolute value
Take $f(x) = |x|$.  

- If $x > 0$: $\nabla f(x) = 1$.  
- If $x < 0$: $\nabla f(x) = -1$.  
- If $x = 0$: derivative doesn’t exist. But  
  $$
  \partial f(0) = \{ g \in [-1, 1] \}.
  $$
Any slope between $-1$ and $1$ is a valid subgradient at the kink.
 Intuition: At $x=0$, instead of one tangent line, there’s a whole fan of supporting lines.



## Example 2: Hinge loss
$f(x) = \max(0, 1-x)$.  

- If $x < 1$: $\nabla f(x) = -1$.  
- If $x > 1$: $\nabla f(x) = 0$.  
- If $x = 1$:  
  $$
  \partial f(1) = [-1, 0].
  $$



## Why subgradients matter
- They generalize gradients to nonsmooth convex functions.  
- Subgradient descent update:
  $$
  x_{k+1} = x_k - \alpha_k g_k, \quad g_k \in \partial f(x_k).
  $$
- Convergence is guaranteed, though slower than gradient descent:
  - Smooth case: $O(1/k)$ rate  
  - Nonsmooth case: $O(1/\sqrt{k})$ rate  


## Proximal Operators
Nonsmooth penalties (like $L_1$ norm, indicator functions) appear frequently:  
- Lasso: $\min_x \tfrac{1}{2}\|Ax-b\|^2 + \lambda \|x\|_1$  (L1 norm is nonsmooth)
- SVM: hinge loss $\max(0, 1-y\langle w,x\rangle)$  
- Constraints: e.g., $x \in C$ for some convex set $C$

Plain gradient descent cannot directly handle the nonsmooth part.

The proximal operator of a function $g$ with step size $\alpha > 0$ is:

$$
\text{prox}_{\alpha g}(v) 
= \arg\min_x \Big( g(x) + \frac{1}{2\alpha}\|x-v\|^2 \Big).
$$

- Interpretation:  
  - Stay close to $v$ (the quadratic term)  
  - While reducing the penalty $g(x)$  

- Geometric meaning: A regularized projection of $v$ onto a region encouraged by $g$.  


## Example 1: $L_1$ norm (soft-thresholding)
Let $g(x) = \lambda \|x\|_1 = \lambda \sum_i |x_i|$.  
Then:

$$
\text{prox}_{\alpha g}(v)_i = 
\begin{cases}
v_i - \alpha\lambda, & v_i > \alpha \lambda \\
0, & |v_i| \le \alpha \lambda \\
v_i + \alpha\lambda, & v_i < -\alpha \lambda
\end{cases}
$$

This is the soft-thresholding operator:

- Shrinks small entries of $v$ to zero → sparsity.  
- Reduces magnitude of large entries but keeps their sign.  

👉 This is the key step in Lasso regression and compressed sensing.

 
## Example 2: Indicator function
Let $g(x) = I_C(x)$, where $I_C(x)=0$ if $x \in C$, and $\infty$ otherwise.  
Then:

$$
\text{prox}_{\alpha g}(v) = \Pi_C(v),
$$

the Euclidean projection of $v$ onto $C$.  

Example: if $C$ is the unit ball $\{x: \|x\|\le 1\}$, prox just normalizes $v$ if it’s outside.

 
## Example 3: Squared $\ell_2$ norm
If $g(x) = \frac{\lambda}{2}\|x\|^2$, then

$$
\text{prox}_{\alpha g}(v) = \frac{1}{1+\alpha\lambda} v.
$$

This is just a shrinkage toward the origin.

---

## Why proximal operators matter
They allow efficient algorithms for composite objectives:

$$
\min_x f(x) + g(x),
$$

where:
- $f$ is smooth (differentiable with Lipschitz gradient)  
- $g$ is convex but possibly nonsmooth  

Proximal gradient method (ISTA):
$$
x_{k+1} = \text{prox}_{\alpha g}\big(x_k - \alpha \nabla f(x_k)\big).
$$

This generalizes gradient descent by replacing the plain update with a proximal step that handles $g$.

- If $g=0$: reduces to gradient descent  
- If $f=0$: reduces to proximal operator (e.g. projection, shrinkage)  

 
## 3. Intuition Summary

- Subgradients:  
  - Generalized “slopes” for nonsmooth convex functions.  
  - At corners, we have a set of possible slopes (subdifferential).  
  - Enable subgradient descent with convergence guarantees.  

- Proximal operators:  
  - Generalized update steps for nonsmooth regularizers.  
  - Combine a gradient-like move with a “correction” that enforces structure (sparsity, constraints).  
  - Core of algorithms like ISTA, FISTA, ADMM.  

---

## 4. Big Picture in ML

- Subgradients: Let us train models with nonsmooth losses (SVM hinge loss, $L_1$).  
- Proximal operators: Let us efficiently solve regularized problems (Lasso, group sparsity, constrained optimization).  
- Intuition:  
  - Subgradient = "any slope that supports the function"  
  - Proximal = "soft move toward minimizing the nonsmooth part"  

---


### Subgradients & Proximal Operators
- Subgradient: $g$ is a subgradient if  
$$
f(y) \ge f(x)+g^\top(y-x), \quad \forall y
$$  
- Proximal operator:  
$$
\text{prox}_{\alpha g}(v) = \arg\min_x \Big(g(x) + \frac{1}{2\alpha}\|x-v\|^2\Big)
$$  
- Context: Needed for nonsmooth functions (e.g., L1-regularization, hinge loss).  
- ML relevance: SVM hinge loss, Lasso, sparse dictionary learning. Proximal methods handle shrinkage or projection efficiently.  
- Intuition:  
  - Subgradient: Like a tangent for a function that isn’t smooth—provides a direction to descend.  
  - Proximal operator: Think of it as a “soft step” toward minimizing a nonsmooth function, like gently nudging a point toward a feasible or sparse region.


## Convex Optimisation Problems

A convex optimisation problem has the form:

$$
\begin{aligned}
& \min_x \quad & f_0(x) \\
& \text{s.t.} \quad & f_i(x) \leq 0, \quad i=1, \dots, m \\
& & h_j(x) = 0, \quad j=1, \dots, p,
\end{aligned}
$$
where $f_0$ and $f_i$ are convex functions, and $h_j$ are affine. The feasible set is convex, and any local minimum is a global minimum.


