# Convex Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if its domain $\mathrm{dom}(f) \subseteq \mathbb{R}^n$ is convex and, for all $x_1, x_2 \in \mathrm{dom}(f)$ and all $\theta \in [0,1]$:

$$
f(\theta x_1 + (1-\theta)x_2) \le \theta f(x_1) + (1-\theta) f(x_2)
$$

- **Convex domain:** Any line segment between two points in $\mathrm{dom}(f)$ stays entirely inside the domain.  
- **Graph intuition:** The graph of $f$ lies below the straight line connecting any two points on it — it’s “bowl-shaped” or at least flat in all directions.  

**ML intuition:** Convex functions ensure that **local minima are global minima**, which is crucial for reliable model training.


## First-Order Condition

For a convex function $f$ on a convex domain:

### Differentiable Case

If $f$ is differentiable at $x$, the **gradient** $\nabla f(x)$ satisfies, for all $y \in \mathrm{dom}(f)$:

$$
f(y) \ge f(x) + \nabla f(x)^T (y - x)
$$

- Geometric meaning: The tangent hyperplane at $x$ lies below the graph everywhere.  
- ML intuition: Gradient points in the direction of **steepest ascent**, and its negative is a descent direction.

### Non-Differentiable Case (Subgradients)

If $f$ is convex but not differentiable at $x$, a **subgradient** $g \in \mathbb{R}^n$ satisfies:

$$
f(y) \ge f(x) + g^T (y - x), \quad \forall y \in \mathrm{dom}(f)
$$

- The set of all such $g$ is called the **subdifferential**:

$$
\partial f(x) = \{ g \in \mathbb{R}^n \mid f(y) \ge f(x) + g^T (y - x), \ \forall y \in \mathrm{dom}(f) \}
$$

- Geometric meaning: Even at a “kink,” there exists a hyperplane (with slope $g$) that supports the graph from below.  
- If $f$ is differentiable, $\partial f(x) = \{\nabla f(x)\}$.

**ML intuition:** Subgradients let us optimize **nonsmooth functions** like $L_1$ regularization or hinge loss.


## Second-Order (Hessian) Condition

If $f$ is twice differentiable, $f$ is convex if and only if:

$$
\nabla^2 f(x) \succeq 0, \quad \forall x \in \mathrm{dom}(f)
$$

- Positive semidefinite Hessian means the function curves upward or is flat in all directions — **never downward**.  
- Intuition: “Curvature is nonnegative in all directions.”

 
### Examples of Convex Functions

1. **Quadratic functions:** $f(x) = \frac{1}{2} x^T Q x + b^T x + c$, $Q \succeq 0$.  
2. **Norms:** $\|x\|_p$, $p \ge 1$.  
3. **Exponential:** $f(x) = e^x$.  
4. **Negative logarithm:** $f(x) = -\log(x)$ for $x>0$.  
5. **Linear functions:** $f(x) = a^T x + b$.


## Subgradients & Nonsmooth Functions

Many ML problems involve **nonsmooth convex functions**:

- Absolute value: $f(x) = |x|$  
- Hinge loss: $f(x) = \max(0, 1-x)$  
- $L_1$ norm: $f(x) = \|x\|_1 = \sum_i |x_i|$  

**Definition:** $g \in \mathbb{R}^n$ is a **subgradient** of $f$ at $x$ if:

$$
f(y) \ge f(x) + g^T (y-x), \quad \forall y \in \mathbb{R}^n
$$

- The set of all subgradients is $\partial f(x)$.  
- Geometric meaning: Subgradients define **supporting hyperplanes** at kinks.

 
### Example 1: Absolute Value

$f(x) = |x|$  

- $x > 0$: $\nabla f(x) = 1$  
- $x < 0$: $\nabla f(x) = -1$  
- $x = 0$: derivative doesn’t exist, but

$$
\partial f(0) = \{ g \in [-1,1] \}
$$

**Intuition:** At $x=0$, there’s a **fan of supporting lines**.

 
### Example 2: Hinge Loss

$f(x) = \max(0,1-x)$  

- $x < 1$: $\nabla f(x) = -1$  
- $x > 1$: $\nabla f(x) = 0$  
- $x = 1$:  

$$
\partial f(1) = [-1,0]
$$

**ML relevance:** SVM optimization uses these subgradients.


## Convex Optimization Problems

A general convex optimization problem:

$$
\begin{aligned}
& \min_x \quad & f_0(x) \\
& \text{s.t.} \quad & f_i(x) \le 0, \quad i=1,\dots,m \\
& & h_j(x) = 0, \quad j=1,\dots,p
\end{aligned}
$$

- $f_0, f_i$: convex  
- $h_j$: affine  
- Feasible set is convex, so **any local minimum is global**

**ML relevance:** This framework covers:

- Linear and quadratic programs  
- SVM, Lasso, Ridge regression  
- Constrained deep learning layers
