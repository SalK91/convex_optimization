# Optimality Conditions for Convex Optimization Problems

Convex optimization problems have the important property that any local minimum is also a global minimum. Depending on whether the problem is unconstrained or constrained, and whether the function is differentiable or not, the optimality conditions differ.


## 1. Unconstrained Problems

### 1.1 Differentiable Convex Functions

For a differentiable convex function $f:\mathbb{R}^n \to \mathbb{R}$:

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

**Optimality condition:**

$$
\nabla f(\hat{x}) = 0
$$

Intuition: The gradient points in the direction of steepest increase, so a zero gradient indicates a flat spot (minimum).

Examples:

a) Quadratic function:
$$f(x) = x^2 - 4x + 7$$  
$\nabla f(x) = 2x - 4$ → set to 0 → $\hat{x} = 2$

b) Sum of Squared Errors
$$f(x) = \sum_{i=1}^{n} (x - y_i)^2$$  
$\nabla f(x) = 2\sum_{i=1}^{n} (x - y_i)$ → set to 0 → $\hat{x} = \frac{1}{n}\sum_{i=1}^{n} y_i$ (the **mean**)


### 1.2 Non-Differentiable Convex Functions

For convex but non-differentiable functions:

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

Optimality condition:

$$
0 \in \partial f(\hat{x})
$$

Intuition: The subgradient generalizes the derivative for functions with "kinks."

Examples:

a) Sum of Absolute Errors (SAE):
$$f(x) = \sum_{i=1}^{n} |x - y_i|$$  
$\hat{x}$ = **median** of the data points.

b) Maximum function:
$$f(x) = \max(x-1, 2-x)$$  
Subgradient: $\partial f(x) = \begin{cases} 1, & x > 1.5 \\ [-1,1], & x = 1.5 \\ -1, & x < 1.5 \end{cases}$ → minimum at $x = 1.5$



## 2. Constrained Problems

Consider a convex optimization problem:

$$
\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad x \in \mathcal{X}
$$

where $f$ is convex and $\mathcal{X} \subseteq \mathbb{R}^n$ is a convex feasible set.


### Interior Point:

If $\hat{x}$ lies strictly inside the feasible set, then the unconstrained condition applies:

$$
0 \in \partial f(\hat{x})
$$

Intuition: There are no boundary restrictions, so the gradient (or subgradient) must vanish.


### Boundary Point:

If $\hat{x}$ lies on the boundary of $\mathcal{X}$, then for $\hat{x}$ to be optimal:

- The **negative gradient** must lie in the **normal cone** of $\mathcal{X}$ at $\hat{x}$:

$$
- \nabla f(\hat{x}) \in N_{\mathcal{X}}(\hat{x})
$$

- Equivalently, the gradient must form an angle of at least $90^\circ$ with any feasible direction $d$ inside $\mathcal{X}$:

$$
\nabla f(\hat{x})^\top d \ge 0 \quad \forall d \in T_{\mathcal{X}}(\hat{x})
$$

where $T_{\mathcal{X}}(\hat{x})$ is the **tangent (feasible) cone** at $\hat{x}$, and $N_{\mathcal{X}}(\hat{x})$ is the **normal cone** at $\hat{x}$.

Intuition: At the boundary, the optimal direction cannot point into the feasible set because any movement along a feasible direction increases the objective.

### Compact Form

Combining interior and boundary cases:

$$
0 \in \partial f(\hat{x}) + N_{\mathcal{X}}(\hat{x})
$$

where:

- $N_{\mathcal{X}}(\hat{x}) = \{0\}$ for interior points  
- $N_{\mathcal{X}}(\hat{x})$ is the normal cone for boundary points  

This is a **general convex optimality condition for constrained problems**, valid for both differentiable and non-differentiable $f$.


### Intution
Imagine a region of allowed points, called the feasible set 
𝑋
X. Points strictly inside the region form the interior, where movement in any direction is possible without leaving the set. The edges and corners of the region form the boundary, where movement is restricted because you can only move along directions that remain feasible. Consider standing at a point 
𝑥
^
x
^
 on this boundary. From here, you cannot move freely in all directions; you can only move along directions that stay inside the feasible set. These allowable directions form what is called the tangent cone at 
𝑥
^
x
^
, encompassing movements along the boundary or slightly into the interior.

Opposing these feasible directions is the normal cone, which consists of vectors that point outward from the feasible region, effectively “blocking” any movement that would stay inside. At an optimal boundary point, the gradient of the objective function points outward, lying within the normal cone. This means that moving along any feasible direction — whether along the boundary or slightly into the interior — cannot decrease the objective function. The gradient “pushes against” all allowable moves, so any small displacement that respects the constraints either increases the objective or leaves it unchanged.

This behavior contrasts with an interior optimum, where the gradient is zero and movement in any direction does not change the objective. At a boundary optimum, the gradient is non-zero but oriented such that all feasible directions are blocked from reducing the objective. Even though the gradient is not zero, the point is still optimal because the boundary restricts movement: every allowed step either raises the objective or keeps it the same. In this way, a boundary point can be a true optimum, and the outward-pointing gradient is the formal expression of the intuitive idea that you cannot “go downhill” without leaving the feasible region.

### 2.3 Example: Quadratic with Constraint

$$
\min f(x) = x^2 \quad \text{s.t. } x \ge 1
$$

- Feasible set: $\mathcal{X} = [1, \infty)$  
- Gradient: $\nabla f(x) = 2x$  

**Check optimality:**

- Interior check ($x>1$): $2x = 0 \implies x = 0$ → infeasible  
- Boundary check ($x=1$): $-\nabla f(1) = -2 \in N_{\mathcal{X}}(1) = \mathbb{R}_+$ → satisfied  

**Solution:** $\hat{x} = 1$
