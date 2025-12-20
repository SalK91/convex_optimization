# Chapter 7: First-Order and Geometric Optimality Conditions

Optimization problems seek points where no infinitesimal movement can improve the objective. For convex functions, first-order conditions give precise geometric and analytic criteria for such points to be optimal. They extend the familiar “zero gradient” condition to nonsmooth and constrained settings, linking gradients, subgradients, and the geometry of feasible regions.

These conditions form the conceptual bridge between unconstrained minimization and the Karush–Kuhn–Tucker (KKT) framework developed in the next chapter.

 
## Orders of Optimality: Why First Order is Enough in Convex Optimization

For a differentiable function $f : \mathbb{R}^n \to \mathbb{R}$, the “order’’ of an optimality condition refers to how many derivatives (or generalized derivatives) we examine around a candidate minimizer $x^\star$:

| Order        | Object inspected         | Role                                          |
|-------------|--------------------------|-----------------------------------------------|
| First-order | $\nabla f(x^\star)$ or subgradients | Detects existence of a local descent direction |
| Second-order| Hessian $\nabla^2 f(x^\star)$       | Examines curvature (minimum vs saddle vs maximum) |
| Higher-order| Third derivative and beyond | Rarely used; only for degenerate cases with vanishing curvature |

In general nonconvex optimization, these conditions are used together: a point may have $\nabla f(x^\star) = 0$ but still be a saddle or a local maximum, so curvature (second order) must also be checked.

For convex functions, the situation is much simpler. A convex function already has non-negative curvature everywhere:

$$
\nabla^2 f(x) \succeq 0 \quad \text{whenever the Hessian exists}.
$$

Therefore:

- any stationary point (where the first-order condition holds) cannot be a local maximum or saddle,  
- if the function is proper and lower semicontinuous, first-order conditions are enough to guarantee global optimality.

As a result, in convex optimization we typically rely only on first-order conditions, possibly expressed in terms of subgradients and geometric objects (normal cones, tangent cones). This collapse of the hierarchy is one of the key simplifications that makes convex analysis powerful.

 
## Motivation

Consider the basic convex problem
$$
\min_{x \in \mathcal{X}} f(x),
$$
where $f$ is convex and $\mathcal{X}$ is a convex set.

Intuitively, a point $\hat{x}$ is optimal if there is no feasible direction in which we can move and strictly decrease $f$. In the unconstrained case, every direction is feasible. In the constrained case, only directions that stay inside $\mathcal{X}$ are allowed.

Thus, optimality can be seen as an equilibrium:

- the objective’s tendency to decrease (captured by its gradient or subgradient)  
- is exactly balanced by the geometric restrictions imposed by the feasible set.

In machine learning, this appears as:

- training a model until the gradient is (approximately) zero in unconstrained problems, or  
- training until the force from regularization/constraints balances the data fit term (e.g., in $\ell_1$-regularized models).

First-order optimality conditions formalize this equilibrium in both smooth and nonsmooth, constrained and unconstrained settings.

 
## Unconstrained Convex Problems

For the unconstrained problem
$$
\min_x f(x),
$$
with $f$ convex, the optimality conditions are especially simple.

### Smooth case

If $f$ is differentiable, then a point $\hat{x}$ is optimal if and only if
$$
\nabla f(\hat{x}) = 0.
$$

Convexity ensures that any point where the gradient vanishes is a global minimizer, not just a local one.

### Nonsmooth case

If $f$ is convex but not necessarily differentiable, the gradient is replaced by the subdifferential. The condition becomes
$$
0 \in \partial f(\hat{x}).
$$

Interpretation:

- The origin lies in the set of all subgradients at $\hat{x}$.  
- Geometrically, there exists a horizontal supporting hyperplane to the epigraph of $f$ at $(\hat{x}, f(\hat{x}))$.  
- No direction in $\mathbb{R}^n$ gives a first-order improvement in the objective.

For smooth $f$, this reduces to the usual condition $\nabla f(\hat{x}) = 0$.


## Constrained Convex Problems

Now consider the constrained problem
$$
\min_{x \in \mathcal{X}} f(x),
$$
where $f$ is convex and $\mathcal{X} \subseteq \mathbb{R}^n$ is a nonempty closed convex set.

If $\hat{x}$ lies strictly inside $\mathcal{X}$, then there is locally no distinction from the unconstrained case: all nearby directions are feasible. In that case,
$$
0 \in \partial f(\hat{x})
$$
remains the necessary and sufficient condition for optimality.

The interesting case is when $\hat{x}$ lies on the boundary of $\mathcal{X}$.

### First-order condition with constraints

The general first-order optimality condition for the constrained convex problem is:
$$
0 \in \partial f(\hat{x}) + N_{\mathcal{X}}(\hat{x}).
$$

That is, there exist

- a subgradient $g \in \partial f(\hat{x})$, and  
- a normal vector $v \in N_{\mathcal{X}}(\hat{x})$

such that
$$
g + v = 0.
$$

Interpretation:

- The objective’s slope $g$ is exactly balanced by a normal vector $v$ coming from the constraint set.  
- If we decompose space into feasible and infeasible directions, there is no feasible direction along which $f$ can decrease.  
- Geometrically, the epigraph of $f$ and the feasible set meet with aligned supporting hyperplanes at $\hat{x}$.

Special cases:

- If $\hat{x}$ is an interior point, then $N_{\mathcal{X}}(\hat{x}) = \{0\}$, so we recover the unconstrained condition $0 \in \partial f(\hat{x})$.  
- If $\mathcal{X}$ is an affine set, the normal cone is the orthogonal complement of its tangent subspace, and the condition aligns with equality-constrained optimality.





 
