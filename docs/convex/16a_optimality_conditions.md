# Chapter 7: First-Order and Geometric Optimality Conditions


Optimization problems seek points where no infinitesimal movement can improve the objective. For convex functions, first-order conditions provide precise geometric and analytic criteria for such points to be optimal. They generalize the idea of “zero gradient” to nonsmooth and constrained settings, linking gradients, subgradients, and the geometry of feasible regions.

These conditions form the conceptual bridge between unconstrained minimization and the Karush–Kuhn–Tucker (KKT) theory developed in the next chapter.


### 7.1  Understanding “nth-Order” Optimality Conditions

For a differentiable function $f : \mathbb{R}^n \to \mathbb{R}$, the “order’’ of an optimality condition refers to how many derivatives (or generalized derivatives) we inspect around a candidate minimizer $x^*$:

| Order | Uses | Meaning |
|-------|------|----------|
| First-order | $\nabla f(x^*)$ (or subgradients) | Checks whether any local descent direction exists |
| Second-order | Hessian $\nabla^2 f(x^*)$ | Examines curvature to ensure the point is bowl-shaped (no local maxima or saddle) |
| Third and higher | Higher derivatives | Rarely used; detect flat or degenerate cases when curvature vanishes |

In general optimization, these successive tests ensure a point is truly a *local* minimizer. However, in convex optimization, this hierarchy collapses beautifully:


### Why Only First-Order Conditions Matter for Convex Functions

A convex function already has non-negative curvature everywhere — its Hessian is automatically positive semidefinite wherever it exists:

$$
\nabla^2 f(x) \succeq 0, \quad \forall x.
$$

Therefore, once the first-order condition is satisfied, no direction can decrease $f$ — not locally, but globally.  Convexity guarantees that the landscape is bowl-shaped everywhere, so a stationary point is the unique global minimum. In contrast, nonconvex functions can have zero gradient points that are maxima or saddles; second- or higher-order checks are needed to tell them apart. Hence, convex optimization requires only the first-order condition — it captures both necessity and sufficiency for global optimality. This remarkable simplification is one of the reasons convex analysis is so powerful and elegant.



## 7.2 Motivation

Consider minimizing a convex function $f$ over a convex set $\mathcal{X}$:

$$
\min_{x \in \mathcal{X}} f(x).
$$

Even for simple convex objectives, we need a way to check when a point $\hat{x}$ is optimal. In unconstrained problems, this means no direction can reduce $f$.  
With constraints, we only consider feasible directions — those that stay inside $\mathcal{X}$.

In both cases, optimality can be understood as an *equilibrium condition*:  
the gradient (or subgradient) of $f$ is balanced by the “forces’’ from the constraints. These equilibrium conditions are the first-order optimality conditions.

In machine learning, this reasoning appears everywhere — verifying when a trained model reaches stationarity (zero gradient) or when a sparsity constraint (like $\ell_1$ regularization) is active.

 
## 7.3 Unconstrained Convex Problems

For the unconstrained problem

$$
\min_x f(x),
$$

the first-order optimality condition is simple:

If $f$ is differentiable, $\hat{x}$ is optimal if and only if

$$
\nabla f(\hat{x}) = 0.
$$

If $f$ is convex but possibly nonsmooth, the gradient is replaced by the subdifferential:

$$
0 \in \partial f(\hat{x}).
$$

Intuitively, this means that the origin lies inside the set of all subgradients at $\hat{x}$.  Geometrically, every supporting hyperplane to the graph of $f$ at $(\hat{x}, f(\hat{x}))$ has zero slope — the function cannot be decreased by moving in any direction.

For smooth functions, $\nabla f(\hat{x}) = 0$ means that the tangent plane is horizontal. For nonsmooth functions, $0 \in \partial f(\hat{x})$ means there exists at least one horizontal supporting plane among all possible tangents.


## 7.4 Constrained Convex Problems

Now consider minimizing $f(x)$ over a closed convex set $\mathcal{X} \subseteq \mathbb{R}^n$:

$$
\min_{x \in \mathcal{X}} f(x).
$$

If $\hat{x} \in \operatorname{int}(\mathcal{X})$ (the interior), the situation is identical to the unconstrained case:
no boundary prevents motion, so

$$
0 \in \partial f(\hat{x}).
$$

However, if $\hat{x}$ lies on the boundary of $\mathcal{X}$, we must restrict movement to feasible directions — those that stay within the set.  
At such points, not all directions are allowed, and the gradient (or subgradient) may point outward from the feasible region.

### Tangent and Normal Cones

At a point $x \in \mathcal{X}$, define the tangent cone $T_x(\mathcal{X})$ as the set of feasible directions:

$$
T_x(\mathcal{X}) = \{\, d : \exists\, t_k \downarrow 0,\; x + t_k d \in \mathcal{X} \,\}.
$$

It captures all directions in which one can move infinitesimally without leaving $\mathcal{X}$.  

The normal cone is its polar:

$$
N_x(\mathcal{X}) = \{\, v : v^\top d \le 0,\; \forall d \in T_x(\mathcal{X}) \,\}.
$$

The normal cone consists of vectors pointing *outward* from $\mathcal{X}$ — the directions orthogonal (or opposing) to every feasible direction.

Geometrically, at an optimal boundary point, the gradient of $f$ must lie inside the normal cone:  
if you try to move within $\mathcal{X}$ (inside the tangent cone), $f$ cannot decrease further.


### First-Order Condition with Constraints

For constrained convex optimization, the unified first-order condition is

$$
0 \in \partial f(\hat{x}) + N_{\mathcal{X}}(\hat{x}).
$$

This means that there exists a subgradient $g \in \partial f(\hat{x})$ and a normal vector $v \in N_{\mathcal{X}}(\hat{x})$ such that $g + v = 0$.  
Equivalently, the objective’s slope is exactly counterbalanced by the constraint’s normal pressure.

If $\hat{x}$ lies in the interior of $\mathcal{X}$, $N_{\mathcal{X}}(\hat{x}) = \{0\}$, so the condition reduces to the unconstrained one ($0 \in \partial f(\hat{x})$).  
If $\hat{x}$ is on the boundary, the constraint pushes back against the descent direction.

### Geometric Interpretation

At optimality:

- The gradient (or a subgradient) points into the normal cone of the feasible set.  
- The tangent cone defines all directions along which the function cannot decrease.  
- The inclusion $0 \in \partial f(\hat{x}) + N_{\mathcal{X}}(\hat{x})$ encodes equilibrium between descent forces and boundary constraints.

This picture generalizes the intuitive idea from single-variable calculus:  
the derivative changes sign at a minimum, while with constraints, the derivative at the boundary is balanced by the constraint’s barrier.

