# Chapter 13: Optimization Algorithms for Equality-Constrained Problems

Equality-constrained optimization arises whenever variables must satisfy exact relationships, such as conservation laws, normalization, or linear invariants. In this chapter we focus on problems of the form

$$
\min_x \; f(x) \quad \text{s.t.} \quad A x = b.
$$

where $f : \mathbb{R}^n \to \mathbb{R}$ is (typically convex and differentiable) and $A \in \mathbb{R}^{p \times n}$ has rank $p$. This linear equality structure appears in constrained least squares, portfolio optimization, and many ML formulations that impose exact balance or normalization constraints.


 
## Geometric View — Optimization on an Affine Manifold

The constraint $A x = b$ defines an affine set

$$
\mathcal{X} = \{ x \in \mathbb{R}^n \mid A x = b \}.
$$


If $\operatorname{rank}(A) = p$, then $\mathcal{X}$ is an $(n-p)$-dimensional affine subspace of $\mathbb{R}^n$: a “flat” lower-dimensional plane embedded in the ambient space. Optimization now happens *along this plane*, not in all of $\mathbb{R}^n$. Any feasible direction $d$ must keep us in $\mathcal{X}$, so it must satisfy

$$
A (x + d) = b \quad \Rightarrow \quad A d = 0.
$$

Thus, feasible directions lie in the null space of $A$:

$$
\mathcal{D}_{\text{feas}} = \{ d \in \mathbb{R}^n \mid A d = 0 \} = \operatorname{Null}(A).
$$

At an optimal point $x^\star \in \mathcal{X}$, moving in any feasible direction $d$ cannot decrease $f$. For differentiable $f$, this means

$$
\nabla f(x^\star)^\top d \ge 0 \quad \text{for all } d \text{ with } A d = 0.
$$

Equivalently, $\nabla f(x^\star)$ must be orthogonal to all feasible directions, i.e. it lies in the row space of $A$. Therefore there exists a vector of Lagrange multipliers $\nu^\star$ such that

$$
\nabla f(x^\star) = A^\top \nu^\star.
$$

This is the basic geometric optimality condition: at the optimum, the gradient of $f$ is a linear combination of the constraint normals (rows of $A$), and every feasible direction is orthogonal to $\nabla f(x^\star)$.



## Lagrange Function and KKT System

The Lagrangian for the equality-constrained problem is

$$
\mathcal{L}(x,\nu)
=
f(x) + \nu^\top (A x - b),
$$

where $\nu \in \mathbb{R}^p$ are Lagrange multipliers. The first-order (KKT) conditions for a point $(x^\star,\nu^\star)$ to be optimal are

$$
\begin{aligned}
\nabla_x \mathcal{L}(x^\star,\nu^\star) &= \nabla f(x^\star) + A^\top \nu^\star = 0 
\quad &\text{(stationarity)},\\
A x^\star &= b 
\quad &\text{(primal feasibility)}.
\end{aligned}
$$

When $f$ is convex and $A$ has full row rank, these conditions are necessary and sufficient for global optimality. For Newton-type methods we linearize these conditions around a current iterate $(x,\nu)$ and solve for corrections $(\Delta x,\Delta \nu)$ from

$$
\begin{bmatrix}
\nabla^2 f(x) & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\ \Delta \nu
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(x) + A^\top \nu \\
A x - b
\end{bmatrix}.
$$

This linear system is called the (equality-constrained) KKT system. At the optimum the right-hand side is zero.


## Quadratic Objectives

A particularly important case is a convex quadratic objective

$$
f(x) = \tfrac{1}{2} x^\top P x + q^\top x + r,
$$

with $P \succeq 0$. The equality-constrained problem

$$
\min_x \tfrac{1}{2} x^\top P x + q^\top x + r 
\quad \text{s.t.} \quad A x = b
$$

has KKT conditions

$$
\begin{bmatrix}
P & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
x^\star \\ \nu^\star
\end{bmatrix}
=
-
\begin{bmatrix}
q \\ -b
\end{bmatrix}.
$$

If $P \succ 0$ and $A$ has full row rank, this system has a unique solution $(x^\star,\nu^\star)$. This is the standard linear system solved in equality-constrained least squares and quadratic programming.

Examples in ML and statistics:

- constrained least squares with sum-to-one constraints on coefficients;  
- portfolio optimization with $ \mathbf{1}^\top w = 1$;  
- quadratic surrogate subproblems inside second-order methods.

The structure of the KKT matrix (symmetric, indefinite, with blocks $P$, $A$) can be exploited by specialized linear solvers and factorizations.

 
## Null-Space (Reduced Variable) Method

When the constraints are linear and of full row rank, a natural approach is to eliminate them explicitly.

Choose:

- a particular feasible point $x_0$ satisfying $A x_0 = b$,  
- a matrix $Z \in \mathbb{R}^{n \times (n-p)}$ whose columns form a basis of the null space of $A$:
  $$
  A Z = 0.
  $$

Then every feasible $x$ can be written as

$$
x = x_0 + Z y, \quad y \in \mathbb{R}^{n-p}.
$$

Substituting into the objective yields an unconstrained reduced problem in the smaller variable $y$:

$$
\min_{y} \; \phi(y) := f(x_0 + Z y).
$$

Gradients and Hessians transform as

$$
\nabla_y \phi(y) = Z^\top \nabla_x f(x_0 + Z y), \qquad
\nabla_y^2 \phi(y) = Z^\top \nabla_x^2 f(x_0 + Z y) \, Z.
$$

We can now apply any unconstrained method (gradient descent, CG, Newton) to $\phi(y)$. The corresponding updates in the original space are mapped back via $x = x_0 + Z y$.

Key points:

- Optimization is restricted to feasible directions $\operatorname{Null}(A)$ by construction.  
- The dimension drops from $n$ to $n-p$, which can be advantageous if $p$ is large.  
- The cost is computing and storing a suitable null-space basis $Z$, which may destroy sparsity and be expensive for large-scale problems.

Null-space methods are attractive when:

- the number of constraints is moderate,  
- a good factorization of $A$ is available,  
- and we want an unconstrained algorithm in reduced coordinates.

 
## Newton’s Method for Equality-Constrained Problems

For a twice-differentiable convex $f$, we can derive an equality-constrained Newton step by solving a local quadratic approximation subject to linearized constraints.

At a point $x$, approximate $f(x+d)$ by its second-order Taylor expansion:

$$
f(x+d) \approx f(x)
+ \nabla f(x)^\top d
+ \tfrac{1}{2} d^\top \nabla^2 f(x) d.
$$

We seek a step $d$ that approximately minimizes this quadratic model while remaining feasible to first order, i.e.

$$
\begin{aligned}
\min_d & \quad \tfrac{1}{2} d^\top \nabla^2 f(x) d + \nabla f(x)^\top d\\
\text{s.t.} & \quad A d = 0.
\end{aligned}
$$

The KKT conditions for this quadratic subproblem are

$$
\begin{bmatrix}
\nabla^2 f(x) & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
d \\ \lambda
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(x) \\ 0
\end{bmatrix}.
$$

Solving this system gives the Newton step $d_{\text{nt}}$ and a multiplier update $\lambda$. The primal update is

$$
x_{k+1} = x_k + \alpha_k d_{\text{nt}},
$$

with a step size $\alpha_k \in (0,1]$ chosen by line search to ensure sufficient decrease and preservation of feasibility (for equality constraints, $A d_{\text{nt}} = 0$ guarantees $A x_{k+1} = b$ whenever $A x_k = b$).

Geometrically:

- unconstrained Newton would move by $-\nabla^2 f(x)^{-1} \nabla f(x)$;  
- equality-constrained Newton projects this step onto the tangent space $\{ d : A d = 0 \}$ of the affine constraint set.

For strictly convex $f$ with positive definite Hessian on the feasible directions, this method enjoys quadratic convergence near the solution, much like the unconstrained Newton method.

 

## Connections to Machine Learning and Signal Processing

Linear equality constraints appear naturally in ML and related areas:

| Setting | Equality constraint | Interpretation |
|--------|---------------------|----------------|
| Portfolio optimization | $\mathbf{1}^\top w = 1$ | Weights sum to one (full investment) |
| Constrained regression | $C x = d$ | Enforce domain-specific linear relations between coefficients |
| Mixture models / convex combinations | $\mathbf{1}^\top \alpha = 1, \; \alpha \ge 0$ | Mixture weights form a probability simplex |
| Fairness constraints (linearized) | $A w = 0$ | Enforce equal averages across groups or balance conditions |
| Physics-informed models (discretized) | $A x = b$ | Discrete conservation laws (mass, charge, energy) |

More generally, nonlinear equality constraints (e.g. $W^\top W = I$ for orthonormal embeddings, or $\|w\|_2^2 = 1$ for normalized weights) lead to optimization on curved manifolds. Techniques from this chapter extend to those settings when combined with Riemannian optimization or local parameterizations, but here we focus on the linear case as the fundamental building block.






