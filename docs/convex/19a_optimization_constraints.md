# Chapter 13: Optimization Algorithms for Equality-Constrained Problems

Equality-constrained optimization arises whenever the variables must satisfy one or more exact relations — such as conservation laws, normalization, or fairness criteria. We study algorithms for minimizing a function subject to linear or nonlinear equality constraints:

$$
\min_x \; f(x) \quad \text{s.t.} \quad A x = b.
$$

Such problems are fundamental in convex optimization, quadratic programming, and many ML formulations involving exact invariants.

 
## 13.1 Geometric View — Optimization on an Affine Manifold

The constraint $A x = b$ defines an affine set, a lower-dimensional plane within $\mathbb{R}^n$. The feasible region is:

$$
\mathcal{X} = \{ x \in \mathbb{R}^n \mid A x = b \}.
$$

If $A \in \mathbb{R}^{p \times n}$ has full row rank ($\operatorname{rank}(A)=p$), then $\mathcal{X}$ is an $(n-p)$-dimensional affine manifold.

Geometrically, optimization proceeds not over all $\mathbb{R}^n$, but along this manifold. At the optimum, the gradient $\nabla f(x^\star)$ cannot point in a direction that stays feasible—hence it must be orthogonal to the feasible surface. This gives the first key optimality relation:

$$
\nabla f(x^\star) = A^\top \nu^\star,
$$

where $\nu^\star$ is a vector of Lagrange multipliers capturing how sensitive the objective is to constraint perturbations.

> Intuition:  
> The gradient of the objective at the optimum lies in the span of the constraint normals (rows of $A$).  
> Any feasible direction must lie in the null space of $A$, orthogonal to $\nabla f(x^\star)$.

 

## 13.2 Lagrange Function and KKT System

Define the Lagrangian:

$$
\mathcal{L}(x, \nu) = f(x) + \nu^\top (A x - b).
$$

The first-order (KKT) conditions for a feasible point $(x^\star, \nu^\star)$ to be optimal are:

$$
\begin{aligned}
\nabla f(x^\star) + A^\top \nu^\star &= 0, \\
A x^\star &= b.
\end{aligned}
$$

These equations express stationarity and feasibility simultaneously. They can be combined into the KKT linear system:

$$
\begin{bmatrix}
\nabla^2 f(x) & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta \nu
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(x) + A^\top \nu \\
A x - b
\end{bmatrix}.
$$

At the optimum, the right-hand side is zero.

> ML Connection:  
> Lagrange multipliers $\nu$ quantify trade-offs between objectives and hard constraints —  
> for instance, enforcing weight normalization in a neural layer, balance constraints in fair classification, or conservation laws in physics-informed networks.

 
## 13.3 The Quadratic Case

For a quadratic objective
$$
f(x) = \tfrac{1}{2}x^\top P x + q^\top x + r,
$$
with $P \succeq 0$, the KKT conditions reduce to a linear system:

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

This is a saddle-point system, solvable by factorization or elimination. If $P \succ 0$ and $A$ has full row rank, the solution $(x^\star, \nu^\star)$ is unique.

> In ML, such systems appear in constrained least squares, e.g. enforcing $\sum_i w_i = 1$ in portfolio optimization or convex combination weights in mixture models.

 
## 13.4 The Null-Space (Reduced Variable) Method

If $A$ has full row rank, we can find a particular feasible point $x_0$ such that $A x_0 = b$, and a basis $Z$ for the null space of $A$ satisfying $A Z = 0$.  
Then any feasible $x$ can be written as:

$$
x = x_0 + Z y, \quad y \in \mathbb{R}^{n-p}.
$$

Substituting into the objective gives a reduced problem:

$$
\min_y \; f(x_0 + Z y).
$$

This is an unconstrained problem in $y$, solvable by gradient or Newton methods.  
The reduced gradient and Hessian are:

$$
\nabla_y f = Z^\top \nabla_x f, \qquad \nabla_y^2 f = Z^\top \nabla_x^2 f \, Z.
$$

> Interpretation: Optimization proceeds only along feasible directions — those that do not violate the constraints (i.e., within $\operatorname{Null}(A)$).  
> This is equivalent to projecting all gradient steps onto the tangent space of the constraint manifold.

 

## 13.5 Newton’s Method for Equality-Constrained Problems

For a twice differentiable $f$, the equality-constrained Newton step solves the quadratic subproblem:

$$
\begin{aligned}
\min_d & \quad \tfrac{1}{2} d^\top \nabla^2 f(x) d + \nabla f(x)^\top d, \\
\text{s.t.} & \quad A d = 0.
\end{aligned}
$$

This produces the step $(d, \lambda)$ from the linearized KKT system:

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

The update is $x_{k+1} = x_k + \alpha d$, ensuring $A x_{k+1} = b$ if $A x_k = b$.

Geometric insight:  
The Newton direction is the projection of the unconstrained Newton step onto the tangent space of the feasible set (directions satisfying $A d = 0$).  
Thus, each step stays within the affine constraint manifold.

> In practice:  
> The KKT system is typically solved by *Schur complement factorization*:
> $$
> (A (\nabla^2 f)^{-1} A^\top) \lambda = A (\nabla^2 f)^{-1} \nabla f,
> $$
> which then yields $d = -(\nabla^2 f)^{-1} (\nabla f + A^\top \lambda)$.

 
## 13.6 Infeasible Start Newton Method

When starting from an infeasible point ($A x_0 \ne b$),  
we relax the constraint and drive feasibility progressively.  
At iteration $k$, compute $(\Delta x, \Delta \nu)$ by solving:

$$
\begin{bmatrix}
\nabla^2 f(x_k) & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\ \Delta \nu
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(x_k) + A^\top \nu_k \\
A x_k - b
\end{bmatrix}.
$$

Then update:

$$
x_{k+1} = x_k + \alpha \Delta x, \quad
\nu_{k+1} = \nu_k + \alpha \Delta \nu.
$$

This method enforces feasibility gradually, converging to $(x^\star, \nu^\star)$ under mild conditions.

> In ML contexts, infeasible starts are typical — we rarely have feasible initialization (e.g., in constrained autoencoders or regularized fairness models).  
> The infeasible Newton method ensures consistent progress in both primal feasibility ($A x = b$) and dual stationarity ($\nabla f + A^\top \nu = 0$).

 

## 13.7 Computational Considerations

- Factorization: KKT systems can be large but structured. Exploiting sparsity in $\nabla^2 f$ and $A$ is essential in high-dimensional problems.
- Stability: Adding small regularization to the (0,0) block of the KKT matrix improves conditioning:
  $$
  \begin{bmatrix}
  \nabla^2 f + \delta I & A^\top \\
  A & -\delta I
  \end{bmatrix}.
  $$
- Schur Complement: Eliminating $\Delta x$ yields a smaller linear system in $\Delta \nu$, which can be more efficient when $p \ll n$.

 

## 13.8 Connections to Machine Learning

Equality-constrained optimization appears in several ML and signal processing settings:

| Example | Equality Constraint | Interpretation |
|----------|---------------------|----------------|
| Portfolio optimization | $\mathbf{1}^\top w = 1$ | Weights must sum to 1 |
| Fair classification | $A w = 0$ | Enforces equal outcomes across groups |
| Orthogonal embeddings | $W^\top W = I$ | Preserves independence / energy |
| Normalization layers | $\|w\|_2^2 = 1$ | Scale invariance constraint |
| Physics-informed models | $\text{div}(F)=0$ | Conservation of mass / charge |

### Summary: Approaches to Equality-Constrained Optimization

| Approach | Constraint Type | Feasibility (Local/Global) | Core Idea | Advantages | Limitations / Drawbacks | Typical ML / Optimization Use |
|---------------|--------------------|--------------------------------|----------------|----------------|-----------------------------|-----------------------------------|
| Null-Space (Variable Elimination) | Linear, full-rank $A$ | Global | Parameterize feasible $x = x_0 + Z y$ with $A Z = 0$ | Converts to unconstrained problem; dimension reduction; exact | Requires null-space basis $Z$; destroys sparsity; expensive for large $A$ | Constrained least squares, small-scale convex programs |
| Local Parameterization (Manifold Method) | Nonlinear $g(x) = 0$ | Local (around feasible point) | Use implicit function theorem: locally express $x = x(y)$ | Captures nonlinear manifold structure; geometric insight | Valid only locally; requires Jacobians; expensive | Manifold learning, orthogonal embeddings, equality-regularized networks |
| KKT / Lagrange System | Linear or nonlinear | Global (if convex) | Solve coupled system $\nabla f + A^\top \nu = 0$, $A x = b$ | Keeps structure; allows dual interpretation; works for large sparse systems | Larger system; more variables | Quadratic programming, convex solvers, equality-constrained ML models |
| Primal–Dual Newton Method | Linear or nonlinear | Global (convex) | Newton’s method on full KKT system | Quadratic convergence near optimum; stable numerically | Requires Hessians and factorizations | Interior-point solvers, primal–dual optimization, barrier methods |
| Penalty / Augmented Lagrangian | General (convex or nonconvex) | Approximate (drives feasibility) | Add penalty term $\tfrac{\rho}{2}\|A x - b\|^2$ or dual updates | Simple to implement; smooth transition from unconstrained | Needs tuning of $\rho$; slow convergence to exact feasibility | Regularized fairness, soft constraints, physics-informed networks |
| Projection / Normalization Step | Linear or nonlinear (simple form) | Iterative (after each step) | Project back to feasible set: $x_{k+1} = \Pi_{\{A x = b\}}(x_{k+1})$ | Keeps updates feasible; easy for simple constraints | Costly for complex $A$; may distort gradient direction | Normalization layers, unit-norm or balance constraints |
