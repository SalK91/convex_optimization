# Chapter 8: Optimization Principles – From Gradient Descent to KKT

At this point we understand:

- how to recognise convex functions,
- how to talk about feasible sets,
- how to describe optimality with gradients or subgradients.

Now we turn to *constrained* optimisation. We first recall unconstrained optimisation and gradient descent, then develop the Karush–Kuhn–Tucker (KKT) conditions, which are the first-order optimality conditions for constrained convex optimisation (Boyd and Vandenberghe, 2004).

 
## 8.1 Unconstrained convex minimisation

Consider
$$
\min_x f(x),
$$
where $f$ is convex and differentiable.

Gradient descent is the iterative method:
$$
x^{(k+1)} = x^{(k)} - \alpha_k \nabla f(x^{(k)}),
$$
for some step size $\alpha_k > 0$.

Intuition:

- Move opposite the gradient to reduce $f$.
- Under suitable conditions on step size (e.g. Lipschitz gradient), this converges to a global minimiser if $f$ is convex.

If $f$ is strongly convex, we get uniqueness of the minimiser and faster convergence.

 

## 8.2 Equality-constrained optimisation and Lagrange multipliers

Consider
$$
\begin{array}{ll}
\text{minimise} & f(x) \\
\text{subject to} & h_j(x) = 0,\quad j=1,\dots,p,
\end{array}
$$
where $f$ and each $h_j$ are differentiable.

We define the Lagrangian
$$
L(x,\lambda)
=
f(x) + \sum_{j=1}^p \lambda_j h_j(x),
$$
where $\lambda_j$ are the Lagrange multipliers.

A necessary condition for $x^*$ to be optimal (under suitable regularity assumptions) is:

1. Stationarity:
   $$
   \nabla_x L(x^*, \lambda^*) = 0
   \quad \Longleftrightarrow \quad
   \nabla f(x^*) + \sum_j \lambda_j^* \nabla h_j(x^*) = 0.
   $$
2. Primal feasibility:
   $$
   h_j(x^*) = 0 \quad \text{for all } j.
   $$

Geometrically, stationarity says: the gradient of $f$ at $x^*$ lies in the span of the gradients of the active constraints. In words, you cannot move in any feasible direction without increasing $f$.

 

## 8.3 Inequality constraints and KKT

Now consider the general problem:
$$
\begin{array}{ll}
\text{minimise} & f(x) \\
\text{subject to} & g_i(x) \le 0,\quad i=1,\dots,m, \\
& h_j(x) = 0,\quad j=1,\dots,p.
\end{array}
$$

We form the Lagrangian
$$
L(x,\lambda,\mu)
=
f(x)
+ \sum_{j=1}^p \lambda_j h_j(x)
+ \sum_{i=1}^m \mu_i g_i(x),
$$
with multipliers $\lambda \in \mathbb{R}^p$ (unrestricted) and $\mu \in \mathbb{R}^m$ with $\mu_i \ge 0$.

The Karush–Kuhn–Tucker (KKT) conditions consist of:

1. Primal feasibility:
   $$
   g_i(x^*) \le 0,\quad i=1,\dots,m,
   \qquad
   h_j(x^*) = 0,\quad j=1,\dots,p.
   $$

2. Dual feasibility:
   $$
   \mu_i^* \ge 0,\quad i=1,\dots,m.
   $$

3. Stationarity:

      $\nabla f(x^*) 
      + \sum_{j=1}^p \lambda_j^* \nabla h_j(x^*)
      + \sum_{i=1}^m \mu_i^* \nabla g_i(x^*)
      = 0$

4. Complementary slackness:
   $$
   \mu_i^* g_i(x^*) = 0
   \quad \text{for all } i.
   $$

Complementary slackness means:

- If a constraint $g_i(x) \le 0$ is strictly inactive at $x^*$ (i.e. $g_i(x^*) < 0$), then $\mu_i^* = 0$.
- If $\mu_i^* > 0$, then the constraint is tight: $g_i(x^*) = 0$.

This matches geometric intuition: only active constraints can “push back” on the optimiser.

 
## 8.4 KKT and convexity

For general nonlinear problems, KKT conditions are *necessary* under regularity assumptions. For convex problems, KKT conditions are often necessary and sufficient for optimality. In other words, if the problem is convex and a point satisfies KKT, that point is globally optimal.

This is extremely powerful:

- You can certify optimality (and thus global optimality) just by finding multipliers $\lambda^*, \mu^*$ that satisfy KKT.
- KKT conditions are constructive: they are what solvers try to satisfy.

 
## 8.5 Geometric picture

At the optimal point $x^*$:

- $\nabla f(x^*)$ is balanced by a conic combination of the normals of the active inequality constraints plus a linear combination of the equality constraint normals.
- The objective cannot be decreased by moving in any feasible direction.

Visually: the contour of $f$ is “tangent” to the feasible region. The Lagrange multipliers encode the direction and strength of that tangency.


## 8.6 Constraint qualifications 

To guarantee that KKT multipliers exist and KKT conditions apply cleanly, we usually need a mild regularity condition called a constraint qualification. The most common is Slater’s condition for convex problems:

> If the problem is convex and there exists a strictly feasible point $\tilde{x}$ such that  
> $g_i(\tilde{x}) < 0$ for all $i$ and $h_j(\tilde{x}) = 0$ for all $j$,  
> then strong duality holds and KKT conditions are necessary and sufficient (Boyd and Vandenberghe, 2004).

