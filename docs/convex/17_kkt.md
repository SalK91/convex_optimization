# Chapter 8: Optimization Principles – From Gradient Descent to KKT

At this point we understand:

- how to recognise convex functions,
- how to talk about feasible sets,
- how to describe optimality with gradients or subgradients.

Now we turn to *constrained* optimisation. We first recall unconstrained optimisation and gradient descent, then develop the Karush–Kuhn–Tucker (KKT) conditions, which are the first-order optimality conditions for constrained convex optimisation.

 
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
   Primal feasibility simply means that the point $x^*$ satisfies all the original constraints of the optimization problem.

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

   > Dual feasibility says the “penalty coefficients” for inequality constraints can only push you inward (not reward you for violating constraints).

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


