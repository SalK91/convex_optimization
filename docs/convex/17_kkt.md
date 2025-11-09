# Chapter 8: Optimization Principles – From Gradient Descent to KKT

At this point we understand:

- how to recognise convex functions,
- how to talk about feasible sets,
- how to describe optimality with gradients or subgradients.

This chapter unifies these ideas. We begin with unconstrained optimization and the gradient descent principle, then extend to equality and inequality constraints — culminating in the Karush–Kuhn–Tucker (KKT) conditions, the cornerstone of constrained convex optimization.

 
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

 > In machine learning, this is the foundation of training via backpropagation — each step reduces the loss by following the negative gradient of the cost function with respect to model parameters.


## 8.2 Equality-constrained optimisation and Lagrange multipliers

Now suppose we add equality constraints:

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

Geometrically, stationarity says: At an equality-constrained optimum, the gradient of $f$ is orthogonal to the feasible set — it points in a direction that cannot reduce $f$ without violating a constraint.
 

## 8.3 Inequality constraints and KKT

Now consider the general convex problem:
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
with dual varibales $\lambda \in \mathbb{R}^p$ (unrestricted) and $\mu \in \mathbb{R}^m$ with $\mu_i \ge 0$.

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

## 8.4 Slater’s Condition – Ensuring Strong Duality

For the KKT conditions to not only hold but also guarantee optimality and zero duality gap, the problem must satisfy a regularity condition known as Slater’s condition.

### Definition

For the convex problem above, if all $f$ and $g_i$ are convex and all $h_j$ are affine, then Slater’s condition holds if there exists at least one strictly feasible point:

$$
\exists\, x^{\text{slater}} \text{ such that } 
h_j(x^{\text{slater}}) = 0, \ \forall j, 
\quad \text{and} \quad 
g_i(x^{\text{slater}}) < 0, \ \forall i.
$$

This means there is some point that satisfies the equalities exactly and the inequalities strictly — i.e., a point inside the feasible region, not merely on its boundary.

### Dual Problems and the Duality Gap

Every constrained problem (the primal) has a dual:
$$
p^* = \min_x f(x), \qquad
d^* = \max_{\lambda,\mu\ge0} g(\lambda,\mu),
$$
where $g(\lambda,\mu)$ is the dual function obtained from the Lagrangian.

By weak duality, for all feasible $x,(\lambda,\mu)$,
$$
d^* \le p^*.
$$
The difference  
$$
\text{duality gap} = p^* - d^*
$$
measures how far apart the primal and dual optima are.

- If $p^*>d^*$, the gap is positive (weak duality only).  
- If $p^*=d^*$, we have strong duality — the primal and dual attain the same optimum.


### What Slater’s Condition Guarantees

For convex problems, Slater’s condition ensures:
1. Strong duality: $p^*=d^*$ (duality gap = 0).  
2. Dual attainment: finite $(\lambda^*,\mu^*)$ exist.  
3. KKT conditions are necessary and sufficient for optimality.

Intuitively, Slater’s condition says the feasible region has “breathing room’’; it’s not pinched to the boundary.  
Then the dual hyperplanes can exactly touch the primal surface — they *kiss* at the optimum, eliminating the gap.

### Examples

(a) Condition Holds → Zero Gap
$$
\min_x x^2 \quad \text{s.t. } x \ge 1.
$$
Strictly feasible point $x=2$ satisfies $g(x)=-1<0$.  
Hence $p^*=d^*=1$ — no duality gap.

(b) Condition Fails → Positive Gap
$$
\min_x x \quad \text{s.t. } x^2 \le 0.
$$
Feasible set $\{0\}$ has no interior; no strictly feasible point.  
Dual cannot attain equality: $p^*>d^*$.



## 8.5 Geometric and Physical Interpretation

- The gradient of the objective is balanced by the weighted gradients of the active constraints.  
- Each multiplier $\mu_i$ or $\lambda_j$ acts like a *tension* or *shadow price* that enforces feasibility.
- The KKT system generalizes the condition $\nabla f(x^*) = 0$:
  - In the unconstrained case, there are no forces — pure gradient equilibrium.
  - With constraints, these forces push the solution back into the feasible region.

> Physically, you can imagine optimization as minimizing potential energy subject to rigid walls (constraints).  
> At equilibrium, the total force — gradient of $f$ plus constraint reactions — equals zero.
> Convexity ensures the landscape is bowl-shaped.  
> Slater’s condition ensures the bowl has interior volume so that primal and dual solutions coincide.  
> Together they make the KKT framework both elegant and powerful — the foundation upon which Lagrange duality theory is built.

