# Chapter 9: Lagrange Duality Theory

Duality is one of the most beautiful and useful ideas in convex optimisation. Every constrained optimisation problem (the primal) has an associated dual problem. The dual problem:

- provides a lower bound on the optimal primal value,
- often has structure that is easier to analyse,
- gives certificates of optimality,
- interprets multipliers as “prices” of constraints.

In convex optimisation, under mild assumptions, the primal and dual optimal values are equal.

 
## 9.1 The primal problem

We consider the general problem:
$$
\begin{array}{ll}
\text{minimise} & f(x) \\
\text{subject to} & g_i(x) \le 0,\quad i=1,\dots,m, \\
& h_j(x) = 0,\quad j=1,\dots,p.
\end{array}
$$

Assume $f$ and the $g_i$ are convex, and $h_j$ are affine. This is a convex optimisation problem.

We call $f^\star$ the optimal value:
$$
f^\star = \inf \{ f(x) : g_i(x) \le 0,\ h_j(x) = 0 \}.
$$

Here, *infimum* means the smallest value $f(x)$ can approach — even if it is not exactly attained.


## 9.2 Why Duality?

Before any equations, let’s understand the high-level idea.

A constrained optimization problem can be viewed as a *trade-off*:
> Minimize $f(x)$ while paying penalties for violating constraints.

If we allow violations but penalize them proportionally, we get a relaxed problem.  
How high should these penalties be? That’s what the dual variables $\mu_i, \lambda_j$ represent.

- $\mu_i$ (for inequalities): how costly it is to violate constraint $g_i(x) \le 0$.  
- $\lambda_j$ (for equalities): how much the objective changes when equality constraint $h_j(x)=0$ is relaxed.

Thus, duality converts *constraints* into *prices* or *forces* that shape the optimization landscape.


## 9.2 The Lagrangian

The Lagrangian function incorporates both the objective and constraints:
$$
L(x, \lambda, \mu)
= f(x)
+ \sum_{i=1}^m \mu_i g_i(x)
+ \sum_{j=1}^p \lambda_j h_j(x),
$$
with dual variables $\mu \in \mathbb{R}^m$, $\lambda \in \mathbb{R}^p$.

- $\mu_i \ge 0$ are multipliers for inequality constraints,
- $\lambda_j$ are free (can be any sign) for equality constraints.

If $\mu_i > 0$, violating the $i$th constraint is penalized heavily; if $\mu_i = 0$, that constraint is inactive.




## 9.4 The Dual Function – Lower Bounds from Penalties

For fixed $(\lambda, \mu)$, we define the dual function:
$$
\theta(\lambda, \mu) = \inf_x L(x, \lambda, \mu).
$$

Intuitively:

- We pick penalties $(\lambda, \mu)$ for constraint violations.
- We minimize $L$ with respect to $x$ — allowing constraint violations but paying for them.
- The resulting value $\theta(\lambda, \mu)$ is a *lower bound* on the original problem’s optimum $f^*$.

Formally, for any feasible $x$ and any $\mu \ge 0$,
$$
L(x, \lambda, \mu)
= f(x) + \sum_i \mu_i g_i(x) + \sum_j \lambda_j h_j(x)
\le f(x),
$$
since $g_i(x) \le 0$ and $\mu_i \ge 0$.  
Taking the infimum over all $x$ gives:
$$
\theta(\lambda, \mu) \le f^*.
$$

This property is known as weak duality:
> The dual function provides lower bounds on the primal optimum.
 
  
### Properties of the Dual Function

- $\theta(\lambda, \mu)$ is concave, even if $f$ itself is not convex.  
  (Infimum of affine functions in $(\lambda, \mu)$ is concave.)
- It may take the value $-\infty$ if the Lagrangian is unbounded below.

The dual function defines a new optimization problem — the dual problem — where we maximize this lower bound.

## 9.5 The Dual Problem

We now *maximize* the dual function subject to $\mu \ge 0$:
$$
\begin{array}{ll}
\text{maximize}_{\lambda, \mu} & \theta(\lambda, \mu) \\
\text{subject to} & \mu \ge 0.
\end{array}
$$

This is the Lagrange dual problem.

Let $d^*$ denote the optimal dual value.  
From weak duality, we always have:
$$
d^* \le f^*.
$$

The dual problem is always a concave maximization — equivalently, a convex optimization problem in the variables $(\lambda, \mu)$.

## 9.6 Strong Duality and the Zero Duality Gap

When $d^* = f^*$, we say strong duality holds.  
The difference $f^* - d^*$ is called the duality gap.

- Weak duality: $d^* \le f^*$ (always true).  
- Strong duality: $d^* = f^*$ (no gap).

Strong duality means the dual gives *exactly the same value* as the primal — and optimal multipliers $(\lambda^*, \mu^*)$ exist.

For convex problems, this beautiful property holds under Slater’s condition (see Chapter 8):

> If there exists a strictly feasible point $\tilde{x}$ such that  
> $g_i(\tilde{x}) < 0$ for all $i$, and $h_j(\tilde{x}) = 0$ for all $j$,  
> then strong duality holds — the duality gap is zero.

Consequences:

- $f^* = d^*$ (zero gap).  
- Dual variables $(\lambda^*, \mu^*)$ exist and are finite.  
- The KKT conditions are both necessary and sufficient for optimality.

 
## 9.7 Duality and the KKT Conditions

The KKT conditions (from Chapter 8) are the points where:

1. The primal is feasible ($g_i(x^*) \le 0$, $h_j(x^*) = 0$),
2. The dual is feasible ($\mu_i^* \ge 0$),
3. The gradients balance (stationarity):
   $$
   \nabla f(x^*) + \sum_i \mu_i^* \nabla g_i(x^*) + \sum_j \lambda_j^* \nabla h_j(x^*) = 0,
   $$
4. Complementary slackness holds ($\mu_i^* g_i(x^*) = 0$).

When these hold, $(x^*, \lambda^*, \mu^*)$ is a primal–dual optimal pair and  
$$
f(x^*) = \theta(\lambda^*, \mu^*) = f^* = d^*.
$$

Geometrically, the primal and dual surfaces *touch* at the optimum — the tangent plane defined by $(\lambda^*, \mu^*)$ supports $f$ exactly.

 
## 9.8 Interpreting Dual Variables

Dual variables have rich interpretations:

- $\mu_i^*$: the *shadow price* of relaxing inequality $g_i(x) \le 0$.  
  A large $\mu_i^*$ means this constraint is expensive — small relaxation significantly reduces $f$.
- $\lambda_j^*$: the price or force associated with equality $h_j(x)=0$.  
  Changing the equality’s right-hand side shifts the objective by roughly $\lambda_j^*$.

In economic or resource allocation problems:
- The dual problem represents *pricing* of limited resources.  
- The primal problem represents *allocation* given prices.

In machine learning:
- SVMs: dual variables correspond to support vectors.  
- Lasso and Elastic Net: $\ell_1$ penalties can be viewed as dual constraints on coefficient magnitudes.  
- Regularized losses: duality expresses the trade-off between data fit and model complexity.

