# Chapter 9: Lagrange Duality Theory

Duality is one of the central organizing principles in convex optimization. Every constrained problem (the primal) has an associated dual problem, whose structure often provides:

- lower bounds on the primal optimal value,
- certificates of optimality,
- interpretations of constraint “prices,”
- and alternative algorithmic routes to solutions.

In convex optimization, duality is especially powerful: under mild conditions, the primal and dual attain the same optimal value. This equality — *strong duality* — lies behind the theory of KKT conditions, interior-point methods, and many ML algorithms such as SVMs.


## The Primal Problem

Consider the general convex problem

$$
\begin{array}{ll}
\text{minimize} & f(x) \\
\text{subject to} & g_i(x) \le 0,\quad i=1,\dots,m, \\
 & h_j(x) = 0,\quad j=1,\dots,p,
\end{array}
$$

where:

- $f$ and each $g_i$ are convex,
- each equality constraint $h_j$ is affine.

The optimal value is

$$
f^\star = \inf\{ f(x) : g_i(x) \le 0,\ h_j(x)=0 \}.
$$

The infimum allows for the possibility that the best value is approached but not attained.

## Why Duality?

A constrained problem can be viewed as:

> minimize $f(x)$ but pay a penalty whenever constraints are violated.

If the penalties are chosen “correctly,” one can recover the original constrained problem from an unconstrained penalized problem. Dual variables — $\mu_i$ for inequalities and $\lambda_j$ for equalities — precisely encode these penalties:

- $\mu_i$ measures how costly it is to violate $g_i(x)\le 0$,
- $\lambda_j$ measures the sensitivity of the objective to relaxing $h_j(x)=0$.

Duality converts constraints into prices, and transforms geometry into algebra.


## The Lagrangian

The Lagrangian function is

$$
L(x, \lambda, \mu)
= f(x) + \sum_{i=1}^m \mu_i g_i(x)
+ \sum_{j=1}^p \lambda_j h_j(x),
$$

with:

- $\mu_i \ge 0$ for inequality constraints,
- $\lambda_j \in \mathbb{R}$ unrestricted for equalities.

Interpretation:

- If $\mu_i > 0$, violating $g_i(x)\le 0$ incurs a penalty proportional to $\mu_i$.
- If $\mu_i = 0$, that constraint does not influence the Lagrangian at that point.

 
## The Dual Function: Lower Bounds from Penalties

Fix $(\lambda,\mu)$ and minimize the Lagrangian with respect to $x$:

$$
\theta(\lambda, \mu) = \inf_x L(x,\lambda,\mu).
$$

Because $g_i(x) \le 0$ for feasible $x$ and $\mu_i \ge 0$,

$$
L(x,\lambda,\mu) \le f(x),
$$

so taking the infimum over all $x$ yields

$$
\theta(\lambda,\mu) \le f^\star.
$$

Thus $\theta$ always produces lower bounds on the true optimal value (weak duality).

### Properties of the Dual Function

- $\theta(\lambda,\mu)$ is always concave in $(\lambda,\mu)$ (infimum of affine functions).
- It may be $-\infty$ if the Lagrangian is unbounded below.

## The Dual Problem

The dual problem maximizes these lower bounds:

$$
\begin{array}{ll}
\text{maximize}_{\lambda,\mu} & \theta(\lambda,\mu) \\
\text{subject to} & \mu \ge 0.
\end{array}
$$

Let $d^\star$ be the optimal dual value.  
Weak duality guarantees:

$$
d^\star \le f^\star.
$$

The dual problem is always a concave maximization, i.e., a convex optimization problem in $(\lambda,\mu)$.

## Strong Duality and the Duality Gap

If

$$
d^\star = f^\star,
$$

we say strong duality holds. The duality gap is zero.

### Slater’s Condition 

If:

- $g_i$ are convex,
- $h_j$ are affine,
- and there exists a $\tilde{x}$ such that  
  $$
  g_i(\tilde{x}) < 0,\quad h_j(\tilde{x}) = 0,
  $$

then:

- strong duality holds ($f^\star = d^\star$),
- dual maximizers exist,
- KKT conditions fully characterize primal–dual optimality.

Slater’s condition ensures the feasible region has interior — the constraints are not tight everywhere.


## Duality and the KKT Conditions

When strong duality holds, the primal and dual meet at a point satisfying the KKT conditions:

### Primal feasibility
$$
g_i(x^\star) \le 0,\qquad h_j(x^\star)=0.
$$

### Dual feasibility
$$
\mu_i^\star \ge 0.
$$

### Stationarity
$$
\nabla f(x^\star)
+ \sum_{i=1}^m \mu_i^\star \nabla g_i(x^\star)
+ \sum_{j=1}^p \lambda_j^\star \nabla h_j(x^\star)
= 0.
$$

### Complementary slackness
$$
\mu_i^\star g_i(x^\star) = 0,\qquad \forall i.
$$

Together these conditions ensure:

$$
f(x^\star) = \theta(\lambda^\star,\mu^\star)
= f^\star = d^\star.
$$

Geometrically, the gradients of the active constraints form a supporting hyperplane that “touches’’ the objective exactly at the optimum.


## Interpretation of Dual Variables

Dual variables have consistent interpretations across optimization, ML, and economics.

### Shadow Prices / Constraint Forces

- $\mu_i^\star$: the *shadow price* for relaxing $g_i(x)\le 0$.  
  Large $\mu_i^\star$ means the constraint is tight and costly to relax.

- $\lambda_j^\star$: sensitivity of the optimal value to perturbations of $h_j(x)=0$.

### ML Interpretations

- Support Vector Machines: dual variables select support vectors (only points with $\mu_i^\star > 0$ matter).
- L1-Regularization / Lasso: can be viewed through a dual constraint on parameter magnitudes.
- Regularized learning problems: the dual expresses the balance between data fit and model complexity.

Duality often reveals structure that is hidden in the primal, providing clearer geometric insight and sometimes simpler optimization paths.
 

 

