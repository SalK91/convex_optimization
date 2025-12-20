# Chapter 14: Optimization Algorithms for Inequality-Constrained Problems

In many applications, we must optimize an objective while respecting *inequality* constraints: nonnegativity of variables, margin constraints in SVMs, capacity or safety limits, physical bounds, fairness budgets, and more. Mathematically, the feasible region is now a convex set with a boundary, and the optimizer often lies on that boundary.

This chapter introduces algorithms for solving such problems, focusing on *logarithmic barrier* and *interior-point* methods. These are the workhorses behind modern general-purpose convex solvers (for LP, QP, SOCP, SDP) and provide a smooth way to enforce inequalities while still using Newton-type methods.
 
## Problem Setup

We consider the general convex problem with inequality and equality constraints
$$
\begin{aligned}
\text{minimize} \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \le 0,\quad i = 1,\dots,m,\\
& A x = b,
\end{aligned}
$$
where

- $f_0, f_1,\dots,f_m$ are convex, typically twice differentiable,
- $A \in \mathbb{R}^{p \times n}$ has full row rank,
- there exists a strictly feasible point $\bar{x}$ such that
  $f_i(\bar{x}) < 0$ for all $i$ and $A\bar{x} = b$ (Slater’s condition).

Under these assumptions:

- the problem is convex,
- strong duality holds (zero duality gap),
- and the KKT conditions characterize optimality.

### Examples

| Problem type        | $f_0(x)$                               | Constraints $f_i(x)\le0$       | ML / applications                       |
|---------------------|-----------------------------------------|--------------------------------|-----------------------------------------|
| Linear program (LP) | $c^\top x$                             | $a_i^\top x - b_i \le 0$       | resource allocation, feature selection  |
| Quadratic program   | $\tfrac12 x^\top P x + q^\top x$       | linear                          | SVMs, ridge with box constraints       |
| QCQP                | quadratic                              | quadratic                       | portfolio optimization, control         |
| Entropy models      | $\sum_i x_i \log x_i$                  | $F x - g \le 0$                | probability calibration, max-entropy    |
| Nonnegativity       | arbitrary convex                       | $-x_i \le 0$                   | sparse coding, nonnegative factorization|

Many machine-learning training problems can be written in this template by expressing regularization, margins, fairness, or safety conditions as convex inequalities.


## Indicator-Function View of Constraints

Conceptually, we can write inequality constraints using an indicator function. Define
$$
I_{-}(u) =
\begin{cases}
0, & u \le 0,\\[4pt]
+\infty, & u > 0.
\end{cases}
$$

Then the inequality-constrained problem is equivalent to
$$
\min_x \; f_0(x)
      + \sum_{i=1}^m I_{-}\big(f_i(x)\big)
\quad \text{subject to } A x = b.
$$

- If $x$ is feasible ($f_i(x) \le 0$ for all $i$), the indicators contribute $0$.
- If any constraint is violated ($f_i(x) > 0$), the objective becomes $+\infty$.

This formulation is clean but not numerically friendly:

- $I_{-}$ is discontinuous and nonsmooth.
- We cannot directly apply Newton-type methods.

The key idea of barrier methods is to *replace* the hard indicator with a smooth approximation that grows to $+\infty$ as we approach the boundary.


## Logarithmic Barrier Approximation

We approximate the indicator $I_{-}$ with a smooth barrier function
$$
\Phi(u) = -\log(-u), \quad u < 0.
$$

For each inequality $f_i(x) \le 0$, we introduce a barrier term $-\log(-f_i(x))$. For a given parameter $t > 0$, we solve the *barrier subproblem*
$$
\min_x \; f_0(x) + \frac{1}{t} \,\phi(x)
\quad \text{subject to } A x = b,
$$
where
$$
\phi(x) = \sum_{i=1}^m -\log(-f_i(x)), \qquad
\mathrm{dom}\,\phi = \{x : f_i(x) < 0 \ \forall i\}.
$$

Equivalently,
$$
\min_x \; t f_0(x) + \phi(x)
\quad \text{subject to } A x = b.
$$

Interpretation:

- The barrier term $\phi(x)$ is finite only for strictly feasible points ($f_i(x) < 0$).
- As $x$ approaches the boundary $f_i(x) \to 0^-$, the term $-\log(-f_i(x)) \to +\infty$.
- The parameter $t$ controls the trade-off:
  - small $t$ (large $1/t$) → strong barrier, solution stays deep inside the feasible set;
  - large $t$ → barrier is weaker, solutions can move closer to the boundary.

As $t \to \infty$, solutions of the barrier subproblem approach the solution of the original constrained problem.

 
## Derivatives of the Barrier

Let
$$
\phi(x) = -\sum_{i=1}^m \log(-f_i(x)).
$$
Then $\phi$ is convex and twice differentiable on its domain. Its gradient and Hessian are
$$
\nabla \phi(x) = \sum_{i=1}^m \frac{1}{-f_i(x)} \, \nabla f_i(x),
$$
$$
\nabla^2 \phi(x)
=
\sum_{i=1}^m \frac{1}{[f_i(x)]^2} \, \nabla f_i(x)\nabla f_i(x)^\top
+ \sum_{i=1}^m \frac{1}{-f_i(x)} \, \nabla^2 f_i(x).
$$

Key features:

- As $f_i(x) \uparrow 0$ (approaching the boundary from inside), the factor $1/(-f_i(x))$ blows up, so $\|\nabla \phi(x)\|$ becomes very large.
- This creates a strong *repulsive force* that prevents iterates from crossing the boundary.
- The barrier “pushes” the solution away from constraint violation, while the original objective $f_0(x)$ pulls toward lower cost.

The barrier subproblem
$$
\min_x \; t f_0(x) + \phi(x)
\quad \text{s.t. } A x = b
$$
is a *smooth equality-constrained* problem. We can therefore apply equality-constrained Newton methods (Chapter 13) at each fixed $t$.


## Central Path and Approximate KKT Conditions

For each $t > 0$, let $x^\star(t)$ be a minimizer of the barrier problem
$$
\min_x \; t f_0(x) + \phi(x)
\quad \text{s.t. } A x = b.
$$

The set $\{x^\star(t) : t > 0\}$ is called the *central path*. As $t \to \infty$, $x^\star(t)$ converges to a solution $x^\star$ of the original inequality-constrained problem.

We can associate approximate dual variables to $x^\star(t)$:
$$
\lambda_i^\star(t) = \frac{1}{t\,(-f_i(x^\star(t)))}, \quad i=1,\dots,m.
$$

Then the KKT-like relations hold:
$$
\begin{aligned}
\nabla f_0(x^\star(t)) + \sum_{i=1}^m \lambda_i^\star(t)\,\nabla f_i(x^\star(t)) + A^\top v^\star(t) &= 0,\\[4pt]
A x^\star(t) &= b,\\[4pt]
\lambda_i^\star(t) &\ge 0,\\[4pt]
-\lambda_i^\star(t) f_i(x^\star(t)) &= \frac{1}{t}, \quad i = 1,\dots,m.
\end{aligned}
$$

Compare with the exact KKT conditions (for optimal $(x^\star,\lambda^\star,v^\star)$):
$$
\lambda_i^\star f_i(x^\star) = 0.
$$

Along the central path we have the *relaxed* complementarity condition
$$
\lambda_i^\star(t) \,f_i\big(x^\star(t)\big) = -\frac{1}{t},
$$
which tends to $0$ as $t \to \infty$. Hence the barrier formulation naturally yields approximate primal–dual solutions whose KKT residuals shrink as we increase $t$.

## Geometric and Physical Intuition

Consider the barrier-augmented objective
$$
\Psi_t(x) = t f_0(x) + \phi(x)
= t f_0(x) - \sum_{i=1}^m \log(-f_i(x)).
$$

We can interpret this as:

- $t f_0(x)$: an “external potential” pulling us toward low objective values.
- $-\log(-f_i(x))$: repulsive potentials that become infinite near the boundary $f_i(x)=0$.

At a minimizer $x^\star(t)$, we have
$$
\nabla \Psi_t(x^\star(t))
= t \nabla f_0(x^\star(t)) + \sum_{i=1}^m \frac{1}{-f_i(x^\star(t))}\,\nabla f_i(x^\star(t)) = 0
\quad (\text{up to components orthogonal to } \operatorname{Null}(A)).
$$

The gradient of the objective is exactly balanced by a weighted sum of constraint gradients. This is a *force-balance condition*:

- constraints “push back” more strongly when $x$ is close to their boundary,
- the interior-point iterates follow a smooth path that stays strictly feasible
  and moves gradually toward the optimal boundary point.

This picture explains both:

- why iterates never leave the feasible region, and  
- why the method naturally generates dual variables (the weights on constraint gradients).

 
## The Barrier Method

The barrier method solves the original inequality-constrained problem by solving a sequence of barrier subproblems with increasing $t$.

### Algorithm: Barrier Method (Conceptual Form)

Given:

- a strictly feasible starting point $x$ ($f_i(x) < 0$, $A x = b$),
- initial barrier parameter $t > 0$,
- barrier growth factor $\mu > 1$ (e.g. $\mu \in [10,20]$),
- accuracy tolerance $\varepsilon > 0$,

repeat:

1. Centering step  
   Solve the equality-constrained problem
   $$
   \min_x \; t f_0(x) - \sum_{i=1}^m \log(-f_i(x))
   \quad \text{s.t. } A x = b
   $$
   using an equality-constrained Newton method.  
   (In practice, we start from the previous solution and take a small number of Newton steps rather than “solve exactly”.)

2. Update iterate  
   Let $x$ be the resulting point (the approximate minimizer for current $t$).

3. Check stopping criterion  
   For the barrier problem, one can show
   $$
   f_0(x) - p^\star \le \frac{m}{t},
   $$
   where $p^\star$ is the optimal value of the original problem.  
   If
   $$
   \frac{m}{t} < \varepsilon,
   $$
   then stop: $x$ is guaranteed to be within $\varepsilon$ (in objective value) of optimal.

4. Increase $t$  
   Set $t := \mu t$ to weaken the barrier and move closer to the true boundary, then go back to Step 1.

Key parameters:

| Symbol | Role                                  |
|--------|----------------------------------------|
| $t$    | barrier strength (larger $t$ = weaker barrier, closer to solution) |
| $\mu$  | growth factor for $t$                 |
| $\varepsilon$ | desired accuracy (duality-gap based) |
| $m$    | number of inequality constraints      |

In practice:

- $\varepsilon$ is often in the range $10^{-3}$–$10^{-8}$,
- $\mu$ is chosen to balance outer iterations vs inner Newton steps,
- the centering step is usually solved to modest accuracy, not exactness.

 
## From Barrier Methods to Interior-Point Methods

Pure barrier methods conceptually “solve a sequence of problems for increasing $t$”. Modern *interior-point methods* refine this idea:

- they update *both* primal variables $x$ and dual variables $(\lambda, v)$,
- they use Newton’s method on the (perturbed) KKT system,
- they follow the central path by simultaneously enforcing:
  - primal feasibility ($f_i(x) \le 0$, $A x = b$),
  - dual feasibility ($\lambda_i \ge 0$),
  - relaxed complementarity ($-\lambda_i f_i(x) \approx 1/t$).

A typical *primal–dual* step solves a linearized KKT system of the form
$$
\begin{aligned}
\nabla f_0(x) + \sum_i \lambda_i \nabla f_i(x) + A^\top v &= 0,\\
f_i(x) &\le 0,\quad \lambda_i \ge 0,\\
-\lambda_i f_i(x) &\approx \frac{1}{t},\\
A x &= b.
\end{aligned}
$$

Newton’s method applied to these equations yields search directions for $(x,\lambda,v)$ that move toward the central path and reduce primal and dual residuals simultaneously. This is what modern LP/QP/SOCP/SDP solvers implement.

You do not need to implement these methods from scratch to use them: in practice, you describe your problem in a modeling language (e.g. CVX, CVXPY, JuMP) and rely on an interior-point solver under the hood.

 
## Computational and Practical Notes

Some important practical aspects:

1. Equality-constrained Newton inside  
   Each barrier subproblem is solved by equality-constrained Newton (Chapter 13). The main cost is solving the KKT linear system at each Newton step.

2. Strict feasibility  
   Barrier and interior-point methods require a strictly feasible starting point $x$ with $f_i(x) < 0$.  
   - Sometimes this is easy (e.g. nonnegativity constraints with a positive initial vector).  
   - Otherwise, a separate *phase I* problem is solved to find such a point or to certify infeasibility.

3. Step size control  
   Because the barrier blows up near the boundary, too aggressive Newton steps may try to leave the feasible region. A backtracking line search is used to ensure:
   - sufficient decrease in the barrier objective,
   - and preservation of strict feasibility ($f_i(x) < 0$ remains true).

4. Accuracy vs cost  
   The duality-gap bound $m/t$ provides a clear trade-off:
   - small $m/t$ (large $t$) → high accuracy, more iterations,
   - larger $m/t$ → faster but less precise.

5. Sparsity and structure  
   For large problems, exploiting sparsity in $A$ and in the Hessians $\nabla^2 f_i(x)$ is crucial. Interior-point methods scale well when linear algebra is carefully optimized.

 
## Equality vs Inequality-Constrained Algorithms

Finally, it is helpful to contrast the equality-only case (Chapter 13) with the inequality case.

| Aspect              | Equality constraints $A x = b$                           | Inequality constraints $f_i(x) \le 0$                        |
|---------------------|-----------------------------------------------------------|-------------------------------------------------------------|
| Feasible set        | Affine subspace                                          | General convex region with boundary                        |
| Typical algorithms  | Lagrange/KKT, equality-constrained Newton, null-space    | Barrier methods, primal–dual interior-point methods         |
| Feasibility during iteration | Can start infeasible and converge to $A x = b$  | Iterates kept strictly feasible ($f_i(x) < 0$)              |
| Complementarity     | Not present (only equalities)                            | $\lambda_i f_i(x) = 0$ at optimum, or $\approx -1/t$ along central path |
| Geometric picture   | Optimization on a flat manifold                          | Optimization in a convex region, repelled from boundary     |
| ML relevance        | Normalization, linear invariants, balance constraints    | Nonnegativity, margin constraints, safety/fairness limits   |

In summary:

- Equality-constrained methods operate directly on an affine manifold using KKT and Newton.  
- Inequality-constrained methods use smooth barriers (or primal–dual perturbed KKT systems) to stay in the interior and gradually approach the boundary and the optimal point.

Interior-point methods unify these perspectives and are the backbone of modern convex optimization software.

