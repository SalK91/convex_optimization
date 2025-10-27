# Chapter 8: Lagrange Duality Theory

Duality is one of the most beautiful and useful ideas in convex optimisation. Every constrained optimisation problem (the **primal**) has an associated **dual** problem. The dual problem:

- provides a lower bound on the optimal primal value,
- often has structure that is easier to analyse,
- gives certificates of optimality,
- interprets multipliers as “prices” of constraints.

In convex optimisation, under mild assumptions, the primal and dual optimal values are equal (Boyd and Vandenberghe, 2004; Rockafellar, 1970).

---

## 8.1 The primal problem

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

> Infimum (inf): the greatest lower bound of a set — the smallest value a function can approach, even if it is not attained.

---

## 8.2 The Lagrangian

We define the **Lagrangian**:
$$
L(x,\lambda,\mu)
=
f(x)
+ \sum_{i=1}^m \mu_i g_i(x)
+ \sum_{j=1}^p \lambda_j h_j(x),
$$
with multipliers $\mu \in \mathbb{R}^m$ and $\lambda \in \mathbb{R}^p$. For inequality constraints, we will later require $\mu_i \ge 0$.

Think of $\mu_i$ and $\lambda_j$ as “penalties” for violating the constraints.

---

## 8.3 The dual function

For fixed multipliers $(\lambda,\mu)$, define the **dual function**:
$$
\theta(\lambda,\mu)
=
\inf_x L(x,\lambda,\mu).
$$

Important:

- $\theta(\lambda,\mu)$ is always **concave** in $(\lambda,\mu)$, even if $f$ is not convex.
- For any $\mu \ge 0$,
$$
\theta(\lambda,\mu) \le f^\star.
$$

This last fact is called **weak duality**:
> The dual function gives lower bounds on the primal optimum.

Proof sketch of weak duality:  
For any feasible $x$ (i.e. satisfying $g_i(x) \le 0$, $h_j(x) = 0$) and any $\mu \ge 0$,
$$
L(x,\lambda,\mu)
=
f(x)
+ \sum_i \mu_i g_i(x)
+ \sum_j \lambda_j h_j(x)
\le f(x),
$$
because $g_i(x) \le 0$ and $\mu_i \ge 0$.  
So $\theta(\lambda,\mu) = \inf_x L(x,\lambda,\mu) \le f(x)$ for all feasible $x$.  
Taking the infimum over feasible $x$ gives $\theta(\lambda,\mu) \le f^\star$.

---

## 8.4 The dual problem

We now **maximise** the lower bound. The **Lagrange dual problem** is:
$$
\begin{array}{ll}
\text{maximise}_{\lambda,\mu} & \theta(\lambda,\mu) \\
\text{subject to} & \mu \ge 0.
\end{array}
$$

Because $\theta$ is concave and we are maximising it, the dual problem is always a **concave maximisation problem** (i.e. a convex optimisation problem in standard form).

Let $d^\star$ denote the optimal dual value.  
From weak duality, $d^\star \le f^\star$ always.

---

## 8.5 Strong duality and Slater’s condition

If $d^\star = f^\star$, we say **strong duality** holds.

For convex problems, strong duality typically holds under a mild regularity condition known as **Slater’s condition** (Boyd and Vandenberghe, 2004):

> If the problem is convex and there exists a strictly feasible point $\tilde{x}$ such that  
> $g_i(\tilde{x}) < 0$ for all $i$ and $h_j(\tilde{x}) = 0$ for all $j$,  
> then strong duality holds.

Consequences of strong duality:

- The gap $f^\star - d^\star$ is zero.
- There exist optimal multipliers $(\lambda^*, \mu^*)$.
- KKT conditions hold and characterise optimality.

---

## 8.6 KKT revisited via duality

The Karush–Kuhn–Tucker (KKT) conditions from Chapter 7 can also be seen as the conditions under which:

1. $x^*$ minimises the Lagrangian over $x$,
2. $(\lambda^*, \mu^*)$ maximises $\theta(\lambda,\mu)$,
3. complementary slackness holds,
4. primal feasibility and dual feasibility hold.

Under convexity + Slater, a point is optimal **if and only if** it satisfies KKT (Boyd and Vandenberghe, 2004). So KKT is both necessary and sufficient.

This is the unification:

- primal feasibility,
- dual feasibility,
- complementary slackness,
- stationarity (zero subgradient of $L$ w.r.t. $x$).

---

## 8.7 Interpretation of multipliers

The dual variables $\mu_i^*$ and $\lambda_j^*$ have interpretations:

- $\mu_i^*$ can be seen as the “shadow price” of relaxing constraint $g_i(x) \le 0$. If $\mu_i^*$ is large, then constraint $i$ is “expensive” to satisfy — it is strongly active.
- $\lambda_j^*$ plays a similar role for equality constraints.

In resource allocation problems, these multipliers act like market prices. In regularised estimation, they act like trade-off coefficients chosen by the optimisation itself.

---

## 8.8 Example: Linear programming dual

Consider a linear program in standard form:
$$
\begin{array}{ll}
\text{minimise} & c^\top x \\
\text{subject to} & Ax = b, \\
& x \ge 0~.
\end{array}
$$

Its dual is
$$
\begin{array}{ll}
\text{maximise} & b^\top y \\
\text{subject to} & A^\top y \le c~.
\end{array}
$$

This is a classical primal–dual pair. Linear programming is convex, Slater’s condition typically holds (assuming strict feasibility), and therefore strong duality holds. The LP duality theory you may have seen in undergraduate optimisation is just a special case of Lagrange duality (Boyd and Vandenberghe, 2004; Rockafellar, 1970).

---

## 8.9 Duality as geometry

Duality is geometry in disguise. The dual problem is finding the “best supporting hyperplane” that underestimates the primal objective over the feasible set. This is exactly the picture of supporting hyperplanes from Chapter 4, and exactly the subgradient picture from Chapter 6. Appendix B makes this geometric relationship precise in terms of support functions.

---

## 8.10 Summary

1. The Lagrangian builds a bridge between constrained problems and unconstrained ones.
2. The dual function gives lower bounds (weak duality).
3. Maximising the dual function gives the dual problem.
4. Under Slater’s condition, strong duality holds: no duality gap.
5. KKT conditions fall naturally out of duality.
6. Dual variables are interpretable and useful in analysis, sensitivity, and economics.
