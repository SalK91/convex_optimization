# Chapter 7: First-Order Optimality Conditions in Convex Optimization

Convex optimization enjoys a remarkable property: every local minimum is also a global minimum.  
This chapter develops the unified first-order conditions that determine whether a point is optimal — in both unconstrained and constrained convex problems, smooth or nonsmooth.

These conditions form the conceptual bridge between the subgradient theory of Chapter 6 and the KKT framework of Chapter 8.

 
## 7.1 Why Optimality Conditions Matter

Knowing when a point is optimal is essential for:

- Certifying convergence of algorithms (e.g., gradient or proximal methods),
- Understanding how constraints affect the solution,
- Handling nonsmooth objectives (e.g., hinge loss, $\ell_1$ regularization),
- Building geometric intuition about “stationarity.”

 
## 7.2 Unconstrained Convex Problems

We start with the simplest case:
\[
\min_{x \in \mathbb{R}^n} f(x),
\]
where \( f \) is convex.

### (a) Smooth Case

If \( f \) is differentiable, a point \(\hat{x}\) is optimal iff
\[
\nabla f(\hat{x}) = 0.
\]

This is the classical first-order condition.

### (b) Nonsmooth Case

When \(f\) is convex but nondifferentiable, we use subgradients (Chapter 6).

Optimality condition:
\[
0 \in \partial f(\hat{x}),
\]
where \(\partial f(x)\) is the subdifferential set.


 
## 7.3 Constrained Convex Problems

Let \( f:\mathbb{R}^n\to\mathbb{R} \) be convex and \( \mathcal{X}\subseteq\mathbb{R}^n \) be a convex feasible set.  
We consider:
\[
\min_{x\in\mathcal{X}} f(x).
\]

The geometry of \( \mathcal{X} \) now interacts with the stationarity condition.

### (a) Interior Points

If \(\hat{x}\in\operatorname{int}(\mathcal{X})\), the constraint is inactive locally, so
\[
0\in\partial f(\hat{x}).
\]
Same as the unconstrained case.

 
### (b) Boundary Points and Normal Cones

If \(\hat{x}\) lies on the boundary of \(\mathcal{X}\), feasible motion is restricted.

A point \(\hat{x}\) is optimal iff
\[
-\,\nabla f(\hat{x}) \in N_{\mathcal{X}}(\hat{x}),
\]
where \( N_{\mathcal{X}}(\hat{x}) \) is the normal cone to \(\mathcal{X}\) at \(\hat{x}\):
\[
N_{\mathcal{X}}(\hat{x})
= \{\, v \mid v^\top (x-\hat{x}) \le 0,\; \forall x\in\mathcal{X} \,\}.
\]

Geometric form: for every feasible direction \(d\) in the tangent cone
\(T_{\mathcal{X}}(\hat{x})\),
\[
\nabla f(\hat{x})^\top d \ge 0.
\]
The gradient points outward—any move within \(\mathcal{X}\) would not decrease \(f\).

 
### (c) Unified Compact Form

Both interior and boundary cases can be written as:
\[
0 \in \partial f(\hat{x}) + N_{\mathcal{X}}(\hat{x}).
\]

This is the general first-order optimality condition for convex problems with convex constraints.

 
## 7.4 Tangent and Normal Cones — Intuition

- Tangent cone \(T_{\mathcal{X}}(x)\): directions one can move within \(\mathcal{X}\).
- Normal cone \(N_{\mathcal{X}}(x)\): directions pointing *outward*, orthogonal to all feasible directions.

At an optimum:
- Interior point → gradient = 0.
- Boundary point → gradient points into the normal cone (outward from feasible region).

> The gradient (or subgradient) must “balance” the boundary forces — this geometric picture underlies KKT conditions (next chapter).

