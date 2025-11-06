# Chapter 11: Regularized Approximation – Balancing Fit and Complexity

Many practical optimization problems involve a trade-off between fitting observed data and controlling model complexity.  
Regularization formalizes this trade-off as a convex optimization problem that balances these two competing goals.  

Building on Chapter 10 (Pareto Optimality), this chapter shows that regularized models correspond to specific points on a Pareto frontier between data fidelity and simplicity.  
 


## 11.1 Motivation: Fit vs. Complexity

When fitting a model, we want both:

1. Data fidelity: minimize the loss or error $f(x)$,  
2. Model simplicity: penalize unnecessary complexity $R(x)$.

This yields a bicriterion problem:
\[
\min_{x \in \mathbb{R}^n} (f(x), R(x)).
\]

Since improving both simultaneously is typically impossible, we form a scalarized problem:
\[
\min_x \; f(x) + \lambda R(x), \qquad \lambda > 0.
\]

- $f(x)$ — data-fitting term (e.g. $\|Ax-b\|_2^2$)  
- $R(x)$ — regularizer (e.g. $\|x\|_1$, $\|x\|_2^2$, total variation)  
- $\lambda$ — trade-off parameter controlling bias–variance or fit–complexity balance.

Small $\lambda$ → better fit, possible overfitting.  
Large $\lambda$ → simpler, possibly underfit model.



## 11.2 Bicriterion Optimization and the Pareto Frontier

Regularization is a scalarised multi-objective problem (Chapter 10).  
A solution $x^*$ is Pareto optimal if no other feasible $x$ can reduce $f(x)$ without increasing $R(x)$.

- For convex $f$ and $R$, every Pareto-optimal solution can be obtained for some $\lambda \ge 0$.  
- The mapping between $\lambda$ and the constraint level $R(x) \le t$ is monotonic (though not one-to-one).  
- Nonconvex objectives may yield Pareto frontiers that cannot be fully recovered by weighted sums.

> Regularization thus selects one point on the fit–complexity Pareto frontier.



## 11.3 Why Keep $x$ Small?

Ill-posed or noisy inverse problems ($Ax \approx b$ with ill-conditioned $A$) often admit many unstable solutions.  
Large $\|x\|$ values tend to overfit noise.

Regularization — especially $\ell_2$ — stabilizes the solution by shrinking $x$.

Example: Ridge Regression
\[
\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2.
\]
The optimality condition (normal equations):
\[
(A^\top A + \lambda I)x = A^\top b.
\]
- The matrix $A^\top A + \lambda I$ is positive definite for $\lambda>0$.  
- Even if $A$ is rank-deficient, the solution is unique and stable.  
- Larger $\lambda$ improves conditioning but increases bias.



## 11.4 Constrained and Lagrangian Forms

Regularized problems can be equivalently written as constrained convex programs:
\[
\min_x f(x) \quad \text{s.t. } R(x) \le t.
\]

### Lagrangian Formulation

The Lagrangian is
\[
\mathcal{L}(x,\lambda) = f(x) + \lambda (R(x)-t).
\]
The associated penalized form
\[
\min_x f(x) + \lambda R(x)
\]
corresponds to solving the constrained problem for some $t>0$.

### KKT and Duality Connection
Under convexity and Slater’s condition:
\[
0 \in \partial f(x^*) + \lambda^* \partial R(x^*), \quad
\lambda^* \ge 0, \quad
R(x^*) \le t, \quad
\lambda^*(R(x^*)-t)=0.
\]

- Penalized and constrained forms yield the same optimality structure.  
- The mapping $\lambda \leftrightarrow t$ is monotonic but not bijective.  
- Regularization parameters thus act as Lagrange multipliers, weighting one objective against another (Chapter 10).



## 11.5 Common Regularizers

### (a) L2 Regularization (Ridge)
\[
R(x)=\|x\|_2^2.
\]

- Smooth, strongly convex → unique minimizer.  
- Shrinks coefficients uniformly; improves numerical conditioning.  
- Bayesian view: corresponds to Gaussian prior $x\sim \mathcal{N}(0,\tau^2I)$.

### (b) L1 Regularization (Lasso)
\[
R(x)=\|x\|_1 = \sum_i |x_i|.
\]

- Convex but not smooth → promotes sparsity.  
- The $\ell_1$ ball’s corners align with coordinate axes, leading to zeros in the solution.  
- Proximal operator (soft-thresholding):
  \[
  \operatorname{prox}_{\tau\|\cdot\|_1}(v)
  = \operatorname{sign}(v)\max(|v|-\tau,0).
  \]
- Bayesian view: Laplace prior $\sim e^{-|x_i|/\tau}$.

### (c) Elastic Net
\[
R(x)=\alpha\|x\|_1+(1-\alpha)\|x\|_2^2.
\]

- Combines sparsity (L1) with stability (L2).  
- Ensures uniqueness under correlated features.

### (d) Beyond L1/L2

| Regularizer | Definition | Effect |
|--|-|--|
| General Tikhonov | $R(x)=\|Lx\|_2^2$ | smoothness via linear operator $L$ |
| Total Variation (TV) | $R(x)=\|\nabla x\|_1$ | piecewise-constant signals |
| Group Lasso | $R(x)=\sum_g \|x_g\|_2$ | structured sparsity |
| Nuclear Norm | $R(X)=\|X\|_* = \sum_i \sigma_i(X)$ | low-rank matrix recovery |

Each regularizer defines a geometry of simplicity, shaping the solution’s structure.



## 11.6 Choosing the Regularization Parameter $\lambda$

### (a) Trade-off Behavior
- Small $\lambda$ → high fit, high variance.  
- Large $\lambda$ → smoother, more biased solution.  
$\lambda$ determines the location on the Pareto frontier.

### (b) Cross-Validation (CV)
Most common selection strategy:

1. Split data into $k$ folds.  
2. Train on $k-1$ folds, validate on the remaining one.  
3. Average validation error, choose $\lambda$ minimizing it.

Best practices

- Standardize features for L1/Elastic Net.  
- For time series: use blocked or rolling CV.  
- Use nested CV for fair model comparison.  
- One-standard-error rule: choose simplest model within 1 SE of best error.

### (c) Analytical / Heuristic Alternatives

- Closed-form rules (ridge shrinkage factor).  
- Information criteria (AIC/BIC for Lasso).  
- Regularization paths: trace $x^*(\lambda)$ as $\lambda$ varies.  
- Inverse problems: discrepancy principle or L-curve method.



## 11.7 Algorithmic Perspective

Regularized convex problems typically take the form
\[
\min_x f(x) + R(x),
\]
where $f$ is smooth convex and $R$ convex, possibly nonsmooth.

Key algorithms:

| Method | Idea | Suitable For |
|||--|
| Proximal Gradient (ISTA/FISTA) | Gradient step on $f$, prox step on $R$ | L1, TV, nuclear norm |
| Coordinate Descent | Update one coordinate at a time | Lasso, Elastic Net |
| ADMM | Split $f$ and $R$ for parallel structure | Large-scale structured problems |

Proximal operators (Appendix G) handle the nonsmooth term efficiently:

- L2 → scaling (shrinkage)  
- L1 → soft-thresholding  
- TV/Nuclear → more advanced proximal maps



## 11.8 Bayesian Interpretation

Regularization corresponds to MAP estimation in probabilistic models.

Assume data model \( b = A x + \varepsilon \) with Gaussian noise  
\(\varepsilon \sim \mathcal{N}(0,\sigma^2 I)\),  
and prior \( x \sim \mathcal{N}(0,\tau^2 I) \). Then:
\[
\min_x \frac{1}{2\sigma^2}\|Ax - b\|_2^2 + \frac{1}{2\tau^2}\|x\|_2^2
\]
is the MAP estimator, with $\lambda = \sigma^2/(2\tau^2)$.  

- Gaussian prior → L2 penalty  
- Laplace prior → L1 penalty (sparse MAP estimate)



