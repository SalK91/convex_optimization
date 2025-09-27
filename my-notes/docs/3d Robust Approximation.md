# Robust Regression: Stochastic vs. Worst-Case Formulations

 
## Setup

We study linear regression with uncertain design matrix:

$$
y = A x + \varepsilon, \quad A = \bar{A} + U,
$$

where  

- $x \in \mathbb{R}^n$ is the decision variable,  
- $y \in \mathbb{R}^m$ is the observed response,  
- $\bar{A}$ is the nominal design matrix,  
- $U$ is an uncertainty term.  

The treatment of $U$ gives rise to two main formulations: **stochastic** (probabilistic uncertainty) and **worst-case** (deterministic uncertainty).

 
## 1. Stochastic Formulation

Assume $U$ is random with  

- $\mathbb{E}[U] = 0$,  
- $\mathbb{E}[U^\top U] = P \succeq 0$,  
- finite second moment,  
- independent of $y$.  

We minimize the **expected squared residual**:

$$
\min_x \; \mathbb{E}\!\left[\|(\bar{A} + U)x - y\|_2^2\right].
$$

### Expansion

$$
\|(\bar{A}+U)x - y\|_2^2
= \|\bar{A}x - y\|_2^2 + 2(\bar{A}x - y)^\top Ux + \|Ux\|_2^2.
$$

- Cross-term vanishes since $\mathbb{E}[U]=0$ and $U$ is independent of $y$:  

$$
\mathbb{E}[(\bar{A}x - y)^\top Ux] = 0.
$$  

- Variance term simplifies:  

$$
\mathbb{E}[\|Ux\|_2^2] = x^\top P x.
$$  

### Resulting Problem

$$
\min_x \; \|\bar{A}x - y\|_2^2 + x^\top P x.
$$

- If $P = \rho I$: **ridge regression (L2 regularization)**.  
- If $P \succeq 0$ general: **generalized Tikhonov regularization**, with anisotropic penalty $\|P^{1/2}x\|_2^2$.  

### Convexity

The Hessian is  

$$
\nabla^2 f(x) = 2(\bar{A}^\top \bar{A} + P) \succeq 0.
$$  

Thus the problem is convex.  
If $P \succ 0$, it is **strongly convex** and the minimizer is unique.  

 
## 2. Worst-Case Formulation

Suppose $U$ is **unknown but bounded**:

$$
\|U\|_2 \leq \rho,
$$

where $\|\cdot\|_2$ is the **spectral norm** (largest singular value).  
We minimize the **worst-case squared residual**:

$$
\min_x \; \max_{\|U\|_2 \leq \rho} \|(\bar{A} + U)x - y\|_2^2.
$$

### Expansion via Spectral Norm Bound

For spectral norm uncertainty:

$$
\max_{\|U\|_2 \leq \rho} \|(\bar{A}+U)x - y\|_2
= \|\bar{A}x - y\|_2 + \rho \|x\|_2.
$$

This identity uses the fact that $Ux$ can align with the residual direction when $\|U\|_2 \leq \rho$.  
**Note:** If a different norm bound is used (Frobenius, $\ell_\infty$, etc.), the expression changes.

### Resulting Problem

$$
\min_x \; \left(\|\bar{A}x - y\|_2 + \rho \|x\|_2\right)^2.
$$

This is convex but **not quadratic**.  
Unlike ridge regression, the regularization is coupled *inside* the residual norm, making the solution more conservative.

 
## 3. Comparison

| Aspect            | Stochastic Formulation | Worst-Case Formulation |
|-------------------|------------------------|-------------------------|
| Model of $U$      | Random, mean zero, finite variance | Deterministic, bounded $\|U\|_2 \leq \rho$ |
| Objective         | $\|\bar{A}x - y\|_2^2 + x^\top P x$ | $(\|\bar{A}x - y\|_2 + \rho\|x\|_2)^2$ |
| Regularization    | Quadratic penalty (ellipsoidal shrinkage) | Norm inflation coupled with residual |
| Geometry          | Ellipsoidal shrinkage of $x$ (Mahalanobis norm) | Inflated residual tube, more conservative |
| Convexity         | Convex quadratic; strongly convex if $P \succ 0$ | Convex but non-quadratic |

---

## ✅ Key Takeaways

- **Stochastic robust regression** → **ridge/Tikhonov regression** (quadratic L2 penalty).  
- **Worst-case robust regression** → **inflated residual norm with L2 penalty inside the loss**, more conservative than ridge.  
- Both are convex, but their **geometry differs**:  
  - Stochastic: smooth ellipsoidal shrinkage of coefficients.  
  - Worst-case: enlarged residual “tube” that hedges against adversarial perturbations.  
