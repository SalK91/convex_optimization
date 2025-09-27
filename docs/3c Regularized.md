# Regularized Approximation

## 1. Motivation: Fit vs. Complexity
When fitting a model, we often want to **balance two competing goals**:

1. **Data fidelity**: minimize how poorly the model fits the observed data ($f(x)$).  
2. **Model simplicity**: discourage overly complex solutions ($R(x)$).

This is naturally a **bicriterion optimization problem**:

- Criterion 1: $f(x)$ = data-fitting term (e.g., least squares loss $\|Ax-b\|_2^2$).  
- Criterion 2: $R(x)$ = regularization term (e.g., $\|x\|_1$, $\|x\|_2^2$, TV).  

Since minimizing both simultaneously is usually impossible, we form the **scalarized problem**:

$$
\min_x \; f(x) + \lambda R(x), \quad \lambda > 0
$$

Here, $\lambda$ controls the trade-off: small $\lambda$ emphasizes fit, large $\lambda$ emphasizes simplicity.

---

## 2. Bicriterion and Pareto Frontier
- **Pareto optimality**: a solution $x^\star$ is Pareto optimal if no other $x$ improves one criterion without worsening the other.  
- Weighted sum method:  
  - For convex $f$ and $R$, every Pareto optimal solution can be obtained from some $\lambda \ge 0$.  
  - For nonconvex problems, weighted sums may miss parts of the frontier.  

Thus, regularization is a way of choosing a point on the **Pareto frontier** between fit and complexity.

---

## 3. Why Keep $x$ Small?
Ill-posed or noisy problems (e.g., $Ax \approx b$ with ill-conditioned $A$) often admit solutions with very large $\|x\|$.  
- These large values overfit noise and are unstable.  
- Regularization (especially $\ell_2$) **controls the size of $x$**, yielding stable and robust solutions.  

**Example (Ridge regression):**

$$
\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2
$$

Leads to the normal equations:

$$
(A^\top A + \lambda I)x = A^\top b
$$

- $A^\top A + \lambda I$ is always positive definite.  
- Even if $A$ is rank-deficient, the solution is unique and stable.

---

## 4. Lagrangian Interpretation
Regularized approximation is equivalent to a **constrained optimization** formulation:

$$
\min_x f(x) \quad \text{s.t.} \quad R(x) \le t
$$

for some bound $t > 0$.  

### KKT and Duality
- The Lagrangian is:

$$
\mathcal{L}(x, \lambda) = f(x) + \lambda(R(x)-t)
$$

- Under convexity and Slater’s condition, strong duality holds.  
- KKT conditions:

$$
0 \in \partial f(x^\star) + \lambda^\star \partial R(x^\star), \quad 
\lambda^\star \ge 0, \quad
R(x^\star) \le t, \quad 
\lambda^\star (R(x^\star)-t) = 0
$$

- The penalized form:

$$
\min_x f(x) + \lambda R(x)
$$

has the same optimality condition:

$$
0 \in \partial f(x^\star) + \lambda \partial R(x^\star)
$$

Hence, solving the penalized problem corresponds to solving the constrained one for some $t$, though the $\lambda \leftrightarrow t$ mapping is monotone but not one-to-one.

---

## 5. Common Regularizers

### L2 (Ridge)
$R(x) = \|x\|_2^2$  
- Strongly convex → unique solution.  
- Encourages small coefficients, smooth solutions.  
- Bayesian view: Gaussian prior on $x$.  
- Improves conditioning of $A^\top A$.  

### L1 (Lasso)
$R(x) = \|x\|_1$  
- Convex, but not strongly convex → solutions may be non-unique.  
- Promotes **sparsity**: many coefficients exactly zero.  
- Geometric view: $\ell_1$ ball has corners aligned with coordinate axes; intersections often occur at corners → sparse solutions.  
- Bayesian view: Laplace prior on $x$.  
- Proximal operator: **soft-thresholding**  
  $$
  \operatorname{prox}_{\tau \|\cdot\|_1}(v) = \text{sign}(v) \max(|v|-\tau, 0)
  $$

### Elastic Net
$R(x) = \alpha \|x\|_1 + (1-\alpha)\|x\|_2^2$  
- Combines L1 sparsity with L2 stability.  
- Ensures uniqueness even when features are correlated.  

### Beyond L1/L2
- **General Tikhonov:** $R(x) = \|Lx\|_2^2$, where $L$ encodes smoothness (e.g., derivative operator).  
- **Total Variation (TV):** $R(x) = \|\nabla x\|_1$, promotes piecewise-constant signals.  
- **Group Lasso:** $R(x) = \sum_g \|x_g\|_2$, induces structured sparsity.  
- **Nuclear Norm:** $R(X) = \|X\|_\ast$ (sum of singular values), promotes low-rank matrices.  

---

## 6. Choosing the Regularization Parameter $\lambda$

### Trade-off
- **Too small $\lambda$:** weak regularization → overfitting, unstable solutions.  
- **Too large $\lambda$:** strong regularization → underfitting, biased solutions.  

$\lambda$ determines where on the Pareto frontier the solution lies.

### Practical Selection
- **Cross-validation (CV):**  
  - Split data into $k$ folds.  
  - Train on $k-1$ folds, validate on the held-out fold.  
  - Average validation error across folds.  
  - Choose $\lambda$ minimizing average error.  

- **Best practices:**  
  - Standardize features before using L1/Elastic Net.  
  - For time series, use blocked or rolling CV (avoid leakage).  
  - Use **nested CV** for model comparison.  
  - One-standard-error rule: prefer larger $\lambda$ within one SE of min error → simpler model.  

### Alternatives
- **Analytical rules** (ridge regression has closed-form shrinkage).  
- **Information criteria** (AIC/BIC; heuristic for Lasso).  
- **Regularization path** (trace solutions as $\lambda$ varies, pick best by validation error).  
- **Inverse problems:** discrepancy principle, L-curve, generalized CV.  

---

## 7. Algorithmic Perspective
Regularized problems often have the form:

$$
\min_x f(x) + R(x)
$$

where $f$ is smooth convex and $R$ is convex but possibly nonsmooth.

- **Proximal Gradient (ISTA, FISTA):**  
  Iterative updates using gradient of $f$ and prox of $R$.  
- **Coordinate Descent:** very effective for Lasso/Elastic Net.  
- **ADMM:** handles separable structures and constraints well.  

Proximal operators are key:  
- L2: shrinkage (scaling).  
- L1: soft-thresholding.  
- TV/nuclear norm: more advanced proximal maps.  

---

## 8. Bayesian Interpretation
- Regularization corresponds to **MAP estimation**.  
- Example: Gaussian noise $\varepsilon \sim \mathcal{N}(0,\sigma^2 I)$ and Gaussian prior $x \sim \mathcal{N}(0,\tau^2 I)$ yields:

$$
\min_x \frac{1}{2\sigma^2}\|Ax-b\|_2^2 + \frac{1}{2\tau^2}\|x\|_2^2
$$

So $\lambda = \frac{\sigma^2}{2\tau^2}$ (up to scaling).  
- L1 corresponds to a Laplace prior, inducing sparsity.

---

## 9. Key Takeaways
- Regularized approximation = **bicriterion optimization** (fit vs. complexity).  
- Penalized and constrained forms are connected via duality and KKT.  
- Regularization stabilizes ill-posed problems and improves generalization.  
- Choice of regularizer shapes the solution (small $\ell_2$, sparse $\ell_1$, structured TV/group/nuclear).  
- $\lambda$ is critical — usually chosen by cross-validation or problem-specific heuristics.  
- Proximal algorithms make regularized optimization scalable.  
- Bayesian view ties $\lambda$ to prior assumptions and noise models.
