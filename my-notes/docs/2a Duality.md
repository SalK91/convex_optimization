# Duality Theory, KKT Conditions, and Duality Gap

Duality theory is a central tool in optimization and machine learning. It provides alternative perspectives on problems, certificates of optimality, and insights into algorithm design. Applications include Support Vector Machines (SVMs), Lasso, and ridge regression.


## 1. Convex Optimization Problem

Consider the standard convex optimization problem:

$$
\begin{aligned}
& \min_{x \in \mathbb{R}^n} & f_0(x) \\
& \text{s.t.} & f_i(x) \le 0, \quad i = 1, \dots, m \\
& & h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
$$

- $f_0$: objective function.  
- $f_i$: convex inequality constraints.  
- $h_j$: affine equality constraints.  

Feasible set:

$$
\mathcal{D} = \{ x \in \mathbb{R}^n \mid f_i(x) \le 0, \ h_j(x) = 0 \}.
$$



## 2. Lagrangian Function

The **Lagrangian** incorporates constraints into the objective:

$$
\mathcal{L}(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x)
$$

- $\lambda_i \ge 0$: dual variables for inequalities.  
- $\nu_j$: dual variables for equalities.  

**Intuition:** $\lambda_i$ represents the “price” of violating constraint $f_i(x) \le 0$. Larger $\lambda_i$ penalizes violations more.

 
## 3. Dual Function and Infimum

The **dual function** is:

$$
g(\lambda, \nu) = \inf_x \mathcal{L}(x, \lambda, \nu), \quad \lambda \ge 0
$$

### 3.1 Infimum

- The **infimum** (inf) of a function is its **greatest lower bound**: the largest number that is less than or equal to all function values.  
- Formally, for $f(x)$:

$$
\inf_x f(x) = \sup \{ y \in \mathbb{R} \mid f(x) \ge y, \ \forall x \}.
$$

- **Intuition:**  
  - If $f(x)$ has a minimum, the infimum equals the minimum.  
  - If no minimum exists, the infimum is the value approached but never reached.

**Examples:**

1. $f(x) = x^2$, $x \in \mathbb{R}$ → $\inf f(x) = 0$ at $x = 0$.  
2. $f(x) = 1/x$, $x > 0$ → $\inf f(x) = 0$, never attained.  

 
### 3.2 Supremum

- The **supremum** (sup) of a set $S \subset \mathbb{R}$ is the **least upper bound**: the smallest number greater than or equal to all elements of $S$.

$$
\sup S = \inf \{ y \in \mathbb{R} \mid y \ge s, \ \forall s \in S \}
$$

**Example:** $S = \{ x \in \mathbb{R} \mid x < 1 \}$ → $\sup S = 1$, although no maximum exists.

 
### 3.3 Why the Dual Function Provides a Lower Bound

For any feasible $x \in \mathcal{D}$ and $\lambda \ge 0$, $\nu$:

$$
\lambda_i f_i(x) \le 0, \quad \nu_j h_j(x) = 0 \implies \mathcal{L}(x, \lambda, \nu) \le f_0(x)
$$

Taking the infimum over all $x$:

$$
g(\lambda, \nu) = \inf_x \mathcal{L}(x, \lambda, \nu) \le f_0(x), \quad \forall x \in \mathcal{D}
$$

Thus, for any dual variables:

$$
g(\lambda, \nu) \le p^\star
$$

- **Interpretation:** The dual function is always a **lower bound** on the primal optimum.  
- **Geometric intuition:** Think of the dual as the **highest “floor”** under the primal objective that supports the feasible region.  

 
## 4. The Dual Problem

The **dual problem** seeks the tightest lower bound:

$$
\begin{aligned}
\max_{\lambda, \nu} \quad & g(\lambda, \nu) \\
\text{s.t.} \quad & \lambda \ge 0
\end{aligned}
$$

- Dual optimal value: $d^\star = \max_{\lambda \ge 0, \nu} g(\lambda, \nu)$.  
- Always satisfies **weak duality**: $d^\star \le p^\star$.  
- If convexity + Slater's condition hold, **strong duality**: $d^\star = p^\star$.

 
## 5. Duality Gap

The **duality gap** measures the difference between primal and dual optima:

$$
\text{Gap} = p^\star - d^\star \ge 0
$$

- **Zero gap**: strong duality (common in convex ML problems).  
- **Positive gap**: weak duality only; dual provides only a lower bound.  

### 5.1 Causes of Positive Gap

1. Nonconvex objective.  
2. Constraint qualification fails (e.g., Slater’s condition not satisfied).  
3. Dual problem is infeasible or unbounded.

### 5.2 Example

Primal problem:

$$
\min_x -x^2 \quad \text{s.t. } x \ge 1
$$

- Primal optimum: $p^\star = -1$ at $x^\star = 1$  
- Dual problem: $d^\star = -\infty$ (unbounded below)  
- Gap: $p^\star - d^\star = \infty$ → positive duality gap.

**Interpretation:** dual gives a guaranteed lower bound but may not achieve the primal optimum.

 
## 6. Karush–Kuhn–Tucker (KKT) Conditions

For convex problems with strong duality, KKT conditions fully characterize optimality.

Let $x^\star$ be primal optimal and $(\lambda^\star, \nu^\star)$ dual optimal:

1. **Primal feasibility**:  
$$
f_i(x^\star) \le 0, \quad h_j(x^\star) = 0
$$

2. **Dual feasibility**:  
$$
\lambda^\star \ge 0
$$

3. **Stationarity**:  
$$
0 \in \nabla f_0(x^\star) + \sum_i \lambda_i^\star \nabla f_i(x^\star) + \sum_j \nu_j^\star \nabla h_j(x^\star)
$$

- Use **subgradients** for nonsmooth problems (e.g., Lasso).  

4. **Complementary slackness**:  
$$
\lambda_i^\star f_i(x^\star) = 0, \quad \forall i
$$

**Intuition:** Only active constraints contribute; the “forces” of objective and constraints balance.

 
## 7. Applications in Machine Learning

### 7.1 Ridge Regression

$$
\min_w \frac12 \|y - Xw\|_2^2 + \frac{\lambda}{2} \|w\|_2^2
$$

- Smooth shrinkage, unique solution.  
- Dual view useful in **kernelized ridge regression**.

### 7.2 Lasso Regression

$$
\min_w \frac12 \|y - Xw\|_2^2 + \lambda \|w\|_1
$$

- KKT conditions explain sparsity:

$$
X_j^\top (y - Xw^\star) = \lambda s_j, \quad
s_j \in 
\begin{cases}
\{\text{sign}(w_j^\star)\}, & w_j^\star \neq 0 \\
[-1,1], & w_j^\star = 0
\end{cases}
$$

- Basis for **coordinate descent and soft-thresholding algorithms**.

### 7.3 Support Vector Machines (SVMs)

- Dual depends only on inner products $x_i^\top x_j$, enabling **kernel methods**.  
- Often more efficient if number of features $d$ exceeds number of data points $n$.

--- 

## 8. Constrained vs. Penalized Optimization

- **Constrained form**:

$$
\min_w \text{Loss}(w) \quad \text{s.t. } R(w) \le t
$$

- **Penalized form**:

$$
\min_w \text{Loss}(w) + \lambda R(w)
$$

- Lagrange multiplier $\lambda$ acts as a “price” on the constraint.  
- Equivalence holds for convex problems, but mapping $t \leftrightarrow \lambda$ may be non-unique.

 
## 9. Summary

1. **Infimum and supremum:** greatest lower bound and least upper bound.  
2. **Dual function:** $g(\lambda, \nu) = \inf_x \mathcal{L}(x, \lambda, \nu)$ always provides a **lower bound**.  
3. **Duality gap:** $p^\star - d^\star$, zero under strong duality, positive when dual does not attain primal optimum.  
4. **KKT conditions:** necessary and sufficient for convex problems with strong duality.  
5. **ML connections:** Ridge, Lasso, and SVM exploit duality for **computation, sparsity, and kernelization**.

**Key intuition:** The dual function can be visualized as the **highest supporting “floor”** under the primal objective. Maximizing it gives the tightest lower bound, and when strong duality holds, it meets the primal optimum exactly.
