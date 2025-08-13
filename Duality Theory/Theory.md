# Duality Theory and KKT Conditions

Duality theory provides a powerful lens for understanding and solving convex optimization problems.  
By introducing **dual variables** and reformulating problems, we can sometimes find simpler or more insightful solutions — and gain certificates of optimality.



## Lagrangian Formulation

Given the convex optimization problem:
$$
\begin{aligned}
& \min_{x \in \mathbb{R}^n} & f_0(x) \\
& \text{s.t.} & f_i(x) \le 0, \quad i = 1, \dots, m \\
& & h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
$$
we define the **Lagrangian**:
$$
\mathcal{L}(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x)
$$
where:
- $\lambda_i \ge 0$ are **dual variables** (Lagrange multipliers) for inequality constraints.
- $\nu_j$ are dual variables for equality constraints.



## Primal and Dual Problems

**Primal Problem:** The original problem.  
**Dual Problem:** Maximize the best lower bound on the primal objective, given by:
$$
g(\lambda, \nu) = \inf_{x} \; \mathcal{L}(x, \lambda, \nu)
$$
The **dual optimization problem** is:
$$
\begin{aligned}
\max_{\lambda, \nu} \quad & g(\lambda, \nu) \\
\text{s.t.} \quad & \lambda \ge 0
\end{aligned}
$$



## Weak and Strong Duality

- **Weak Duality:** For any $\lambda \ge 0$, $\nu$, the dual objective $g(\lambda, \nu)$ is a **lower bound** on the primal optimal value.
$$
g(\lambda, \nu) \le p^\star
$$
- **Strong Duality:** When the primal is convex and **Slater’s condition** holds (strict feasibility), the optimal values of primal and dual match:
$$
d^\star = p^\star
$$



## Karush–Kuhn–Tucker (KKT) Conditions

For convex problems where strong duality holds, the **KKT conditions** are **necessary and sufficient** for optimality.

Given primal variables $x^\star$ and dual variables $(\lambda^\star, \nu^\star)$, the KKT conditions are:

1. **Primal Feasibility:**
$$
f_i(x^\star) \le 0, \quad h_j(x^\star) = 0
$$

2. **Dual Feasibility:**
$$
\lambda^\star \ge 0
$$

3. **Stationarity:**
$$
\nabla f_0(x^\star) + \sum_{i=1}^m \lambda_i^\star \nabla f_i(x^\star) + \sum_{j=1}^p \nu_j^\star \nabla h_j(x^\star) = 0
$$

4. **Complementary Slackness:**
$$
\lambda_i^\star f_i(x^\star) = 0, \quad \forall i
$$



## Geometric Interpretation

- The **dual variables** can be interpreted as the "prices" of relaxing constraints.  
- Complementary slackness means: if a constraint is **inactive** (strictly satisfied), its multiplier is zero; if it’s **active**, the multiplier can be positive.  
- At optimality, the gradient of the Lagrangian with respect to $x$ is zero — the "forces" from the objective and constraints balance perfectly.



## Why Duality Matters for Machine Learning

- **Support Vector Machines (SVMs)** are often solved in the dual form, enabling kernel methods.
- **Lasso regression** can be analyzed using dual norms to understand sparsity.
- Duality gives **bounds** on performance and certificates of optimality.
- Some problems are **easier** in the dual — fewer variables, simpler constraints.

