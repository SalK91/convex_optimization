# Fenchel Duality

Fenchel duality generalizes the idea of **Lagrange duality** to general convex optimization problems. It provides a systematic way to:

- Derive **dual problems** for convex programs  
- Analyze **optimality conditions** via subgradients  
- Link **primal and dual solutions** geometrically and algorithmically  

Intuition: Fenchel duality captures the interplay between a convex function and its conjugate. By representing constraints and objectives via conjugates, we can often solve a dual problem that is simpler or more structured than the primal.

 
## Definitions and Formal Statements

Let $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ and $g: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ be proper convex functions. Consider the primal problem:

\[
\min_{x \in \mathbb{R}^n} \, f(x) + g(Ax)
\]

where $A \in \mathbb{R}^{m \times n}$.

### Fenchel Dual Problem

The Fenchel dual is defined as:

\[
\max_{y \in \mathbb{R}^m} \, -f^*(A^\top y) - g^*(-y)
\]

- $f^*$ and $g^*$ are the **convex conjugates** of $f$ and $g$.  
- $y$ is the **dual variable**, often representing Lagrange multipliers for the linear mapping $Ax$.

### Weak and Strong Duality

- **Weak duality:** For any primal feasible $x$ and dual feasible $y$,

\[
f(x) + g(Ax) \ge -f^*(A^\top y) - g^*(-y)
\]

- **Strong duality:** If $f$ and $g$ are convex and satisfy **constraint qualifications** (e.g., Slater's condition), then

\[
\min_x f(x) + g(Ax) = \max_y -f^*(A^\top y) - g^*(-y)
\]

and the dual optimal solution $y^*$ gives information about the primal optimal $x^*$ via subgradients:

\[
A^\top y^* \in \partial f(x^*), \quad -y^* \in \partial g(Ax^*)
\]

 
## Step-by-Step Analysis / How to Use

1. Identify the convex functions $f$ and $g$ and linear map $A$.  
2. Compute the **convex conjugates** $f^*$ and $g^*$.  
3. Form the dual problem:

\[
\max_y -f^*(A^\top y) - g^*(-y)
\]

4. Solve the dual problem (often easier than the primal).  
5. Recover the primal solution from dual optimality using subgradients:

\[
x^* \in \partial f^*(A^\top y^*)
\]


## Examples

### Example 1: Linear Programming

Primal LP:

\[
\min_{x \ge 0} c^\top x \quad \text{s.t.} \quad Ax = b
\]

- Let $f(x) = c^\top x + \delta_{\{x \ge 0\}}(x)$, $g(z) = \delta_{\{z = b\}}(z)$  
- Conjugates: $f^*(y) = \delta_{\{y \le c\}}(y)$, $g^*(y) = b^\top y$  
- Fenchel dual:

\[
\max_y b^\top y \quad \text{s.t.} \quad A^\top y \le c
\]

This is the **standard LP dual**.

 
### Example 2: Quadratic Problem

Primal: $\min_x \frac{1}{2}\|x\|_2^2 + \delta_C(x)$, where $C$ is convex.  

- $f(x) = \frac{1}{2}\|x\|_2^2$, $g(x) = \delta_C(x)$  
- Conjugates: $f^*(y) = \frac{1}{2}\|y\|_2^2$, $g^*(y) = \sigma_C(y)$  
- Fenchel dual: $\max_y -\frac{1}{2}\|y\|_2^2 - \sigma_C(y)$  

- Optimal $x^*$ recovered via: $x^* = y^*$ (gradient of $f^*$)

 
## Applications / Implications

- **Algorithm Design:** Fenchel duality is the foundation for **primal-dual algorithms** and **proximal splitting methods**.  
- **Optimality Checks:** Dual solutions provide bounds on primal objectives via **weak duality**.  
- **Geometric Interpretation:** The dual problem represents the **tightest linear lower bounds** on the primal objective.  
- **Norms and Conjugates:** Links to support functions and dual norms are direct consequences of Fenchel duality.

---

## Summary / Key Takeaways

- Fenchel duality generalizes Lagrange duality using **convex conjugates**.  
- Weak duality always holds; strong duality requires convexity and constraint qualifications.  
- Dual solutions provide **bounds, optimality conditions, and subgradient relationships** for the primal.  
- Fenchel duality is fundamental in **convex optimization, primal-dual algorithms, and nonsmooth analysis**.
