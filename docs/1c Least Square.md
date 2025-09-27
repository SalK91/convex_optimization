# 🔹 Least Squares (LS) Problem

Least Squares (LS) is one of the **canonical convex optimization problems** in statistics, machine learning, and signal processing. It seeks the vector $x \in \mathbb{R}^n$ that minimizes the **sum of squared errors**:

$$
\min_{x \in \mathbb{R}^n} \; \|A x - b\|_2^2
$$

**Where:**
- $A \in \mathbb{R}^{m \times n}$ — data or measurement matrix,  
- $b \in \mathbb{R}^m$ — observation or target vector,  
- $x \in \mathbb{R}^n$ — decision vector (unknowns to estimate).  


##  Objective Expansion

Expanding the squared norm:

$$
\|A x - b\|_2^2 = (A x - b)^T (A x - b) = x^T A^T A x - 2 b^T A x + b^T b
$$

This is a **quadratic convex function**. In standard quadratic form:

$$
f(x) = \tfrac{1}{2} x^T Q x + c^T x + d
$$

with

$$
Q = 2 A^T A, \quad c = -2 A^T b, \quad d = b^T b
$$

## ✅ Convexity

- $Q = 2 A^T A \succeq 0$ since for any $z$:  

$$
z^T (A^T A) z = \|A z\|_2^2 \geq 0
$$

- Hence LS is **convex**.  
- If $A$ has full column rank, then $A^T A \succ 0$, making the problem **strictly convex** with a **unique minimizer**.  


## 📐 Geometric Intuition

- When $m > n$ (overdetermined system), the equations $A x = b$ may not be solvable.  
- LS finds $x^\star$ such that $A x^\star$ is the **orthogonal projection** of $b$ onto the column space of $A$.  
- The residual $r = b - A x^\star$ is orthogonal to $\text{col}(A)$:

$$
A^T (b - A x^\star) = 0
$$


## 🧮 Solutions

- **Normal Equations (full-rank case):**

$$
x^\star = (A^T A)^{-1} A^T b
$$

- **General Case (possibly rank-deficient):**  
The solution set is affine. The **minimum-norm solution** is given by the **Moore–Penrose pseudoinverse**:

$$
x^\star = A^+ b
$$

- **Numerical Considerations:**  
Normal equations can be ill-conditioned. In practice:
  - Use **QR decomposition** or  
  - **SVD** (stable, gives pseudoinverse).  


# 🔒 Constrained Least Squares (CLS)

Many practical problems require **constraints** on the solution. A general CLS formulation is:

$$
\begin{aligned}
\min_{x} \quad & \|A x - b\|_2^2 \\
\text{s.t.} \quad & G x \leq h \\
& A_{\text{eq}} x = b_{\text{eq}}
\end{aligned}
$$

- Objective: convex quadratic.  
- Constraints: linear.  
- **Therefore: CLS is always a Quadratic Program (QP).**


## Example 1: Wear-and-Tear Allocation (CLS with Inequalities)

Suppose a landlord models annual **apartment wear-and-tear costs** as:

$$
c_t \approx a t + b, \quad t = 1,\dots,n
$$

with parameters $x = [a, b]^T$.  

**CLS formulation:**

$$
\min_{a,b} \sum_{t=1}^n (a t + b - c_t)^2
$$

**Constraints (practical feasibility):**

- Costs cannot be negative:  
$$
a t + b \geq 0, \quad t = 1,\dots,n
$$  

This yields a **CLS problem** with linear inequality constraints, hence a **QP**.  

---

## 💡 Example 2: Energy Consumption Fitting (CLS with Box Constraints)

Suppose we fit energy usage from appliance data:  

- $A \in \mathbb{R}^{m \times n}$ usage matrix,  
- $b \in \mathbb{R}^m$ observed energy bills.  

**CLS formulation:**

$$
\min_{x} \|A x - b\|^2
$$

**Constraints:** each appliance has a usage cap:  

$$
0 \leq x_i \leq u_i, \quad i = 1,\dots,n
$$

This is a **QP with box constraints**, often solved efficiently by projected gradient or interior-point methods.  

# 🔧 Regularized Least Squares (Ridge Regression)

A common extension in ML is **regularized LS**, e.g., **ridge regression**:

$$
\min_x \|A x - b\|^2 + \lambda \|x\|_2^2
$$

- Equivalent to CLS with a quadratic penalty on $x$.  
- Ensures uniqueness even if $A$ is rank-deficient.  
- Solution:

$$
x^\star = (A^T A + \lambda I)^{-1} A^T b
$$


# 📊 Summary

- **Unconstrained LS:** convex quadratic, closed form via pseudoinverse.  
- **CLS:** convex quadratic + linear constraints → **QP**.  
- **Regularized LS:** stabilizes solution, improves generalization.  
- **Geometry:** LS = orthogonal projection; CLS = projection with constraints.  
- **Solvers:**  
  - Small problems: QR/SVD (LS) or active-set (CLS).  
  - Large problems: iterative methods (CG, projected gradient).  
