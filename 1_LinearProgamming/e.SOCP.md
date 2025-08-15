# Second-Order Cone Programming (SOCP) Problem

Second-Order Cone Programming (SOCP) is a class of convex optimization problems that generalizes Linear and (certain) Quadratic Programs by allowing constraints involving **second-order (quadratic) cones**.  

Formally, the problem is:

$$
\begin{aligned}
\text{minimize} \quad & f^T x \\
\text{subject to} \quad & \|A_i x + b_i\|_2 \leq c_i^T x + d_i, \quad i = 1, \dots, m \\
& F x = g
\end{aligned}
$$

**Where:**

- $x \in \mathbb{R}^n$ — the decision vector.  
- $f \in \mathbb{R}^n$ — the linear objective coefficients.  
- $A_i \in \mathbb{R}^{k_i \times n}$, $b_i \in \mathbb{R}^{k_i}$ — define the affine transformation inside the norm for cone $i$.  
- $c_i \in \mathbb{R}^n$, $d_i \in \mathbb{R}$ — define the affine term on the right-hand side.  
- $F \in \mathbb{R}^{p \times n}$, $g \in \mathbb{R}^p$ — define linear equality constraints.


## The Second-Order (Quadratic) Cone

A **second-order cone** in $\mathbb{R}^k$ is:

$$
\mathcal{Q}^k = \left\{ (u,t) \in \mathbb{R}^{k-1} \times \mathbb{R} \ \middle|\ \|u\|_2 \leq t \right\}
$$

**Key properties:**
- Convex set.
- Rotationally symmetric around the $t$-axis.
- Contains all rays pointing “upward” inside the cone.



## Why SOCP is a Convex Optimization Problem

### 1. Convexity of the Objective
- The SOCP objective $f^T x$ is **affine**.
- Affine functions are both convex and concave — no curvature.


### 2. Convexity of the Constraints

Each second-order cone constraint:

$$
\|A_i x + b_i\|_2 \leq c_i^T x + d_i
$$

is convex because:
- The left-hand side $\|A_i x + b_i\|_2$ is a convex function of $x$.
- The right-hand side $c_i^T x + d_i$ is affine.
- The set $\{x \mid \|A_i x + b_i\|_2 \leq c_i^T x + d_i\}$ is a convex set.

Equality constraints $F x = g$ define a **hyperplane**, which is convex.

Since the feasible region is the **intersection of convex sets**, it is convex.



## Feasible Set Geometry

- Each SOCP constraint defines a **rotated or shifted cone** in $x$-space.
- Equality constraints slice the space with flat hyperplanes.
- The feasible set is the intersection of these cones and hyperplanes.



## Special Cases of SOCP

- **Linear Programs (LP):** If all $A_i = 0$, cone constraints reduce to linear inequalities.
- **Certain Quadratic Programs (QP):** Quadratic inequalities of the form $\|Q^{1/2}x\|_2 \leq a^T x + b$ can be rewritten as SOCP constraints.
- **Norm Constraints:** Bounds on $\ell_2$-norms (e.g., $\|x\|_2 \leq t$) are directly SOCP constraints.


## Geometric Intuition

- In **LP**, constraints are flat walls.
- In **QP**, objective is curved but constraints are flat.
- In **SOCP**, constraints themselves are curved (cone-shaped), allowing more modeling flexibility.
- The optimal solution is where the objective plane just “touches” the feasible cone-shaped region.


✅ **Summary:**
- **Objective:** Affine (linear) → convex.  
- **Constraints:** Intersection of affine equalities and convex second-order cone inequalities.  
- **Feasible set:** Convex — shaped by cones and hyperplanes.  
- **Power:** Captures LPs, norm minimization, robust optimization, and some QCQPs.  
- **Solution:** Found efficiently by interior-point methods specialized for conic programming.
