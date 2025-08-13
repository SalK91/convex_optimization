# Quadratic Programming (QP) Problem

Quadratic Programming (QP) is an optimization framework that generalizes Linear Programming by allowing a **quadratic** objective function, while keeping the constraints **linear**. Formally, the problem is:

$$
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} x^T Q x + c^T x + d \\
\text{subject to} \quad & G x \leq h \\
& A x = b
\end{aligned}
$$

**Where:**

- $x \in \mathbb{R}^n$ — the decision vector to be optimized.  
- $Q \in \mathbb{R}^{n \times n}$ — the **Hessian matrix** defining the curvature of the objective.  
- $c \in \mathbb{R}^n$ — the linear cost term.  
- $d \in \mathbb{R}$ — a constant offset (does not affect the minimizer’s location).  
- $G \in \mathbb{R}^{m \times n}$, $h \in \mathbb{R}^m$ — inequality constraints $Gx \leq h$.  
- $A \in \mathbb{R}^{p \times n}$, $b \in \mathbb{R}^p$ — equality constraints $Ax = b$.  


# Why QPs Can Be Convex Optimization Problems

Whether a QP is convex depends on **one key condition**:

- **Convex QP:** The Hessian $Q$ is **positive semidefinite** ($Q \succeq 0$).  
- **Nonconvex QP:** The Hessian has negative eigenvalues (some directions curve downward).


## 1. Convexity of the Objective

The QP objective is:

$$
f(x) = \frac{1}{2} x^T Q x + c^T x + d
$$

This is a **quadratic function**, which is:

- **Convex** if $Q \succeq 0$ (all curvature is flat or bowl-shaped).
- **Strictly convex** if $Q \succ 0$ (curvature is strictly bowl-shaped, ensuring a unique minimizer).
- **Nonconvex** if $Q$ has negative eigenvalues (some directions slope downward).

The gradient and Hessian are:

$$
\nabla f(x) = Qx + c, \quad \nabla^2 f(x) = Q
$$

Since the Hessian is constant in QPs, checking convexity is straightforward:  
> **Positive semidefinite Hessian → Convex objective → No spurious local minima.**


## 2. Convexity of Constraints

Exactly as in LPs:

- Each inequality $a^T x \leq b$ is a **half-space** (convex).  
- Each equality $a^T x = b$ is a **hyperplane** (convex).  

The feasible set:

$$
\mathcal{F} = \{ x \mid Gx \leq h, \quad Ax = b \}
$$

is the **intersection of convex sets**, hence convex.


# The Feasible Set is a Convex Polyhedron

For convex QPs:

- The **feasible region** $\mathcal{F}$ is still a convex polyhedron (because constraints are the same as in LPs).  
- The objective is a convex quadratic "bowl" sitting over that polyhedron.  
- The optimal solution is where the bowl’s lowest point **touches the feasible polyhedron**.


## Geometric Intuition: Visualizing QP

- In **LP**, the objective is a flat plane sliding over a polyhedron.  
- In **convex QP**, the objective is a curved bowl sliding over the same polyhedron.  
- If the bowl’s center lies inside the feasible region, the optimum is at that center.  
- If not, the bowl “leans” against the polyhedron’s faces, edges, or vertices — which is where the optimal solution lies.


---

✅ **Summary:**  
A QP is a convex optimization problem **if and only if** $Q \succeq 0$. In that case:

- **Objective:** Convex quadratic.  
- **Constraints:** Linear, hence convex.  
- **Feasible set:** Convex polyhedron.  
- **Solution:** Found at the point in the feasible set where the quadratic surface reaches its lowest value.
