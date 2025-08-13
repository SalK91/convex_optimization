# Quadratically Constrained Quadratic Programming (QCQP) Problem

A Quadratically Constrained Quadratic Program (QCQP) is an optimization problem in which both the **objective** and the **constraints** can be quadratic functions. Formally, the problem is:

$$
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} x^T Q_0 x + c_0^T x + d_0 \\
\text{subject to} \quad & \frac{1}{2} x^T Q_i x + c_i^T x + d_i \leq 0, \quad i = 1, \dots, m \\
& A x = b
\end{aligned}
$$

**Where:**

- $x \in \mathbb{R}^n$ — the decision vector.  
- $Q_0, Q_i \in \mathbb{R}^{n \times n}$ — symmetric matrices defining curvature of the objective and constraints.  
- $c_0, c_i \in \mathbb{R}^n$ — linear terms in the objective and constraints.  
- $d_0, d_i \in \mathbb{R}$ — constant offsets.  
- $A \in \mathbb{R}^{p \times n}$, $b \in \mathbb{R}^p$ — equality constraints (linear).  


# Why QCQPs Can Be Convex Optimization Problems

QCQPs are **not automatically convex** — convexity requires specific conditions:

1. **Objective convexity:**  
   $Q_0 \succeq 0$ (positive semidefinite Hessian for the objective).

2. **Constraint convexity:**  
   For each inequality constraint $i$, $Q_i \succeq 0$ so that  
   $ \frac{1}{2} x^T Q_i x + c_i^T x + d_i \leq 0$ defines a convex set.

3. **Equality constraints:**  
   Must be affine (linear), e.g., $A x = b$.


## 1. Convexity of the Objective

The QCQP objective:

$$
f_0(x) = \frac{1}{2} x^T Q_0 x + c_0^T x + d_0
$$

is convex **iff** $Q_0 \succeq 0$.

---

## 2. Convexity of Constraints

A single quadratic constraint:

$$
f_i(x) = \frac{1}{2} x^T Q_i x + c_i^T x + d_i \leq 0
$$

defines a convex feasible set **iff** $Q_i \succeq 0$.  

If any $Q_i$ is **not** positive semidefinite, the constraint set becomes **nonconvex**, and the overall problem is nonconvex.

---

# Feasible Set Geometry

- If **all** $Q_i \succeq 0$, inequality constraints define **convex quadratic regions** (ellipsoids, elliptic cylinders, or half-spaces).
- Equality constraints $A x = b$ cut flat slices through these regions.
- The feasible set is the **intersection of convex quadratic sets and affine sets** — hence convex.

---

## Geometric Intuition: Visualizing QCQP

- In **QP**, only the objective is curved; constraints are flat.  
- In **QCQP**, constraints can also be curved — forming shapes like ellipsoids or paraboloids.  
- Convex QCQPs look like a “bowl” objective contained within (or pressed against) curved convex walls.  
- Nonconvex QCQPs can have **holes** or **disconnected regions**, making them much harder to solve.

---

✅ **Summary:**  
A QCQP is a convex optimization problem **if and only if**:

- $Q_0 \succeq 0$ (objective convexity), and  
- $Q_i \succeq 0$ for all $i$ (each quadratic inequality constraint convex), and  
- All equality constraints are affine.  

When these hold:
- **Objective:** Convex quadratic.  
- **Constraints:** Convex quadratic or affine.  
- **Feasible set:** Intersection of convex sets (can be curved).  
- **Solution:** Found where the objective’s minimum touches the convex feasible region.
