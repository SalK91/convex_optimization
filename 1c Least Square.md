# Least Squares (LS) Problem

Least Squares is an optimization method for finding the vector $x$ that best fits a set of linear equations in the sense of minimizing the **sum of squared errors**. Formally, the problem is:

$$
\begin{aligned}
\text{minimize} \quad & \|A x - b\|_2^2
\end{aligned}
$$

**Where:**

- $x \in \mathbb{R}^n$ — the decision vector (unknowns to be estimated).  
- $A \in \mathbb{R}^{m \times n}$ — the data or measurement matrix.  
- $b \in \mathbb{R}^m$ — the observation or target vector.  

If we expand the squared norm:

$$
\|A x - b\|_2^2 = (A x - b)^T (A x - b)
$$

this becomes:

$$
\frac{1}{2} x^T (2 A^T A) x - 2 b^T A x + b^T b
$$



# Why Least Squares is a Convex Optimization Problem

The LS objective can be written as a **quadratic function**:

$$
f(x) = \frac{1}{2} x^T Q x + c^T x + d
$$

where:

$$
Q = 2 A^T A, \quad c = -2 A^T b, \quad d = b^T b
$$



## 1. Convexity of the Objective

- The matrix $Q = 2 A^T A$ is **positive semidefinite** because for any $z$:

$$
z^T (A^T A) z = \|A z\|_2^2 \geq 0
$$

- Therefore, $f(x)$ is **convex** (bowl-shaped) and **strictly convex** if $A$ has full column rank (ensuring $Q \succ 0$).

- This means **Least Squares always has a global minimum** and — in the full-rank case — that minimum is unique.



## 2. Constraints (Optional)

The **basic** LS problem is **unconstrained** — all $x \in \mathbb{R}^n$ are feasible.

However, **constrained least squares** problems can be written as:

$$
\begin{aligned}
\text{minimize} \quad & \|A x - b\|_2^2 \\
\text{subject to} \quad & G x \leq h \\
& A_{\text{eq}} x = b_{\text{eq}}
\end{aligned}
$$

In this case, the constraints are linear, so the feasible set is convex — and the problem becomes a **convex Quadratic Program**.


# Geometric Intuition: Visualizing Least Squares

- The equation $A x = b$ may not have an exact solution if the system is **overdetermined** ($m > n$).  
- Least Squares finds the $x$ that minimizes the **distance** from $A x$ to $b$ in Euclidean space.  
- Geometrically, this is the **orthogonal projection** of $b$ onto the column space of $A$.


✅ **Summary:**  
- **Objective:** Convex quadratic derived from squared Euclidean norm.  
- **Feasible set:** All of $\mathbb{R}^n$ in the basic case, convex polyhedron if constraints are added.  
- **Solution:** Given by the normal equations $A^T A x = A^T b$ (when $A^T A$ is invertible), or by projection formulas.  
- Always convex — special case of a convex Quadratic Program.
