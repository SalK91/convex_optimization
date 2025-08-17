# Linear Programming (LP) Problem

Linear Programming (LP) is a powerful optimization framework used to find the best decision vector $x$ that minimizes a linear objective function, while satisfying a set of linear constraints. Formally, the problem is defined as:

$$
\begin{aligned}
\text{minimize} \quad & c^T x + d \\
\text{subject to} \quad & G x \leq h \\
& A x = b
\end{aligned}
$$

**Where:**

- $x \in \mathbb{R}^n$ — the vector of decision variables we want to determine.  
- $c \in \mathbb{R}^n$ — the cost vector shaping the objective function.  
- $d \in \mathbb{R}$ — a constant scalar offset, which shifts the objective value but *does not* influence the optimizer’s location.  
- $G \in \mathbb{R}^{m \times n}$, $h \in \mathbb{R}^m$ — define the *linear inequality* constraints $G x \leq h$.  
- $A \in \mathbb{R}^{p \times n}$, $b \in \mathbb{R}^p$ — define the *linear equality* constraints $A x = b$.  


# Why LPs Are Convex Optimization Problems

A problem is **convex** if it meets two essential criteria:

1. The **objective function** is convex (ensuring a global minimum).  
2. The **feasible set** defined by constraints forms a convex set.

### Convexity of the Objective

The LP objective $c^T x + d$ is an **affine function**:

$$
f(x) = a^T x + b
$$

where the linear part $a^T x$ is combined with a constant $b$.

> **Key Insight:** Affine functions are both **convex and concave** — they have zero curvature and guarantee no local minima traps.

### Convexity of Constraints

- Each inequality $a^T x \leq b$ defines a **half-space**, a classic convex set.  
- Each equality $a^T x = b$ defines a **hyperplane**, which is also convex.

Since the feasible region is the intersection of these constraints, it **must be convex**.



## The Feasible Set is a Convex Polyhedron

- A **half-space** is the set:

$$
\{ x \mid a^T x \leq b \}
$$

This set is convex because any line segment between two points in the half-space lies entirely inside it.

- A **hyperplane** is the set:

$$
\{ x \mid a^T x = b \}
$$

which is a flat, infinite-dimensional surface — also convex.

The overall feasible region for the LP is:

$$
\mathcal{P} = \{ x \mid G x \leq h, \quad A x = b \}
$$

This region is the **intersection of finitely many half-spaces and hyperplanes**.

> **Fundamental Fact:** The intersection of convex sets is itself convex.

Hence, $\mathcal{P}$ is a **convex polyhedron** — a geometric shape bounded by flat faces.


### Geometric Intuition: Visualizing LP Constraints

- **Inequality constraints** act like **flat walls**, slicing the space and carving out the feasible side.  
- **Equality constraints** are **perfectly flat sheets** slicing through the space.  
- The **feasible region** is the shape that remains after all these cuts — a **convex polyhedron**.

This explains why LP solutions often lie at the vertices or edges of this polyhedron — the cornerstone idea behind efficient algorithms like the simplex method.
