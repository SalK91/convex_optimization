https://www.youtube.com/watch?v=d2jF3SXcFQ8
    1. https://web.stanford.edu/class/ee364a/lectures.html

# Convex Optimization and Case Studies

## Overview
This repository is a curated collection of **concepts**, **algorithms**, and **case studies** in **convex optimization** — a unifying framework that sits at the intersection of applied mathematics, computer science, and engineering.

We focus on:
- **Theoretical foundations** — understanding what makes a problem convex and why convexity matters.
- **Practical algorithms** — from classical methods like simplex and gradient descent to modern interior-point and first-order methods.
- **Real-world case studies** — demonstrating convex optimization in machine learning, control, finance, and beyond.



### Why Convex Optimization?
Convex optimization problems are those where the **objective function** is convex and the **feasible set** (set of points satisfying all constraints) is also convex.  
This structure gives us three remarkable advantages:

1. **Global optimality**  
   - Any local minimum is automatically a global minimum.
2. **Strong theory**  
   - Tools like *duality theory*, *optimality conditions*, and *sensitivity analysis* work elegantly.
3. **Algorithmic efficiency**  
   - Many convex problems can be solved to high precision in polynomial time, even at large scale.

This makes convex optimization a **cornerstone** in:
- Machine learning and AI
- Control systems
- Signal processing
- Operations research
- Finance and portfolio optimization
- Supply chain and logistics
- Resource allocation problems



## Basics: Convex Sets and Functions

To understand convex optimization, we first need the building blocks: **convex sets** and **convex functions**.

### Convex Sets
A set $S \subseteq \mathbb{R}^n$ is **convex** if, for any two points $x_1, x_2 \in S$ and any $\theta$ with $0 \leq \theta \leq 1$:
$$
\theta x_1 + (1 - \theta) x_2 \in S
$$
**Intuition:** If you pick two points in the set and connect them with a straight line, the entire line segment stays inside the set.

**Examples of convex sets:**
- Half-spaces: $\{x \mid a^T x \leq b\}$
- Balls: $\{x \mid \|x\|_2 \leq r\}$
- Affine subspaces: $\{x \mid Ax = b\}$
- Intersections of convex sets (including polyhedra)



### Convex Functions
A function $f : \mathbb{R}^n \to \mathbb{R}$ is **convex** if:
$$
f(\theta x_1 + (1 - \theta) x_2) \leq \theta f(x_1) + (1 - \theta) f(x_2),
\quad \forall \; x_1, x_2, \; \theta \in [0,1]
$$
**Intuition:** The line segment between any two points on the graph lies **above** the graph — think “bowl-shaped.”

**Examples of convex functions:**
- Norms: $\|x\|_p$ for $p \geq 1$
- Quadratics with positive semidefinite Hessian: $f(x) = x^T Q x + b^T x + c$, with $Q \succeq 0$
- Exponential: $f(x) = e^x$
- Negative logarithm: $f(x) = -\log(x)$ on $x > 0$



### Why This Matters for Optimization
If both:
- The **feasible set** (constraints) is convex, and
- The **objective function** is convex (for minimization)

then:
- **Every local optimum is global**
- We can design algorithms with **predictable convergence**
- The geometry of the problem is **well-behaved** (no “traps” from local minima)

This is the foundation for everything else in convex optimization.

## Formulating Convex Optimization Problems

A **convex optimization problem** has the general form:
$$
\begin{aligned}
& \min_{x \in \mathbb{R}^n} \quad & f_0(x) \\
& \text{s.t.} \quad & f_i(x) \leq 0, \quad i = 1, \dots, m \\
& & h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
$$

Where:
- $f_0$ (objective) is a convex function.
- Each $f_i$ (inequality constraint) is convex.
- Each $h_j$ (equality constraint) is affine.

**Key property:**  
The feasible set $\{x \mid f_i(x) \leq 0, h_j(x) = 0 \}$ is convex, ensuring that **local minima are global minima**.

---

### Common Standard Forms
Many convex optimization problems can be expressed in **standard forms**:

1. **Linear Program (LP)**  
   $$
   \min_x \; c^T x \quad
   \text{s.t.} \; Ax \leq b, \; Ex = d
   $$
   - $f_0$ is linear.
   - Feasible set is a polyhedron.

2. **Quadratic Program (QP)**  
   $$
   \min_x \; \frac{1}{2} x^T Q x + c^T x \quad
   \text{s.t.} \; Ax \leq b
   $$
   - $Q \succeq 0$ ensures convexity.

3. **Second-Order Cone Program (SOCP)**  
   $$
   \|A_i x + b_i\|_2 \leq c_i^T x + d_i
   $$
   - Captures problems with norm constraints.

4. **Semidefinite Program (SDP)**  
   $$
   \min_X \; \text{tr}(CX) \quad
   \text{s.t.} \; \mathcal{A}(X) = b, \; X \succeq 0
   $$
   - Variable is a positive semidefinite matrix.



### Geometric Interpretation
- The **objective function** shapes the “height” of the landscape.
- The **constraints** carve out the feasible region.
- In convex problems, the feasible region is **bowl-like or flat-faced**, so the global minimum lies where the lowest contour of the objective touches the region.
