# Support Functions

Support functions are a fundamental tool in convex analysis that capture the “extent” of a convex set in a given direction. They are widely used to:

- Represent convex sets in terms of linear functionals.  
- Compute distances, dual norms, and subgradients.  
- Facilitate duality theory, optimization algorithms, and geometric reasoning in high-dimensional spaces.


## Definitions and Formal Statements

Let $C \subseteq \mathbb{R}^n$ be a nonempty convex set. The support function$\sigma_C: \mathbb{R}^n \to \mathbb{R}$ is defined as:

\[
\sigma_C(y) = \sup_{x \in C} \langle y, x \rangle
\]

- $y$ is the direction vector.  
- $\langle y, x \rangle$ is the standard inner product in $\mathbb{R}^n$.  
- $\sigma_C(y)$ measures the maximum projection of the set $C$ along direction $y$.

### Key Properties

- $\sigma_C$ is **positively homogeneous**: $\sigma_C(\alpha y) = \alpha \sigma_C(y)$ for $\alpha \ge 0$.  
- $\sigma_C$ is **subadditive (convex)**: $\sigma_C(y_1 + y_2) \le \sigma_C(y_1) + \sigma_C(y_2)$.  
- If $C$ is compact, the supremum is attained: there exists $x^* \in C$ such that $\sigma_C(y) = \langle y, x^* \rangle$.  
- The support function **uniquely characterizes a closed convex set**: $C = \{ x : \langle y, x \rangle \le \sigma_C(y) \; \forall y \in \mathbb{R}^n \}$.

---

## Step-by-Step Analysis / How to Use

To compute or apply a support function:

1. Identify the convex set $C$.  
2. Choose a direction vector $y$.  
3. Solve the linear optimization problem:  
   \[
   \sigma_C(y) = \sup_{x \in C} \langle y, x \rangle
   \]  
4. Use properties:  
   - For **norm balls**, $\sigma_{B}(y)$ equals the **dual norm** $\|y\|_*$.  
   - For **polytopes**, the supremum occurs at one of the vertices.  
5. Apply in algorithms: support functions often appear in **dual formulations, subgradient computations, and projection-based methods**.

---

## Examples

### Example 1: Unit $\ell_2$ Ball

Let $C = \{x \in \mathbb{R}^n : \|x\|_2 \le 1\}$. Then

\[
\sigma_C(y) = \sup_{\|x\|_2 \le 1} \langle y, x \rangle = \|y\|_2
\]

- Intuition: the farthest point along $y$ on the unit ball is in the **direction of $y$**.  

### Example 2: Unit $\ell_1$ Ball

Let $C = \{x : \|x\|_1 \le 1\}$. Then

\[
\sigma_C(y) = \sup_{\|x\|_1 \le 1} \langle y, x \rangle = \|y\|_\infty
\]

- Intuition: for an $\ell_1$ ball, the extreme point along a direction is at a **vertex**, giving the maximum coordinate magnitude.

### Example 3: Polytope

Let $C = \text{conv}\{v_1, v_2, \dots, v_m\}$, the convex hull of vertices. Then

\[
\sigma_C(y) = \max_{i=1,\dots,m} \langle y, v_i \rangle
\]

- Intuition: the maximum along a direction occurs at a **vertex of the polytope**.

---

## Applications / Implications

- **Duality and Convex Conjugates:** The support function is the **convex conjugate of the indicator function** of the set:  
\[
\sigma_C = \delta_C^*
\]  
This provides a bridge between **sets and functions** in dual optimization formulations.

- **Norms and Dual Norms:** For any norm ball, the support function gives the **dual norm**. This is fundamental in **constrained optimization and step size analysis**.

- **Geometric Algorithms:**  
  - Computing distances between sets  
  - Generating separating hyperplanes  
  - Cutting-plane and projection algorithms

---

## Summary / Key Takeaways

- Support functions measure the **maximum extent of a convex set along a direction**.  
- They are convex, positively homogeneous, and fully characterize closed convex sets.  
- They provide a **link between primal sets and dual norms**, appearing naturally in duality theory and optimization algorithms.  
- For polytopes or norm balls, support functions are **easy to compute**, often giving **geometric intuition** about extreme points and directions.
