# Appendix B: Support Functions and Dual Geometry (Advanced)

This appendix develops a geometric viewpoint on duality using support functions, hyperplane separation, and polarity.

---

## B.1 Support functions

Let $C \subseteq \mathbb{R}^n$ be a nonempty set. The support function of $C$ is
$$
\sigma_C(y) = \sup_{x \in C} y^\top x.
$$

Interpretation:

- For a given direction $y$, $\sigma_C(y)$ tells you how far you can go in that direction while staying in $C$.
- It is the value of the linear maximisation problem
  $$
  \max_{x \in C} y^\top x.
  $$

Key facts:

1. $\sigma_C$ is always convex, even if $C$ is not convex.
2. If $C$ is convex and closed, $\sigma_C$ essentially characterises $C$.  
   In particular, $C$ can be recovered as the intersection of halfspaces
   $$
   x^\top y \le \sigma_C(y)\quad \text{for all } y.
   $$

So support functions encode convex sets by describing all their supporting hyperplanes.

---

## B.2 Support functions and dual norms

If $C$ is the unit ball of a norm $\|\cdot\|$, i.e.
$$
C = \{ x : \|x\| \le 1 \},
$$
then
$$
\sigma_C(y)
=
\sup_{\|x\|\le 1} y^\top x
=
\|y\|_*,
$$
the dual norm of $\|\cdot\|$.

Example:

- For $\ell_2$, $\|\cdot\|_2$ is self-dual, so $\|y\|_2^* = \|y\|_2$.
- For $\ell_1$, the dual norm is $\ell_\infty$.
- For $\ell_\infty$, the dual norm is $\ell_1$.

This shows that dual norms are just support functions of norm balls.

---

## B.3 Indicator functions and conjugates

Define the indicator function of a set $C$:
$$
\delta_C(x) =
\begin{cases}
0 & x \in C, \\
+\infty & x \notin C.
\end{cases}
$$

Its convex conjugate is
$$
\delta_C^*(y)
=
\sup_x (y^\top x - \delta_C(x))
=
\sup_{x \in C} y^\top x
=
\sigma_C(y).
$$

Thus,
> The support function $\sigma_C$ is the convex conjugate of the indicator of $C$.

This is extremely important conceptually:

- Conjugates turn sets into functions.
- Duality in optimisation is often conjugacy in disguise.

---

## B.4 Hyperplane separation revisited

Recall: if $C$ is closed and convex, then at any boundary point $x_0 \in C$ there is a supporting hyperplane
$$
a^\top x \le a^\top x_0
\quad \text{for all } x \in C.
$$

This $a$ is exactly the kind of vector we would use in a support function evaluation. In fact, $a^\top x_0 = \sigma_C(a)$ if $x_0$ is an extreme point (or exposed point) in direction $a$.

Geometric interpretation:

- Lagrange multipliers in the dual problem play the role of these $a$’s.
- They identify supporting hyperplanes that “witness” optimality.

---

## B.5 Duality as support

Consider the (convex) primal problem
$$
\begin{array}{ll}
\text{minimise} & f(x) \\
\text{subject to} & x \in C,
\end{array}
$$
where $C$ is a convex feasible set.

We can rewrite the problem as minimising
$$
f(x) + \delta_C(x).
$$

The convex conjugate of $f + \delta_C$ is
$$
(f + \delta_C)^*(y)
=
\inf_{u+v=y} \left( f^*(u) + \delta_C^*(v) \right)
=
\inf_{u+v=y} \left( f^*(u) + \sigma_C(v) \right).
$$

This is already starting to look like the Lagrange dual: we are constructing a lower bound on $f(x)$ over $x \in C$ using conjugates and support functions (Rockafellar, 1970).

This view makes precise the slogan:
> “Dual variables are hyperplanes that support the feasible set and the objective from below.”

---

## B.6 Geometry of KKT and multipliers

At the optimal point $x^*$ of a convex problem, there is typically a hyperplane that supports the feasible set at $x^*$ and is aligned with the objective. That hyperplane is described by the Lagrange multipliers.

- The multipliers form a certificate that $x^*$ cannot be improved without violating feasibility.
- The dual problem is the search for the “best” such certificate.

This is precisely why KKT conditions are both necessary and sufficient in convex problems that satisfy Slater’s condition (Boyd and Vandenberghe, 2004).

---

## B.7 Why this matters

This geometric point of view is not just pretty:

- It explains why strong duality holds.
- It explains what $\mu_i^*$ and $\lambda_j^*$ “mean.”
- It clarifies why convex analysis is so tightly linked to hyperplane separation theorems.



<!-- # F.2 Support Functions and Dual Geometry

Support functions are one of the most elegant bridges between convex sets and linear optimization. For any convex set, they describe its extent in a given direction — and thus appear naturally in:

- Duality theory and convex conjugates (see Section D.1)  
- Norm analysis and dual norms (Section A.4 and A.5)  
- Subgradient calculations and optimality conditions (Section A.7 and Section D.3)  
- Projection and cutting-plane algorithms in high-dimensional optimization  

Geometrically, a support function tells you:  
> *How far can I go in direction $y$ and still remain inside the set $C$?*

---

## Definition and Geometry

Let $C \subseteq \mathbb{R}^n$ be a nonempty convex set. The support function $\sigma_C : \mathbb{R}^n \to \mathbb{R}$ is defined as:

$$
\sigma_C(y) = \sup_{x \in C} \langle y, x \rangle
$$

- $y$ is the direction vector.  
- $\langle y, x \rangle$ is the inner product (see Section A.2).  
- $\sigma_C(y)$ gives the maximum projection of $C$ along direction $y$.

It corresponds to the furthest point of $C$ in direction $y$, and hence defines a supporting hyperplane to the set.

---

## Key Properties

- Positive Homogeneity:  
  $$
  \sigma_C(\alpha y) = \alpha \sigma_C(y) \quad \text{for } \alpha \ge 0
  $$
- Convexity:  
  $$
  \sigma_C(y_1 + y_2) \le \sigma_C(y_1) + \sigma_C(y_2)
  $$
- Attainment: If $C$ is closed and bounded (compact), the supremum is attained — the max is reached at some $x^\star \in C$.  
- Set Representation:  
  Every closed convex set can be recovered from its support function:
  $$
  C = \{ x \in \mathbb{R}^n \mid \langle y, x \rangle \le \sigma_C(y) \quad \forall y \in \mathbb{R}^n \}
  $$

---

## Computation and Intuition

To compute $\sigma_C(y)$:

1. Specify the convex set $C$ (e.g., a ball, polytope, or feasible region).
2. Fix the direction $y \in \mathbb{R}^n$.
3. Maximize the dot product $\langle y, x \rangle$ over $x \in C$.

This is a linear program over $C$.

### Links to Optimization:
- In duality theory (Section D.1), linear functionals $\langle y, x \rangle$ are used to lower-bound convex functions — support functions arise naturally.
- For constraint sets defined by indicator functions (Section A.8), the support function is their convex conjugate:
  $$
  \sigma_C = \delta_C^*
  $$

---

## Examples

### Example 1: $\ell_2$ Unit Ball

Let $C = \{ x \mid \|x\|_2 \le 1 \}$

Then the support function is:

$$
\sigma_C(y) = \sup_{\|x\|_2 \le 1} \langle y, x \rangle = \|y\|_2
$$

Interpretation: the farthest point in direction $y$ lies on the boundary and aligns with $y$.

👉 This reveals that the support function of a norm ball gives the dual norm — see Section A.4.

---

### Example 2: $\ell_1$ Unit Ball

Let $C = \{ x \mid \|x\|_1 \le 1 \}$

Then:

$$
\sigma_C(y) = \|y\|_\infty
$$

Intuition: in direction $y$, the maximal point in $C$ aligns with the coordinate having largest magnitude.

This dual norm relationship is fundamental in sparsity-inducing optimization (e.g., LASSO in Section F.1).

---

### Example 3: Polytope (Convex Hull)

Let $C = \text{conv}\{v_1, \dots, v_m\}$

Then:

$$
\sigma_C(y) = \max_{i=1,\dots,m} \langle y, v_i \rangle
$$

Interpretation: the maximum projection occurs at one of the vertices of the polytope.

In LP problems (see Section C.3), this is how extreme points determine optimal solutions.

---

## Applications in Optimization

### 🔄 Duality and Convex Conjugates

The support function is the Fenchel conjugate of an indicator function:

$$
\sigma_C(y) = \delta_C^*(y)
$$

This connection underpins many dual optimization frameworks, including saddle-point methods, dual norms, and variational formulations.

---

### 📐 Dual Norms

For any norm $\|\cdot\|$, the support function of its unit ball gives the dual norm:

$$
\sigma_{B}(y) = \sup_{\|x\| \le 1} \langle y, x \rangle = \|y\|_*
$$

See Section A.5 for the definition of dual norms, and Section C.1 for how they affect step sizes and convergence geometry.

---

### 📏 Geometric Use Cases

- Compute distances to sets via duality.  
- Generate separating hyperplanes for convex sets.  
- Implement projection algorithms (e.g., mirror descent in Section K.1).  
- Construct robust constraints and worst-case bounds in uncertainty modeling (see Section E.2).

---

## Summary and Takeaways

- The support function $\sigma_C(y)$ measures how far a convex set extends in direction $y$.
- It is always convex, positively homogeneous, and subadditive.
- Support functions appear in:
  - Duality theory via convex conjugates  
  - Norm analysis via dual norms  
  - Subgradients and projections  
  - Constraint representations and recovery of convex sets
- For norm balls, $\sigma_C$ gives the dual norm.  
- For polytopes, $\sigma_C$ is the max over vertices.  
- For machine learning, support functions help model constraints, regularization penalties, and geometric algorithms.

Mental model:  
Think of a support function as a “radar scan” — it tells you the furthest point of a convex set in any given direction.
 -->
