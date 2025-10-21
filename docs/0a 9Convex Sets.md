# Convex Sets and Related Concepts

Understanding convex sets and their associated geometric structures is **central to convex optimization**. These objects define **feasible regions, constraints, and the geometry that algorithms navigate**. In machine learning, they appear everywhere: in feasible weight sets, regularizers, polyhedral constraints, and in dual formulations.


## Convex Set

A set $C \subseteq \mathbb{R}^n$ is **convex** if, for any two points $x_1, x_2 \in C$, the line segment joining them lies entirely within the set:

$$
\theta x_1 + (1-\theta) x_2 \in C, \quad \forall \theta \in [0,1].
$$

- **Closed sets:** A set is closed if it contains all its limit points. Closed sets ensure that sequences of feasible points do not “escape” the set, which is important for guarantees on convergence in optimization.  
- **Extreme points:** An extreme point of a convex set cannot be written as a convex combination of other points. For polyhedra, these correspond to vertices, and in optimization, **optimal solutions of linear programs lie at extreme points**.
 
## Convex Combination

A **convex combination** of points $x_1, \dots, x_k$ is:

$$
x = \sum_{i=1}^k \theta_i x_i, \quad \theta_i \ge 0, \quad \sum_{i=1}^k \theta_i = 1.
$$

- Think of it as a **weighted average** where weights sum to 1.  
- Convex combinations **never leave the convex hull**, which is the “safe zone” defined by your points.

**ML Intuition:** Gradient-based updates can be seen as **small convex combinations** between the current point and a target direction.

 
## Convex Hull

The **convex hull** of a set $S$ is the set of all convex combinations of points in $S$. It is the **smallest convex set containing $S$**.  

- **Geometric intuition:** Imagine stretching a rubber band around the points — the shape it forms is the convex hull.  
- **ML Relevance:** Convex hulls appear in support vector machines (margin between classes) and in approximating non-convex feasible sets using convex relaxations.

 

## Cones

A **cone** is a set $K \subseteq \mathbb{R}^n$ such that if $x \in K$ and $\alpha \ge 0$, then $\alpha x \in K$.  

- Cones are closed under **nonnegative scaling**, but not necessarily addition.  
- **Conic hull (convex cone):** Collection of all conic combinations of points in $S$:  

$$
\text{cone}(S) = \Big\{ \sum_{i=1}^k \theta_i x_i \;\Big|\; x_i \in S, \; \theta_i \ge 0 \Big\}.
$$

  - A cone is not necessarily a subspace (negative multiples may not be included).  
  - A convex cone is closed under addition and nonnegative scaling.  


## Polar Cones

Given a cone $K \subseteq \mathbb{R}^n$, the **polar cone** is:

$$
K^\circ = \{ y \in \mathbb{R}^n \mid \langle y, x \rangle \le 0, \; \forall x \in K \}.
$$

- Intuition: polar cone vectors form **non-acute angles** with every vector in $K$.  
- Properties:  
    - Always a **closed convex cone**.  
    - If $K$ is a subspace, $K^\circ$ is the **orthogonal complement**.  
    - Duality: $(K^\circ)^\circ = K$ for closed convex cones.  

**ML relevance:** Polar cones naturally appear in **dual problems** and in the derivation of **optimality conditions**.

 
## Tangent Cone

For a set $C$ and point $x \in C$, the **tangent cone** $T_C(x)$ contains all directions in which one can “move infinitesimally” while remaining in $C$:

$$
T_C(x) = \Big\{ d \in \mathbb{R}^n \;\Big|\; \exists t_k \downarrow 0, \; x_k \in C, \; x_k \to x, \; \frac{x_k - x}{t_k} \to d \Big\}.
$$

- **Interior point:** $T_C(x) = \mathbb{R}^n$.  
- **Boundary point:** $T_C(x)$ restricts movement to directions staying inside $C$.  

**ML intuition:** Tangent cones define **feasible directions** for projected gradient steps or constrained optimization.

 
## Normal Cone

For a convex set $C$ at point $x \in C$:

$$
N_C(x) = \{ v \in \mathbb{R}^n \mid \langle v, y - x \rangle \le 0, \; \forall y \in C \}.
$$

- Each $v \in N_C(x)$ defines a **supporting hyperplane** at $x$.  
- Relation: $N_C(x) = \big(T_C(x)\big)^\circ$ — polar of tangent cone.  
- **Interior point:** $N_C(x) = \{0\}$.  
- **Boundary/corner:** $N_C(x)$ is a cone of outward normals.  

**ML relevance:** Appears in **first-order optimality conditions**:

$$
0 \in \partial f(x^*) + N_C(x^*),
$$

where the subgradient of $f$ is balanced by the “push-back” of constraints.

 
## Comparison of Tangent, Normal, and Polar Cones

| Cone | Applies To | Meaning | Interior | Boundary | Key Facts |
|------|------------|---------|----------|----------|-----------|
| **Tangent** $T_C(x)$ | Any set | Feasible directions | $\mathbb{R}^n$ | Restricted | Local geometry |
| **Normal** $N_C(x)$ | Convex sets | Outward blocking directions | $\{0\}$ | Outward cone | Closed, convex; $N_C = T^\circ$ |
| **Polar** $K^\circ$ | Any cone | Non-acute directions | N/A | N/A | Closed, convex; $(K^\circ)^\circ = K$ |

---

## Hyperplanes and Half-Spaces

- **Hyperplane:** $a^T x = b$.  
- **Half-space:** One side of a hyperplane, $a^T x \le b$ or $a^T x \ge b$.  
- Convex sets can be **built from intersections of half-spaces**, which is why linear constraints are convex.  

**Separation & Supporting Hyperplanes:**

- **Separating hyperplane theorem:** Two disjoint convex sets can be separated by a hyperplane.  
- **Supporting hyperplane:** Touches a convex set at a point (or face) without cutting through.  
