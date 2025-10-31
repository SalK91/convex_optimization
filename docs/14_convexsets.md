# Chapter 4: Convex Sets and Geometric Fundamentals

Optimisation problems almost always include constraints. The feasible region — the set of points allowed by those constraints — is often a **convex set**. This chapter builds geometric intuition for convex sets, affine sets, hyperplanes, polyhedra, and supporting hyperplanes (Boyd and Vandenberghe, 2004; Rockafellar, 1970).

---

## 4.1 Convex sets

A set $C \subseteq \mathbb{R}^n$ is **convex** if for any $x,y \in C$ and any $\theta \in [0,1]$,
$$
\theta x + (1-\theta) y \in C~.
$$

Interpretation: for any two feasible points, the entire line segment between them is also feasible. No “indentations.”

### Examples
- An affine subspace: $\{ x : Ax = b \}$.
- A halfspace: $\{ x : a^\top x \le b \}$.
- An $\ell_2$ ball: $\{ x : \|x\|_2 \le r \}$.
- An $\ell_\infty$ ball: $\{ x : \|x\|_\infty \le r \}$, which is a box.
- The probability simplex: $\{ x \in \mathbb{R}^n : x \ge 0, \sum_i x_i = 1 \}$.

A set that is not convex: a crescent shape or annulus. The defining failure is: there exist $x,y$ in the set such that some convex combination leaves the set.

 

## 4.2 Affine sets, hyperplanes, and halfspaces

An **affine set** is a translate of a subspace:
$$
\{ x_0 + v : v \in S \},
$$
where $S$ is a subspace. Affine sets are convex.

A **hyperplane** in $\mathbb{R}^n$ is a set of the form
$$
H = \{ x : a^\top x = b \}
$$
for some nonzero $a \in \mathbb{R}^n$.

A **halfspace** is
$$
\{ x : a^\top x \le b \}.
$$

Halfspaces are convex; intersections of halfspaces are convex. Linear inequality constraints define intersections of halfspaces, and therefore give convex feasible regions.

 
## 4.3 Convex combinations, convex hulls

A **convex combination** of points $x_1,\dots,x_k$ is
$$
\sum_{i=1}^k \theta_i x_i
\quad\text{with}\quad
\theta_i \ge 0,\ \sum_{i=1}^k \theta_i = 1.
$$

The **convex hull** of a set $S$ is the set of all convex combinations of finitely many points of $S$. It is the “smallest” convex set containing $S$.

Convex hulls matter because:

- Polytopes (bounded polyhedra) can be described as convex hulls of finitely many points (their vertices).
- Many relaxations in optimisation replace a complicated nonconvex feasible set by its convex hull.

 
## 4.4 Polyhedra and polytopes

A **polyhedron** is an intersection of finitely many halfspaces:
$$
P = \{ x : Ax \le b \}.
$$
Polyhedra are convex and cane be unbounded. If $P$ is also bounded, it is called a **polytope**.

In linear programming, we minimise a linear objective $c^\top x$ over a polyhedron. The optimal solution, if it exists, is always attained at an extreme point (vertex) of the feasible polyhedron (Boyd and Vandenberghe, 2004).

 
## 4.5 Extreme points

Let $C$ be a convex set. A point $x \in C$ is an **extreme point** if it cannot be expressed as a strict convex combination of two other distinct points in $C$. Formally, $x$ is extreme in $C$ if whenever
$$
x = \theta y + (1-\theta) z,
\quad
0<\theta<1,
\quad
y,z \in C,
$$
then $y = z = x$.

Geometric meaning:

- Extreme points are the “corners.”
- In a polytope, extreme points are precisely the vertices.

This is why linear programming solutions are found at vertices.

## 4.6 Cones

A set $K$ is a cone if for any $x \in K$ and $\alpha \ge 0$, $\alpha x \in K$. A cone is convex if additionally $x,y\in K$ implies $x+y \in K$. Convex cones are important (e.g. nonnegative orthant, PSD matrices cone) because many optimization problems can be cast as cone programs. Cones have extreme rays instead of points (directions that generate edges of the cone). For instance, the extreme rays of the positive orthant in $\mathbb{R}^n$ are the coordinate axes (each axis direction can’t be formed by positive combos of others).

- Cones are closed under **nonnegative scaling**, but not necessarily addition.  
- **Conic hull (convex cone):** Collection of all conic combinations of points in $S$.
- A cone is not necessarily a subspace (negative multiples may not be included).  
- A convex cone is closed under addition and nonnegative scaling.  
- **Polar Cones:** Given a cone $K \subseteq \mathbb{R}^n$, the **polar cone** is:

    $$
    K^\circ = \{ y \in \mathbb{R}^n \mid \langle y, x \rangle \le 0, \; \forall x \in K \}.
    $$

    - Intuition: polar cone vectors form **non-acute angles** with every vector in $K$.  
    - Properties:  
        - Always a **closed convex cone**.  
        - If $K$ is a subspace, $K^\circ$ is the **orthogonal complement**.  
        - Duality: $(K^\circ)^\circ = K$ for closed convex cones.  

- **Tangent Cone:** For a set $C$ and point $x \in C$, the **tangent cone** $T_C(x)$ contains all directions in which one can “move infinitesimally” while remaining in $C$:

    $$
    T_C(x) = \Big\{ d \in \mathbb{R}^n \;\Big|\; \exists t_k \downarrow 0, \; x_k \in C, \; x_k \to x, \; \frac{x_k - x}{t_k} \to d \Big\}.
    $$
    
    - **Interior point:** $T_C(x) = \mathbb{R}^n$.  
    - **Boundary point:** $T_C(x)$ restricts movement to directions staying inside $C$. 
    - Tangent cones define **feasible directions** for projected gradient steps or constrained optimization.

 - **Normal Cone:** For a convex set $C$ at point $x \in C$:
    $$
    N_C(x) = \{ v \in \mathbb{R}^n \mid \langle v, y - x \rangle \le 0, \; \forall y \in C \}.
    $$
    - Each $v \in N_C(x)$ defines a **supporting hyperplane** at $x$.  
    - Relation: $N_C(x) = \big(T_C(x)\big)^\circ$ — polar of tangent cone.  
    - **Interior point:** $N_C(x) = \{0\}$.  
    - **Boundary/corner:** $N_C(x)$ is a cone of outward normals.- Appears in **first-order optimality conditions**:
    $$
    0 \in \partial f(x^*) + N_C(x^*),
    $$
    where the subgradient of $f$ is balanced by the “push-back” of constraints.

 

## 4.7 Supporting hyperplanes and separation

Convex sets can be “touched” or “separated” by hyperplanes.


### Supporting hyperplane theorem
Let $C \subseteq \mathbb{R}^n$ be a nonempty closed convex set, and let $x_0$ be a boundary point of $C$. Then there exists a nonzero $a$ such that
$$
a^\top x \le a^\top x_0 \quad \text{for all } x \in C~.
$$
In words, there is a hyperplane $a^\top x = a^\top x_0$ that “supports” $C$ at $x_0$: it touches $C$ but does not cut through it.

### Separating hyperplane theorem
If $C$ and $D$ are two disjoint nonempty convex sets, then there exists a hyperplane that separates them: some nonzero $a$ and scalar $b$ such that
$$
a^\top x \le b \quad \forall x \in C,
\qquad
a^\top y \ge b \quad \forall y \in D~.
$$

Why do we care?

- These theorems are the geometric heart of duality.
- KKT conditions can be interpreted as existence of a supporting hyperplane that is simultaneously aligned with objective and constraints.
- Subgradients of convex functions correspond to supporting hyperplanes of epigraphs.

 
## 4.7 Feasible regions in convex optimisation

In convex optimisation, the feasible set is typically something like
$$
\{ x : g_i(x) \le 0,\ i=1,\dots,m,\ h_j(x)=0,\ j=1,\dots,p \}.
$$

- If each $g_i$ is convex and each $h_j$ is affine, then the feasible set is convex.
- If $f$ is also convex, then the entire problem is a convex optimisation problem.

Thus, convex sets formalise “what it means for feasible directions to be allowed,” and this is what gives us global optimality guarantees later.
 