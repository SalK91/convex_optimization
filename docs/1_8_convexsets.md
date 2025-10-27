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

---

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

---

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

---

## 4.4 Polyhedra and polytopes

A **polyhedron** is an intersection of finitely many halfspaces:
$$
P = \{ x : Ax \le b \}.
$$
Polyhedra are convex. If $P$ is also bounded, it is called a **polytope**.

In linear programming, we minimise a linear objective $c^\top x$ over a polyhedron. The optimal solution, if it exists, is always attained at an extreme point (vertex) of the feasible polyhedron (Boyd and Vandenberghe, 2004).

---

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

---

## 4.6 Supporting hyperplanes and separation

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

---

## 4.7 Feasible regions in convex optimisation

In convex optimisation, the feasible set is typically something like
$$
\{ x : g_i(x) \le 0,\ i=1,\dots,m,\ h_j(x)=0,\ j=1,\dots,p \}.
$$

- If each $g_i$ is convex and each $h_j$ is affine, then the feasible set is convex.
- If $f$ is also convex, then the entire problem is a convex optimisation problem.

Thus, convex sets formalise “what it means for feasible directions to be allowed,” and this is what gives us global optimality guarantees later.
 
<!-- Convex optimization focuses on problems where both the objective and the feasible region are convex. We have seen linear (affine) sets in linear algebra; now we consider the broader class of convex sets and related geometric notions like extreme points and supporting hyperplanes. These concepts formalize the idea of “no holes or indentations” in feasible regions — any line segment between feasible points stays feasible.

Convex sets: A set $C \subseteq \mathbb{R}^n$ is convex if for any two points $x, y \in C$, the line segment connecting them lies entirely in $C$. Equivalently, for all $0\le \lambda\le 1$:

$$
\lambda x + (1 - \lambda) y \in C
$$

This must hold for every pair $x,y \in C$. For example, an interval [a,b] on the real line is convex, a solid polygon or polyhedron is convex, while a shape that has a “dent” or is disconnected is not convex. Another way to phrase convexity: if you take any weighted average of points in $C$ (with weights summing to 1 and nonnegative), the result stays in $C$. So convex sets are closed under convex combination. This also implies any convex combination of any number of points in $C$ lies in $C$ (by induction). The simplest examples: halfspaces ${x: a^T x \le b}$ are convex (any linear inequality defines a convex halfspace). Hyperplanes ${x: a^T x = b}$ are affine (and convex). Euclidean balls ${x: |x-x_0|_2 \le r}$ are convex (in fact, strictly convex — every line segment’s interior lies in the interior of the ball except at the boundary). The intersection of any collection of convex sets is convex (since it imposes all their line segment conditions simultaneously). Conversely, the convex hull of any set $S$ is the smallest convex set containing $S$, obtained by taking all convex combinations of points in $S$. Convex hulls of finitely many points are called polytopes (generalizing polygons and polyhedra).

**Geometry of convex sets:** Convex sets in the plane look like filled-in shapes without dents (e.g. a triangle or disk), whereas non-convex sets might look like crescent shapes or have indentations where a segment can exit the set. 

Convex sets are the feasible regions of convex optimization problems (possibly intersected with objective level sets). Why convexity is crucial: if $C$ is convex and we have a local minimum of a convex function in $C$, it’s also a global minimum. If $C$ were non-convex, local minima could be spurious (stuck in one “dent” of the region).

**Hyperplanes and halfspaces:** A hyperplane in $\mathbb{R}^n$ is an affine set of dimension $n-1$, defined by a single linear equation $a^T x = b$ (with $a \neq 0$). It divides the space into two halfspaces: $H^+ = {x: a^T x \ge b}$ and $H^- = {x: a^T x \le b}$. Both halfspaces are convex (they are just linear inequality constraints). Hyperplanes are the boundaries of halfspaces and often represent constraints in optimization (e.g. $a^T x \le b$). Hyperplanes are also used as supporting hyperplanes: for a convex set $C$, a supporting hyperplane at boundary point $x_0 \in \partial C$ is one that touches $C$ at $x_0$ and $C$ lies entirely on one side of it. Formally, $a^T x = a^T x_0$ is supporting if $a^T x \ge a^T x_0$ for all $x \in C$ (assuming $C$ lies in $H^+$). Supporting hyperplanes generalize the notion of a tangent line to a convex curve. For a differentiable convex function $f$, the plane $y = f(x_0) + \nabla f(x_0)^T (x - x_0)$ is a supporting hyperplane to the epigraph of $f$ (which is convex); this is essentially the first-order condition of convexity. In polyhedral sets, faces are portions of supporting hyperplanes. The **Separating Hyperplane Theorem** states that given a closed convex set $C$ and a point $y \notin C$, there exists a hyperplane that cleanly separates $y$ and $C$ (one side contains $C$, the other contains $y$). This theorem underlies duality in convex optimization and the idea that constraints can be associated with weights (multipliers) defining such separating hyperplanes.

**Extreme points:** An extreme point of a convex set $C$ is a point that cannot be expressed as a convex combination of other distinct points in $C$. Intuitively, extreme points are “corners” — if you try to take any two different points in $C$ and form a combination, you can’t land back on that extreme point. In a polytope (convex hull of a finite set), the extreme points are its vertices. For example, the extreme points of a cube are its 8 corner points; for a disk (which is convex and smooth), every point on the boundary circle is actually not extreme in the strict sense because you can write a boundary point as a combination of others on an arc if the set has some flatness — however, for a strictly convex set like a circle, no single boundary point can be averaged from two others within the set, so in fact in a circle every boundary point is extreme (any chord lies outside the circle except at endpoints). In general, strictly convex sets (like the $\ell_2$ ball) have every boundary point extreme, whereas polyhedra have a finite number of extreme points.

Extreme points play a key role in optimization: by the Krein–Milman theorem, any compact convex set is the closure of the convex hull of its extreme points. This means if an optimum exists for a linear objective over a compact convex set, it occurs at an extreme point (for linear programming, this is the simplex method’s foundation: one only needs to check vertices). Even for nonlinear convex problems, oftentimes extreme points or extreme “directions” indicate where optima lie, especially under linear objectives or linear constraints. In duality, extreme points of the primal feasible set often correspond to extreme rays of the dual cone of constraints, etc.

**Cones:** A set $K$ is a cone if for any $x \in K$ and $\alpha \ge 0$, $\alpha x \in K$. A cone is convex if additionally $x,y\in K$ implies $x+y \in K$. Convex cones are important (e.g. nonnegative orthant, PSD matrices cone) because many optimization problems can be cast as cone programs. Cones have extreme rays instead of points (directions that generate edges of the cone). For instance, the extreme rays of the positive orthant in $\mathbb{R}^n$ are the coordinate axes (each axis direction can’t be formed by positive combos of others).

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


**Polyhedra and polytopes:** A polyhedron is an intersection of finitely many halfspaces (thus convex): ${x: A x \le b}$. They can be unbounded. A polytope is the convex hull of finitely many points (thus bounded). Minkowski’s theorem says polytope = bounded polyhedron (under mild conditions). The extreme points of a polyhedron are its vertices (corner points), and some constraints may define edges or faces which themselves are lower-dimensional polyhedra. Linear programming deals with polyhedral feasible regions, and it’s well-known that the optimum (if finite) occurs at a vertex (extreme point). This is because a linear function $c^T x$ attains its maximum or minimum over a convex set at an extreme point (unless $c$ is parallel to a face, causing multiple optima along an edge).

From an algorithmic perspective, convex sets allow powerful algorithms (ellipsoid, interior-point) that leverage separation oracles or barrier functions, which wouldn’t work on non-convex sets due to local minima or disconnectedness.

**Convexity preservation and operations:** Many operations preserve convexity: intersections, affine images (linear mapping of a convex set yields a convex set), inverse linear images (preimage of a convex set under a linear map), and Minkowski sums of convex sets (sum of two convex sets $C_1 + C_2 = {x_1+x_2: x_1\in C_1, x_2\in C_2}$ is convex). Taking projections (in the sense of eliminating variables) of a convex set yields a convex set (this is essentially Farkas’ lemma reasoning). However, the image of a convex set under a nonlinear convex function is not necessarily convex.

**Relevance to optimization:** All constraints $g_i(x)\le0$ in a convex optimization problem define convex feasible sets ${x: g_i(x)\le0}$ if each $g_i$ is convex. The feasible region is then an intersection of these convex sets (still convex). The nicer the shape (polyhedral vs curved), the easier usually to optimize. Understanding extreme points helps in combinatorial or structured problems: e.g., the convex hull of all permutation matrices (describing assignment problems) has extreme points that are exactly permutation matrices, so linear objectives optimize at a permutation. In continuous domains, the idea that “difficult” points of a convex set are at the boundary leads to methods like active-set algorithms, which guess which constraints bind (active) at optimum (essentially guessing a face of the feasible set on which the optimum lies).

In summary, affine and convex geometry provides the stage on which convex optimization plays out. Affine sets (hyperplanes) carve out feasible regions, convex sets ensure tractability and global optimality, and geometric features like extreme points and supporting hyperplanes provide insight into where optima occur and how to navigate the feasible landscape. These concepts will also ground our understanding of duality: every constraint (halfspace) will have an associated dual variable, and optimal dual solutions often correspond to supporting hyperplanes of the primal feasible set at the optimal point.  -->