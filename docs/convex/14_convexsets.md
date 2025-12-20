# Chapter 4: Convex Sets and Geometric Fundamentals

Most optimization problems are constrained. The set of points that satisfy these constraints the feasible region determines where an algorithm is allowed to search. In many machine learning and convex optimization problems, this feasible region is a convex set. Convex sets have a simple but powerful geometric property: any line segment between two feasible points remains entirely within the set. This structure eliminates irregularities and makes optimization far more predictable.

This chapter develops the geometric foundations needed to reason about convexity. We introduce affine sets, convex sets, hyperplanes, halfspaces, polyhedra, and supporting hyperplanes. These objects form the geometric language of convex analysis. Understanding their structure is essential for interpreting constraints, proving optimality conditions, and designing efficient algorithms for convex optimization.

## Convex sets

A set $ C \subseteq \mathbb{R}^n $ is convex if for any two points $ x, y \in C $ and any $ \theta \in [0,1] $,
$$
\theta x + (1 - \theta) y \in C.
$$
That is, the entire line segment between $x$ and $y$ lies inside the set. Convex sets have no “holes” or “indentations,” and this geometric regularity is what makes optimization over them tractable.

### Examples
- Affine subspaces: $ \{ x : Ax = b \} $.  
- Halfspaces: $ \{ x : a^\top x \le b \} $.  
- Euclidean balls: $ \{ x : \|x\|_2 \le r \} $.  
- $ \ell_\infty $ balls (axis-aligned boxes): $ \{ x : \|x\|_\infty \le r \} $.  
- Probability simplex: $ \{ x \in \mathbb{R}^n : x \ge 0, \ \sum_i x_i = 1 \} $.  

A set fails to be convex whenever some segment between two feasible points leaves the set—for example, a crescent or an annulus.

 
## Affine sets, hyperplanes, and halfspaces

Affine sets generalize linear subspaces by allowing a shift. A set $A$ is affine if for some point $x_0$ and subspace $S$,
$$
A = \{ x_0 + v : v \in S \}.
$$
Affine sets are always convex, since adding a fixed offset does not affect the convexity of the underlying subspace.

A hyperplane is an affine set defined by a single linear equation:
$$
H = \{ x : a^\top x = b \}, \qquad a \neq 0.
$$
Hyperplanes act as the “flat boundaries” of higher-dimensional space and are the fundamental building blocks of polyhedra.

A halfspace is one side of a hyperplane:
$$
\{ x : a^\top x \le b \}.
$$
Halfspaces are convex and serve as basic local approximations to general convex sets.

 
## Convex combinations and convex hulls

A convex combination of points $ x_1, \dots, x_k $ is a weighted average
$$
\sum_{i=1}^k \theta_i x_i, 
\qquad
\theta_i \ge 0, \qquad \sum_{i=1}^k \theta_i = 1.
$$
Convex sets are precisely those that contain all convex combinations of their points.

The convex hull of a set $S$, denoted $\operatorname{conv}(S)$, is the set of all convex combinations of finitely many points in $S$. It is the smallest convex set containing $S$. Geometrically, it is the shape you obtain by stretching a tight rubber band around the points.

Convex hulls are important because:

- Polytopes can be represented either as intersections of halfspaces or as convex hulls of their vertices.
- Many optimization relaxations replace a difficult nonconvex set by its convex hull, enabling the use of convex optimization techniques.


## Polyhedra and polytopes

A polyhedron is an intersection of finitely many halfspaces:
$$
P = \{ x : Ax \le b \}.
$$
Polyhedra are always convex; they may be bounded or unbounded.

If a polyhedron is also bounded, it is called a polytope. Polytopes include familiar shapes such as cubes, simplices, and more general polytopes that arise as feasible regions in linear programs.

 
## Extreme points

Let $ C $ be a convex set. A point $x \in C$ is an extreme point if it cannot be written as a nontrivial convex combination of other points in the set. Formally, if
$$
x = \theta y + (1 - \theta) z,
\qquad 0 < \theta < 1, \qquad y, z \in C,
$$
implies $ y = z = x $.

Geometrically, extreme points are the “corners” of a convex set. For polytopes, the extreme points are exactly the vertices. Extreme points are essential in optimization because many convex problems—such as linear programs—achieve their optima at extreme points of the feasible region. This geometric fact underlies simplex-type algorithms and supports duality theory.


## Cones

Cones generalize the idea of “directions” in geometry. They capture sets that are closed under nonnegative scaling and play a central role in convex analysis and constrained optimization.

### Basic definition

A set $K \subseteq \mathbb{R}^n$ is a cone if
$$
x \in K, \ \alpha \ge 0
\quad\Longrightarrow\quad
\alpha x \in K.
$$
A cone is convex if it is also closed under addition:
$$
x, y \in K \quad\Longrightarrow\quad x + y \in K.
$$

Cones are not required to contain negative multiples of a vector, so they are generally not subspaces. Instead of extreme points, cones have extreme rays, which represent directions that cannot be formed as positive combinations of other rays. For example, in the nonnegative orthant $ \mathbb{R}^n_{\ge 0} $, each coordinate axis direction is an extreme ray.

### Conic hull

Given any set $S$, its conic hull is the set of all conic combinations:
$$
\operatorname{cone}(S)
=
\left\{
\sum_{i=1}^k \alpha_i s_i : \alpha_i \ge 0,\ s_i \in S
\right\}.
$$
This is the smallest convex cone containing $S$. Conic hulls appear frequently in duality theory and in convex relaxations for optimization.

 
### Polar cones

For a cone $K$, the polar cone is defined as
$$
K^\circ
=
\left\{
y \in \mathbb{R}^n : \langle y, x \rangle \le 0 \ \forall x \in K
\right\}.
$$

Intuition:

- Polar vectors make a nonacute angle with every vector in $K$.  

Key properties:

- $K^\circ$ is always a closed convex cone.  
- If $K$ is a subspace, then $K^\circ$ is the orthogonal complement.  
- For any closed convex cone,  
  $$
  (K^\circ)^\circ = K.
  $$

Polar cones provide the geometric foundation for normal cones, dual cones, and many optimality conditions.

 
### Tangent cones

For a set $C$ and a point $x \in C$, the tangent cone $T_C(x)$ consists of all feasible “infinitesimal directions” from $x$:
$$
T_C(x)
=
\left\{
d : \exists\, t_k \downarrow 0,\ x_k \in C,\ x_k \to x,\ 
\frac{x_k - x}{t_k} \to d
\right\}.
$$

Intuition:

- At an interior point, $T_C(x) = \mathbb{R}^n$: all small moves are allowed.  
- At a boundary point, some directions are blocked; only directions that stay inside the set are feasible.

Tangent cones describe feasible directions for methods such as projected gradient descent or interior-point algorithms.

 
### Normal cones

For a convex set $C$, the normal cone at a point $x \in C$ is
$$
N_C(x)
=
\left\{
v : \langle v, y - x \rangle \le 0 \ \forall y \in C
\right\}.
$$

Interpretation:

- Every $v \in N_C(x)$ defines a supporting hyperplane to $C$ at $x$.  
- At interior points, the normal cone is $\{0\}$.  
- At boundary or corner points, it becomes a pointed cone of outward normals.

A fundamental relationship ties tangent and normal cones together:
$$
N_C(x) = \big( T_C(x) \big)^\circ.
$$

Normal cones appear directly in first-order optimality conditions. For a constrained problem  
$$
\min_{x \in C} f(x),
$$
a point $x^*$ is optimal only if
$$
0 \in \nabla f(x^*) + N_C(x^*).
$$
This expresses a balance between the objective’s slope and the “pushback’’ from the constraint set.

 
Cones,especially tangent and normal cones, are geometric tools that allow us to describe feasibility, optimality, and duality in convex optimization using directional information. They generalize the role that orthogonal complements play in linear algebra to nonlinear and constrained settings.


## Supporting Hyperplanes and Separation

One of the most important geometric facts about convex sets is that they can be *supported* or *separated* by hyperplanes. These results show that convex sets always admit linear boundaries that describe their shape. Later, these ideas reappear in duality, subgradients, and the KKT conditions.

### Supporting Hyperplane Theorem

Let $C \subseteq \mathbb{R}^n$ be nonempty, closed, and convex, and let $x_0$ be a boundary point of $C$. Then there exists a nonzero vector $a$ such that

$$
a^\top x \le a^\top x_0 \qquad \forall x \in C.
$$

This means that the hyperplane

$$
a^\top x = a^\top x_0
$$

touches $C$ at $x_0$ but does not cut through it. The vector $a$ is normal to the hyperplane. Intuitively, a supporting hyperplane is like a flat board pressed against the edge of a convex object. Supporting hyperplanes will later correspond exactly to subgradients of convex functions.

### Separating Hyperplane Theorem

If $C$ and $D$ are nonempty, disjoint convex sets, then a hyperplane exists that separates them. That is, there are a nonzero vector $a$ and scalar $b$ such that

$$
a^\top x \le b \quad \forall x \in C,
\qquad
a^\top y \ge b \quad \forall y \in D.
$$

The hyperplane $a^\top x = b$ places all points of $C$ on one side and all points of $D$ on the other. This is guaranteed purely by convexity. Separation is the geometric foundation of duality, where we attempt to separate the primal feasible region from violations of the constraints.

> ### Why This Matters for Optimisation
> These geometric results are central to convex optimisation:
> - Subgradients correspond to supporting hyperplanes of the epigraph of a convex function.
> - Dual variables arise from separating infeasible points from the feasible region.
> - KKT conditions express the balance between the gradient of the objective and the normals of active constraints.
> - Projection onto convex sets is well-defined because convex sets admit supporting hyperplanes.
>  Supporting and separating hyperplanes are therefore the geometric machinery behind optimality conditions and convex duality.

 ## Mental Map

 ```text
               Convex Sets & Geometric Fundamentals
     Feasible regions, geometry of constraints, and separation
                              │
                              ▼
                 Core idea: convexity removes "bad geometry"
        (segments stay inside → no holes/indentations → tractable)
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Definition of Convex Set                                  │
     │ C convex ⇔  θx + (1-θ)y ∈ C  for all x,y∈C, θ∈[0,1]      │
     │ - Geometry: every chord lies inside                       │
     │ - Optimization: feasible region supports global reasoning │
     └───────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Affine Geometry: the "flat" building blocks               │
     │ - Affine set: x0 + S                                      │
     │ - Hyperplane: {x : aᵀx = b}                               │
     │ - Halfspace:  {x : aᵀx ≤ b}                               │
     │ Role: linear constraints and local linear boundaries      │
     └───────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Convex Combinations & Convex Hull                         │
     │ - Convex combination: Σ θ_i x_i, θ_i≥0, Σθ_i=1            │
     │ - conv(S): all convex combos of points in S               │
     │ Why it matters: convexification / relaxations / geometry  │
     └───────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Polyhedra & Polytopes                                     │
     │ - Polyhedron: intersection of finitely many halfspaces    │
     │   P = {x : Ax ≤ b}                                        │
     │ - Polytope: bounded polyhedron                            │
     │ Why it matters: LP feasible sets; two views (H- vs V-form)│
     └───────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌──────────────────────────────────────────────────────────────┐
     │ Extreme Points (Corners)                                     │
     │ - x extreme ⇔ cannot be written as nontrivial convex combo  │
     │ - For polytopes: extremes = vertices                         │
     │ Optimization link: linear objectives attain optima at corners│
     └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Cones: scaling geometry for constraints & duality         │
     │ - Cone: x∈K, α≥0 ⇒ αx∈K                                  │
     │ - Convex cone: also closed under addition                 │
     │ - Conic hull cone(S): smallest convex cone containing S   │
     │ - Extreme rays replace extreme points                     │
     └───────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ Local Directional Geometry at a Point x                     │
     │ Tangent cone T_C(x): feasible infinitesimal directions      │
     │ - Interior point: T_C(x)=ℝⁿ                                 │
     │ - Boundary: directions restricted                           │
     │ Normal cone N_C(x): outward normals / supporting directions │
     │ - Interior point: N_C(x)={0}                                │
     │ - Boundary/corner: pointed cone of normals                  │
     │ Duality relation: N_C(x) = (T_C(x))°                        │
     └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Supporting Hyperplanes & Separation                       │
     │ Supporting hyperplane at boundary point x0:               │
     │   ∃a≠0 s.t. aᵀx ≤ aᵀx0  for all x∈C                       │
     │ Separating hyperplane for disjoint convex sets C,D:       │
     │   ∃a,b s.t. aᵀx ≤ b ≤ aᵀy  for x∈C, y∈D                   │
     │ Why it matters: geometry behind subgradients and duality  │
     └───────────────────────────────────────────────────────────┘
    

```