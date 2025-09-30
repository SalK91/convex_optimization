- Convexity of sets: A set $C$ is convex if for any $x_1, x_2 \in C$ and $\theta \in [0,1]$, we have $\theta x_1 + (1-\theta) x_2 \in C$.
- Closed sets: A set is closed if it contains all its limit points. The closure of a set is the smallest closed set containing it.
- Extreme points: A point in a convex set is extreme if it cannot be expressed as a convex combination of two other distinct points in the set. For polyhedra, extreme points correspond to vertices.

 
# Convex Combination

A convex combination of $x_1, \dots, x_k$ is
$$
x = \sum_{i=1}^k \theta_i x_i, \quad \theta_i \geq 0, \quad \sum_{i=1}^k \theta_i = 1.
$$

This is simply a weighted average where weights are nonnegative and sum to 1.

# Convex Hull

The convex hull of a set $S$ is the collection of all convex combinations of points in $S$. It is the smallest convex set containing $S$.

```Geometric intuition: Imagine stretching a rubber band around the points; the enclosed region is the convex hull.```

# Cones
- A  cone is a set $K$ such that if $x \in K$ and $\alpha \geq 0$, then $\alpha x \in K$.  
  In words: a cone is closed under nonnegative scalar multiplication.  

- The conic hull (or convex cone) of a set $S$ is the collection of all conic combinations of points in $S$:  
  $$
  \text{cone}(S) = \left\{ \sum_{i=1}^k \theta_i x_i \;\middle|\; x_i \in S, \; \theta_i \geq 0 \right\}.
  $$

- A cone is **not necessarily a subspace** (since a subspace allows all linear combinations, including negative multiples).  
  However, every subspace is a cone (because it is closed under nonnegative scaling).  

- A cone is **not necessarily convex**. To be convex, a set must be closed under addition and convex combinations, which is not guaranteed for a general cone. A **convex cone** is a cone that is also convex, i.e., closed under nonnegative linear combinations.

# Polar Cones
- Given a cone $K \subseteq \mathbb{R}^n$, the **polar cone** of $K$ is defined as  
  $$
  K^\circ = \{ y \in \mathbb{R}^n \;\mid\; \langle y, x \rangle \leq 0 \;\; \forall x \in K \}.
  $$

- Intuitively, the polar cone consists of all vectors that form a **non-acute angle** (inner product $\leq 0$) with every vector in $K$.  

- The polar of any cone is always a **closed convex cone**, even if the original cone $K$ is not convex or not closed.  

- If $K$ is a subspace, then its polar cone $K^\circ$ is the **orthogonal complement** of $K$.  

- A useful duality property: for a closed convex cone $K$,  
  $$
  (K^\circ)^\circ = K.
  $$

# Tangent Cone
- Given a set $C \subseteq \mathbb{R}^n$ and a point $x \in C$, the **tangent cone** (also called the contingent cone) at $x$ is defined as  
  $$
  T_C(x) = \left\{ d \in \mathbb{R}^n \;\middle|\; \exists \, t_k \downarrow 0, \; x_k \in C, \; x_k \to x, \; \frac{x_k - x}{t_k} \to d \right\}.
  $$
- Intuitively: $T_C(x)$ is the set of all directions $d$ in which you can “move infinitesimally” inside $C$ starting from $x$.
- If $x$ is in the **interior** of $C$, then $T_C(x) = \mathbb{R}^n$.  
- If $x$ is on the **boundary**, $T_C(x)$ consists of directions that keep you inside $C$ locally.


# Normal Cone
- For a convex set $C \subseteq \mathbb{R}^n$ and a point $x \in C$, the **normal cone** at $x$ is defined as  
  $$
  N_C(x) = \{ v \in \mathbb{R}^n \;\mid\; \langle v, y - x \rangle \leq 0 \;\; \forall y \in C \}.
  $$

- Each $v \in N_C(x)$ defines a **supporting hyperplane** to $C$ at $x$:  

  $$
  v^T(y-x) \le 0, \quad \forall y \in C
  $$

- Relationship: the normal cone is the **polar cone** of the tangent cone:  
  $$
  N_C(x) = \big(T_C(x)\big)^\circ.
  $$

- $N_C(x)$ is always a **closed convex cone**.
- Intuitively: $N_C(x)$ is the set of vectors pointing “outward” and supporting the set $C$ at $x$.  
- **Interior point** of $C$: $N_C(x) = \{0\}$ (no outward direction, since $x$ is surrounded).  
- **Boundary point**: $N_C(x)$ contains outward-pointing directions normal to the boundary.  
- **Corner/vertex**: $N_C(x)$ is a **cone of outward normals**, capturing multiple “faces” meeting at $x$.  

## Why is the normal cone important?

### First-order optimality condition 
   - For the problem $\min f(x)$ subject to $x \in C$, a point $x^*$ is optimal if:  

     $$
     0 \in \partial f(x^*) + N_C(x^*)
     $$

     This says the slope (subgradient) of $f$ is exactly balanced by the “push back” of the constraint set.  

### Connection to tangent cone
   - The normal cone is the polar cone of the tangent cone:  

     $$
     N_C(x) = \big(T_C(x)\big)^\circ
     $$

     This duality links feasible directions (tangent cone) with blocking directions (normal cone).  

### Geometric interpretation  
   - Normals describe which directions leave the set if you try to move outward.  
   - They capture the “boundary geometry” of $C$, generalizing the idea of perpendicular vectors to surfaces. 

# Comparison of Tangent, Normal, and Polar Cones

| Cone | Applies To | Meaning | Interior Point | Boundary Point | Key Facts |
|------|------------|---------|----------------|----------------|------|
| **Tangent** $T_C(x)$ | Any set (convex nicer) | Feasible move directions | $T_C(x)=\mathbb{R}^n$ | Restricted to stay in $C$ | Local geometry of $C$ |
| **Normal** $N_C(x)$ | Convex sets (gen. exist) | Outward blocking dirs | $N_C(x)=\{0\}$ | Outward rays/cones | Closed, convex; $N_C(x)=(T_C(x))^\circ$ |
| **Polar** $K^\circ$ | Any cone $K$ | Non-acute dirs wrt $K$ | Not point-specific | Not point-specific | Closed, convex; $(K^\circ)^\circ=K$ if closed convex |


# Hyperplanes and Half-spaces
- A hyperplane is the solution set of $a^T x = b$.
- A half-space is one side of a hyperplane, defined as $a^T x \leq b$ or $a^T x \geq b$.
- These objects are convex and serve as building blocks in constraints.

Separation and Supporting Hyperplanes: One of the most powerful results in convex geometry is the separating hyperplane theorem: two disjoint convex sets can be separated by a hyperplane. For a convex set $C$ and a point $x \notin C$, there exists a hyperplane that separates $x$ from $C$. This underpins duality theory in optimisation. A supporting hyperplane touches a convex set at one or more points but does not cut through it.



