# Chapter 2: Linear Algebra Foundations

Convex optimisation is geometric. To talk about convex sets, supporting hyperplanes, projections, and quadratic forms, we need linear algebra. This chapter reviews the specific linear algebra tools we will use throughout: vector spaces, inner products, norms, projections, eigenvalues, and positive semidefinite matrices (Strang, 2016; Boyd and Vandenberghe, 2004).

---

## 2.1 Vector spaces, subspaces, and affine sets

A **vector space** over $\mathbb{R}$ is a set $V$ equipped with addition and scalar multiplication satisfying the usual axioms: closure, associativity, distributivity, etc. In this book we mostly work with $V = \mathbb{R}^n$.

A **subspace** $S \subseteq \mathbb{R}^n$ is a subset that:

1. contains $0$,
2. is closed under addition,
3. is closed under scalar multiplication.

For example, the set of all solutions to $Ax = 0$ is a subspace, called the **nullspace** or **kernel** of $A$.

An **affine set** is a translated subspace. A set $A$ is affine if for any $x,y \in A$ and any $\theta \in \mathbb{R}$,  
$$
\theta x + (1-\theta) y \in A~.
$$
Every affine set can be written as
$$
x_0 + S = \{ x_0 + s : s \in S \},
$$
where $S$ is a subspace. Affine sets appear as the solution sets to linear equality constraints $Ax = b$.

Affine sets are important in optimisation because:
- Feasible sets defined by equality constraints are affine.
- Affine functions preserve convexity.

---

## 2.2 Linear combinations, span, basis, dimension

Given vectors $v_1,\dots,v_k$, any vector of the form
$$
\alpha_1 v_1 + \cdots + \alpha_k v_k
$$
is a **linear combination**. The set of all linear combinations is called the **span**:
$$
\mathrm{span}\{v_1,\dots,v_k\} = \left\{ \sum_{i=1}^k \alpha_i v_i : \alpha_i \in \mathbb{R} \right\}.
$$

A list of vectors is **linearly independent** if no nontrivial linear combination gives $0$. A **basis** of a subspace $S$ is a set of linearly independent vectors whose span is $S$. The number of vectors in a basis is the **dimension** of $S$.

Rank and nullity facts:
- The **column space** of $A$ is the span of its columns. Its dimension is $\mathrm{rank}(A)$.
- The **nullspace** of $A$ is $\{ x : Ax = 0 \}$.
- The **rank-nullity theorem** states:
$$
\mathrm{rank}(A) + \mathrm{nullity}(A) = n,
$$
where $n$ is the number of columns of $A$.

In constrained optimisation, $\mathrm{rank}(A)$ encodes the “number of independent constraints”, and the nullspace encodes feasible directions that do not violate certain constraints (Boyd and Vandenberghe, 2004).

---

## 2.3 Inner products and orthogonality

An **inner product** on $\mathbb{R}^n$ is a map $\langle \cdot,\cdot\rangle : \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ such that for all $x,y,z$ and all scalars $\alpha$:

1. $\langle x,y \rangle = \langle y,x\rangle$ (symmetry),
2. $\langle x+y,z \rangle = \langle x,z \rangle + \langle y,z\rangle$ (linearity in first argument),
3. $\langle \alpha x, y\rangle = \alpha \langle x, y\rangle$,
4. $\langle x, x\rangle \ge 0$ with equality iff $x=0$ (positive definiteness).

In $\mathbb{R}^n$, the standard inner product is the dot product:
$$
\langle x,y \rangle = x^\top y = \sum_{i=1}^n x_i y_i~.
$$

The inner product induces:
- **length (norm)**: $\|x\|_2 = \sqrt{\langle x,x\rangle}$,
- **angle**: 
$$
\cos \theta = \frac{\langle x,y\rangle}{\|x\|\|y\|}~.
$$

Two vectors are **orthogonal** if $\langle x,y\rangle = 0$. A set of vectors $\{v_i\}$ is **orthonormal** if each $\|v_i\| = 1$ and $\langle v_i, v_j\rangle = 0$ for $i\ne j$.

> Orthogonality is the language of optimality. Gradients are orthogonal to level sets; Lagrange multipliers encode orthogonality between objective gradient and constraint normals (Boyd and Vandenberghe, 2004).

### The Cauchy–Schwarz inequality

For any $x,y \in \mathbb{R}^n$:
$$
|\langle x,y\rangle| \le \|x\|\|y\|~,
$$
with equality iff $x$ and $y$ are linearly dependent. This is fundamental in analysis and optimisation; it underlies Hölder’s inequality, dual norms, and more.

---

## 2.4 Norms and distances

A function $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}$ is a **norm** if for all $x,y$ and scalar $\alpha$:

1. $\|x\| \ge 0$ and $\|x\| = 0 \iff x=0$,
2. $\|\alpha x\| = |\alpha|\|x\|$ (absolute homogeneity),
3. $\|x+y\| \le \|x\| + \|y\|$ (triangle inequality).

Important norms:

- Euclidean norm: $\|x\|_2 = \sqrt{x^\top x}$,
- $\ell_1$ norm: $\|x\|_1 = \sum_i |x_i|$,
- $\ell_\infty$ norm: $\|x\|_\infty = \max_i |x_i|$.

Norms induce distances: $d(x,y) = \|x-y\|$. The geometry of a norm ball
$$
\{ x : \|x\| \le 1 \}
$$
matters in optimisation: it describes regularisation constraints (e.g. $\ell_1$ promotes sparsity; $\ell_2$ promotes smoothness).

Each norm $\|\cdot\|$ has a **dual norm** $\|\cdot\|_*$ defined by
$$
\|y\|_* = \sup_{\|x\|\le 1} x^\top y~.
$$
For example, the dual of $\ell_1$ is $\ell_\infty$, and the dual of $\ell_2$ is itself.

Dual norms reappear in duality theory and in support functions (Hiriart-Urruty and Lemaréchal, 2001).

---

## 2.5 Linear maps, matrices, rank

A matrix $A \in \mathbb{R}^{m\times n}$ represents a linear map $x \mapsto Ax$. We care especially about:

- rank$(A)$,
- null$(A)$,
- whether $A$ is symmetric ($A = A^\top$),
- whether $A$ is positive semidefinite.

A linear constraint $Ax = b$ defines an affine set. Inequalities $Cx \le d$ define an intersection of halfspaces, hence a convex polyhedron. Both appear as feasible sets in convex optimisation (Boyd and Vandenberghe, 2004).

---

## 2.6 Eigenvalues, eigenvectors, and positive semidefinite matrices

If $A \in \mathbb{R}^{n\times n}$ is linear, a nonzero $v$ is an **eigenvector** with **eigenvalue** $\lambda$ if
$$
Av = \lambda v~.
$$

When $A$ is **symmetric** ($A = A^\top$), it has:
- real eigenvalues,
- an orthonormal eigenbasis,
- a spectral decomposition
$$
A = Q \Lambda Q^\top,
$$
where $Q$ is orthonormal and $\Lambda$ is diagonal.

A symmetric matrix $Q$ is **positive semidefinite (PSD)** if
$$
x^\top Q x \ge 0 \quad \text{for all } x~.
$$
If $x^\top Q x > 0$ for all $x\ne 0$, then $Q$ is **positive definite (PD)**.

Why this matters: if $f(x) = \tfrac{1}{2} x^\top Q x + c^\top x + d$, then
$$
\nabla^2 f(x) = Q~.
$$
So $f$ is convex iff $Q$ is PSD. Quadratic objectives with PSD Hessians are convex; with indefinite Hessians, they are not (Boyd and Vandenberghe, 2004). This is the algebraic test for convexity of quadratic forms.

---

## 2.7 Orthogonal projections and least squares

Let $S$ be a subspace of $\mathbb{R}^n$. The **orthogonal projection** of a vector $b$ onto $S$ is the unique vector $p \in S$ minimising $\|b - p\|_2$. Geometrically, $p$ is the closest point in $S$ to $b$.

If $S = \mathrm{span}\{a_1,\dots,a_k\}$ and $A = [a_1~\cdots~a_k]$, then projecting $b$ onto $S$ is equivalent to solving the least-squares problem
$$
\min_x \|Ax - b\|_2^2~.
$$
The solution $x^*$ satisfies the **normal equations**
$$
A^\top A x^* = A^\top b~.
$$

This is our first real convex optimisation problem:

- the objective $\|Ax-b\|_2^2$ is convex,
- there are no constraints,
- we can solve it in closed form.

Least squares is not just linear algebra. It is convex optimisation in disguise.

--
