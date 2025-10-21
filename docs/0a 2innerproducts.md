# Inner Products and Geometry

Understanding the geometry of vector spaces is essential for convex optimization. Inner products, norms, orthogonality, and projections provide the tools to analyze gradients, measure distances, and enforce constraints. This chapter develops these concepts rigorously, links them to optimization, and provides examples relevant to machine learning.

## Inner Product and Induced Norm

An inner product on a vector space $V$ is a function $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ satisfying:

- Symmetry: $\langle x, y \rangle = \langle y, x \rangle$  
- Linearity: $\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle$  
- Positive definiteness: $\langle x, x \rangle \ge 0$ with equality only if $x = 0$

The Euclidean inner product is $\langle x, y \rangle = x^\top y$. From any inner product, we can define a norm:

$$
\|x\| = \sqrt{\langle x, x \rangle}.
$$

Cauchy–Schwarz inequality states that for any $x, y \in V$:

$$
|\langle x, y \rangle| \le \|x\| \, \|y\|.
$$

This inequality is fundamental for deriving bounds on gradient steps, dual norms, and subgradient inequalities in optimization.

## Parallelogram Law and Polarization Identity

A norm $\|\cdot\|$ induced by an inner product satisfies the parallelogram law:

$$
\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2.
$$

Intuitively, this law describes a geometric property of Euclidean-like spaces: the sum of the squares of the diagonals of a parallelogram equals the sum of the squares of all sides. It captures the essence of inner-product geometry in terms of vector lengths.


Conversely, any norm satisfying this law arises from an inner product via the polarization identity:

$$
\langle x, y \rangle = \frac{1}{4} \left( \|x + y\|^2 - \|x - y\|^2 \right).
$$

This provides a direct connection between norms and inner products, which is particularly useful when extending linear algebra and optimization concepts beyond standard Euclidean spaces.


### Hilbert Spaces
A Hilbert space is a complete vector space equipped with an inner product. Completeness here means that every Cauchy sequence (a sequence where vectors get arbitrarily close together) converges to a limit within the space. Hilbert spaces generalize Euclidean spaces to potentially infinite dimensions, allowing us to work with functions, sequences, or other objects as “vectors.”

Intuition and relevance in machine learning:

- Many optimization algorithms are first formulated in finite-dimensional Euclidean spaces (\(\mathbb{R}^n\)) but can be generalized to Hilbert spaces, enabling algorithms to handle **infinite-dimensional feature spaces**.
- Reproducing Kernel Hilbert Spaces (RKHS) are Hilbert spaces of functions with a kernel-based inner product. This allows **kernel methods** (e.g., SVMs, kernel ridge regression) to operate efficiently in very high- or infinite-dimensional spaces without explicitly computing high-dimensional coordinates (the "kernel trick").
- The properties like Cauchy–Schwarz, parallelogram law, and induced norms remain valid in Hilbert spaces, which ensures that gradient-based and convex optimization methods can be extended to these functional spaces in a mathematically sound way.



## Gram Matrices and Least-Squares Geometry

Given vectors $x_1, \dots, x_n \in \mathbb{R}^m$, the Gram matrix $G \in \mathbb{R}^{n \times n}$ is

$$
G_{ij} = \langle x_i, x_j \rangle.
$$

i.e., it contains all pairwise inner products between the vectors.  

Properties of the Gram Matrix:

- Symmetric and positive semidefinite:  
  \[
  z^\top G z \ge 0 \quad \text{for all } z \in \mathbb{R}^m
  \]  
- Rank: The rank of \(G\) equals the dimension of the span of \(\{x_1, \dots, x_n\}\).  
- Geometric interpretation: \(G\) encodes the angles and lengths of the vectors, capturing their correlations and linear dependencies.

### Connection to Least-Squares

In least-squares problems

\[
X \beta \approx y,
\]

the normal equations are

\[
X^\top X \beta = X^\top y.
\]

Here, \(X^\top X\) is the Gram matrix of the columns of \(X\). Geometrically:

- \(X^\top X\) measures how the features relate to each other via inner products.  
- Its eigenvalues determine the shape of the error surface in \(\beta\)-space. If columns of \(X\) are nearly linearly dependent, the surface becomes elongated, like a stretched ellipse.  
- The condition number of \(X^\top X\) (ratio of largest to smallest eigenvalue) quantifies this elongation:
  - High condition number (ill-conditioned): some directions in \(\beta\)-space change very slowly under gradient-based updates, leading to slow convergence.  
  - Low condition number (well-conditioned): all directions are updated more evenly, and convergence is faster.

Implications: Preprocessing techniques such as feature scaling, orthogonalization (QR decomposition), or regularization improve the condition number and accelerate convergence of optimization algorithms.

## Orthogonality and Projections

Given a subspace $W \subseteq V$ with orthonormal basis $\{q_1, \dots, q_k\}$, the projection of $x \in V$ onto $W$ is

$$
P_W(x) = \sum_{i=1}^k \langle x, q_i \rangle q_i.
$$

Properties of projections:

- $x - P_W(x)$ is orthogonal to $W$: $\langle x - P_W(x), y \rangle = 0$ for all $y \in W$  
- Projections are linear and idempotent: $P_W(P_W(x)) = P_W(x)$

In optimization, projected gradient methods rely on computing $P_W(x - \alpha \nabla f(x))$, ensuring iterates remain feasible within a subspace or convex set.

Metric projections onto convex sets $C \subseteq \mathbb{R}^n$ satisfy uniqueness and firm nonexpansiveness:

$$
\|P_C(x) - P_C(y)\|^2 \le \langle P_C(x) - P_C(y), x - y \rangle.
$$

This property guarantees algorithmic stability and is fundamental in projected gradient and proximal algorithms. Many convex optimization problems can be reformulated using proximal operators, which generalize metric projections. The firm nonexpansiveness of projections ensures that proximal iterations behave predictably and do not amplify errors. The projection can be thought of as “snapping” a point onto the feasible set in the most efficient way.  Because of convexity, there’s exactly one closest point, and the firm nonexpansiveness ensures that nearby points stay nearby after projection, which is essential for stable numerical algorithms.



## Norms and Unit-Ball Geometry

Norms induce metrics via $d(x, y) = \|x - y\|$. Common examples:

- $\ell_2$ (Euclidean) norm: $\|x\|_2 = \sqrt{\sum_i x_i^2}$  
- $\ell_1$ norm: $\|x\|_1 = \sum_i |x_i|$  
- $\ell_\infty$ norm: $\|x\|_\infty = \max_i |x_i|$

Unit-ball geometry affects optimization behavior. $\ell_1$ balls have corners promoting sparsity, while $\ell_2$ balls are smooth, influencing gradient descent and mirror descent choices. Dual norms are defined as

$$
\|y\|_* = \sup_{\|x\| \le 1} \langle y, x \rangle.
$$

For $\ell_p$ norms, the dual norm is $\ell_q$ with $1/p + 1/q = 1$. Dual norms underpin subgradient inequalities:

$$
\langle g, x - x^* \rangle \le \|g\|_* \|x - x^*\|.
$$

These inequalities are essential in primal–dual and subgradient methods.

## Summary and Connections to Optimization

- Inner products define angles, lengths, and steepest descent directions.  
- The parallelogram law ensures norms arise from inner products in Hilbert spaces.  
- Gram matrices encode feature correlations and conditioning for least-squares problems.  
- Projections onto subspaces and convex sets enforce feasibility and stability in iterative algorithms.  
- Unit-ball geometry and dual norms influence step directions, sparsity, and convergence bounds.

