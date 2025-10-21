# Linear Algebra Basics: Vectors, Matrices, and Subspaces

Linear algebra provides the foundation for convex optimization. It equips us with the tools to reason about high-dimensional spaces, linear transformations, and projections. These concepts appear repeatedly in optimization algorithms such as gradient descent, proximal methods, and low-rank approximations. In this chapter, we develop a rigorous understanding of vector spaces, subspaces, rank, nullspaces, orthonormal bases, and decompositions, with explicit connections to machine learning and numerical considerations.

## Vector Spaces and Subspaces

A vector space $V$ over the real numbers $\mathbb{R}$ is a set of elements, called vectors, equipped with addition and scalar multiplication operations that satisfy closure, associativity, commutativity, distributivity, existence of a zero vector, and existence of additive inverses. A subspace $W \subseteq V$ is a subset that itself forms a vector space under the same operations.

The span of a set of vectors $\{v_1, \dots, v_k\} \subseteq V$ is the set of all linear combinations of these vectors:

$$
\text{span}\{v_1, \dots, v_k\} = \left\{ \sum_{i=1}^k \alpha_i v_i \;\middle|\; \alpha_i \in \mathbb{R} \right\}.
$$

Vectors $v_1, \dots, v_k$ are linearly independent if

$$
\sum_{i=1}^k \alpha_i v_i = 0 \implies \alpha_i = 0 \text{ for all } i.
$$

A basis of a vector space is a set of linearly independent vectors that spans the space. The dimension of the space is the number of vectors in any basis. Once a basis $\{b_1, \dots, b_n\}$ is chosen, any vector $x \in V$ can be uniquely represented as

$$
x = \sum_{i=1}^n x_i b_i
$$

where $x_i$ are the coordinates of $x$ in this basis.


In optimization, subspaces define feasible directions, constraint surfaces, and low-dimensional representations used in dimensionality reduction methods such as principal component analysis and in projected gradient methods.



An affine set is a translate of a subspace. In finite dimensions, an affine set can be written as
$$
\{x \in \mathbb{R}^n : A x = b\},
$$
where $A$ is a matrix and $b$ is a vector. Affine sets generalise lines and planes which need not pass through the origin.
## Inner Product and Orthogonality

An inner product on a vector space $V$ is a map $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ satisfying:

- $\langle x, y \rangle = \langle y, x \rangle$ (symmetry)
- $\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle$ (linearity)
- $\langle x, x \rangle \ge 0$ with equality if and only if $x = 0$ (positive-definiteness)

The Euclidean inner product is $\langle x, y \rangle = x^\top y$, which induces the Euclidean norm $\|x\| = \sqrt{\langle x, x \rangle}$. Two vectors $x$ and $y$ are orthogonal if $\langle x, y \rangle = 0$.

Inner products are essential for defining projections, measuring angles between vectors, and forming orthonormal bases. In optimization, the gradient direction is often interpreted as the direction of steepest ascent with respect to the chosen inner product.

## Rank, Nullspace, and Range

Let $A \in \mathbb{R}^{m \times n}$. The range of $A$ is

$$
\text{range}(A) = \{ y \in \mathbb{R}^m \mid y = Ax \text{ for some } x \in \mathbb{R}^n \},
$$

and the nullspace is

$$
\text{null}(A) = \{ x \in \mathbb{R}^n \mid Ax = 0 \}.
$$

The rank of $A$, denoted $\text{rank}(A)$, is the dimension of the range. The rank-nullity theorem states

$$
\text{rank}(A) + \dim(\text{null}(A)) = n.
$$

Proof sketch: Let $\{v_1, \dots, v_r\}$ be a basis of the range of $A$. Extend these vectors to a basis of $\mathbb{R}^n$ by adding vectors $\{u_1, \dots, u_{n-r}\}$ in the nullspace. This shows that the sum of the dimensions of the range and nullspace equals $n$.

In ML, rank determines feature redundancy: low-rank matrices indicate correlated or dependent features. In optimization, the nullspace gives directions along which the objective is invariant, which is crucial in constrained optimization.

## Orthonormal Bases and QR Decomposition

A set $\{q_1, \dots, q_n\}$ is orthonormal if $\langle q_i, q_j \rangle = \delta_{ij}$. Orthonormal bases simplify computations, improve numerical stability, and make projections straightforward:

$$
P_W(x) = \sum_{i=1}^n \langle x, q_i \rangle q_i
$$

is the projection of $x$ onto the subspace $W = \text{span}\{q_1, \dots, q_n\}$.

The Gram–Schmidt process converts any linearly independent set $\{v_1, \dots, v_n\}$ into an orthonormal set $\{q_1, \dots, q_n\}$ by iteratively subtracting projections onto previously computed vectors.

The QR decomposition of a full-rank matrix $A \in \mathbb{R}^{m \times n}$ is

$$
A = QR
$$

where $Q \in \mathbb{R}^{m \times n}$ has orthonormal columns and $R \in \mathbb{R}^{n \times n}$ is upper triangular. This allows the least-squares solution of $Ax = b$ to be computed as

$$
x = R^{-1} Q^\top b,
$$

which is numerically more stable than using $A^\top A x = A^\top b$ because $Q$ is orthonormal and does not amplify rounding errors.

In machine learning, QR decomposition is a useful linear algebra technique often applied in regression and data preprocessing. It is particularly valuable when solving overdetermined regression problems, where the number of data points (equations) exceeds the number of model parameters (unknowns). In such cases, there is no exact solution that satisfies all equations, so the goal is to find a least-squares solution that minimizes the overall error.

Conversely, an underdetermined system occurs when there are fewer equations than unknowns. In this case, there are infinitely many solutions, and additional constraints (like regularization) are needed to select a meaningful solution.

QR decomposition helps by factorizing the design matrix into an orthogonal matrix Q and an upper triangular matrix R. This factorization simplifies the solution of linear systems, allows computation of orthogonal features, and can improve numerical stability when preprocessing data for iterative optimization algorithms.

## Numerical Considerations

In machine learning, working with matrices requires careful attention to numerical stability. A matrix is said to be ill-conditioned if small changes in its entries or in the data can lead to large changes in the solution. This often happens when the columns of a matrix are nearly linearly dependent or have widely varying magnitudes. Ill-conditioning is typically quantified using the condition number: a high condition number indicates a highly sensitive (ill-conditioned) system, while a low condition number indicates a well-conditioned one.

For example, in a linear regression problem $Ax=b$, if 
A is ill-conditioned, small measurement errors in b or rounding errors in computation can cause large errors in the estimated parameters x. This makes the model unstable or unreliable, much like a tall, narrow tower that can topple from a tiny push.

To mitigate these issues, it is recommended to scale the columns of the matrix so that all features have comparable magnitudes. Additionally, for large or high-dimensional matrices, standard QR decomposition may be unstable. Using Modified Gram–Schmidt or Householder reflections can improve stability by maintaining orthogonality more accurately. Awareness of these numerical concerns is critical when working with high-dimensional datasets, as small errors can propagate and significantly degrade model performance.

## Summary and Connections to Optimization

- Vector spaces and subspaces define feasible sets and directions in optimization.  
- Inner products provide the geometry for gradients, projections, and orthogonality conditions.  
- Rank and nullspace determine uniqueness of solutions and degrees of freedom.  
- Orthonormal bases and QR decomposition enable stable solutions to linear subproblems.  
