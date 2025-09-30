# Linear Algebra Prerequisites

- Vector spaces and norms: We work primarily in $\mathbb{R}^n$, the $n$-dimensional Euclidean space. The Euclidean norm is $\|x\|_2 = \sqrt{x^T x}$, but other norms, such as $\|x\|_1$ or $\|x\|_\infty$, are also important.

- Inner products: An inner product in $\mathbb{R}^n$ is $\langle x, y \rangle = x^T y$.

- Angle in inner product: The angle $\theta$ between two nonzero vectors $x, y \in \mathbb{R}^n$ is defined via  
  $\cos \theta = \dfrac{\langle x, y \rangle}{\|x\|_2 \, \|y\|_2}$.

- Affine sets: A set of the form  $\{x \in \mathbb{R}^n : Ax = b\}$, where $A$ is a matrix and $b$ is a vector. Affine sets are the natural generalisation of lines and planes.


- Linear independence: A set of vectors $\{v_1, v_2, \dots, v_k\}$ is linearly independent if    $\alpha_1 v_1 + \alpha_2 v_2 + \cdots + \alpha_k v_k = 0$  
  implies $\alpha_1 = \alpha_2 = \cdots = \alpha_k = 0$.

- Outer product: Given vectors $u, v \in \mathbb{R}^n$, their outer product is   $u v^T$, which is an $n \times n$ matrix.  (Contrast with the inner product $u^T v$, which is a scalar.)

- Cauchy–Schwarz inequality: For any $x, y \in \mathbb{R}^n$, $|\langle x, y \rangle| \leq \|x\|_2 \, \|y\|_2$.  
  Equality holds iff $x$ and $y$ are linearly dependent.

- Projection onto a vector: The projection of $x$ onto $y$ (with $y \neq 0$) is  $\text{proj}_y(x) = \dfrac{\langle x, y \rangle}{\langle y, y \rangle} y$. This gives the component of $x$ in the direction of $y$.

- Subspace spanned by $k$ perpendicular (orthogonal) vectors: If $\{u_1, u_2, \dots, u_k\}$ are mutually perpendicular (orthogonal) and nonzero, then they form an orthogonal basis for their span. The subspace is $U = \text{span}\{u_1, u_2, \dots, u_k\}$,  and any $x \in U$ can be uniquely written as $x = \alpha_1 u_1 + \alpha_2 u_2 + \cdots + \alpha_k u_k$ with coefficients $\alpha_i = \dfrac{\langle x, u_i \rangle}{\langle u_i, u_i \rangle}$.


- Projection onto a subspace:  Let $U = \text{span}\{u_1, u_2, \dots, u_k\}$, where the vectors are linearly independent. If $U$ has orthonormal basis $\{q_1, q_2, \dots, q_k\}$, then the projection of $x$ onto $U$ is   $\text{proj}_U(x) = \sum_{i=1}^k \langle x, q_i \rangle q_i$. In matrix form, if $Q = [q_1 \; q_2 \; \dots \; q_k]$, then  
  $\text{proj}_U(x) = QQ^T x$.


- Projection onto a convex set:  Projecting two points on convex set is not expansive i.e. the distance between projection of points is always less than equal to the original distance between the points.

## Matrix Concepts

- Rank:  The rank of a matrix $A \in \mathbb{R}^{m \times n}$, denoted $\text{rank}(A)$, is the dimension of its column space (or equivalently, row space). It is the number of linearly independent columns (or rows). $\text{rank}(A) \leq \min(m,n)$.  

- Null space (kernel): The null space of $A \in \mathbb{R}^{m \times n}$ is / $
  \mathcal{N}(A) = \{x \in \mathbb{R}^n : Ax = 0\}.$ It contains all solutions to the homogeneous system $Ax=0$. Its dimension is called the *nullity* of $A$. By the *rank–nullity theorem* $\text{rank}(A) + \text{nullity}(A) = n.$

- Determinant:  For a square matrix $A \in \mathbb{R}^{n \times n}$, the determinant $\det(A)$ is a scalar with these properties:  
    - $\det(A) = 0 \iff A$ is singular (non-invertible).  
    - $\det(AB) = \det(A)\det(B)$.  
    - $\det(A^T) = \det(A)$.  
    - Geometric meaning: $|\det(A)|$ gives the volume scaling factor of the linear map $x \mapsto Ax$.

- Range (column space): The range of $A \in \mathbb{R}^{m \times n}$ is $\mathcal{R}(A) = \{Ax : x \in \mathbb{R}^n\} \subseteq \mathbb{R}^m.$ It is the span of the columns of $A$. $\dim(\mathcal{R}(A)) = \text{rank}(A)$.  


## Subspaces and Related Concepts

- Subspace: A subset $U \subseteq \mathbb{R}^n$ is a subspace if it satisfies:  
    1. $0 \in U$ (contains the zero vector)  
    2. Closed under addition: $u, v \in U \implies u+v \in U$  
    3. Closed under scalar multiplication: $u \in U, \alpha \in \mathbb{R} \implies \alpha u \in U$  

    Examples: column space of a matrix, null space of a matrix, $\mathbb{R}^n$ itself.  

 
- Orthogonal complement: For a subspace $U \subseteq \mathbb{R}^n$, the orthogonal complement $U^\perp$ is  
  $$
  U^\perp = \{v \in \mathbb{R}^n : \langle v, u \rangle = 0 \;\text{for all}\; u \in U \}.
  $$  
    - $U^\perp$ is a subspace.  
    - $\dim(U) + \dim(U^\perp) = n$.  
    - $(U^\perp)^\perp = U$.  


- Affine subspace:  An affine subspace is a translation of a subspace. Formally,  
  $$
  A = \{x_0 + u : u \in U\}
  $$  
  where $U$ is a subspace and $x_0 \in \mathbb{R}^n$ is a fixed point.  

    - Examples: lines or planes that do not pass through the origin.  
    - If $x_0 = 0$, the affine subspace is just a subspace.

## Eigenvalues, Eigenvectors, and Symmetric Matrices

- Eigenvalue and Eigenvector: For a square matrix $A \in \mathbb{R}^{n \times n}$, a scalar $\lambda \in \mathbb{R}$ (or $\mathbb{C}$) is an eigenvalue if there exists a nonzero vector $v \in \mathbb{R}^n$ such that  
  $$
  A v = \lambda v.
  $$  
  The vector $v$ is called an eigenvector corresponding to $\lambda$.  
    - $v \neq 0$  
    - $\det(A - \lambda I) = 0$ gives the characteristic equation to find eigenvalues.  

  
- Symmetric matrix:  A matrix $A \in \mathbb{R}^{n \times n}$ is symmetric if  
  $$
  A = A^T.
  $$   
    1. All eigenvalues of a symmetric matrix are real.  Eigenvectors of symmetric matrices are orthogonal. 
    2. There exists an orthonormal set of eigenvectors that spans $\mathbb{R}^n$.  
    3. $A$ can be diagonalized as  
      $$
      A = Q \Lambda Q^T,
      $$  
      where $Q$ is an orthogonal matrix of eigenvectors ($Q^T Q = I$) and $\Lambda$ is a diagonal matrix of eigenvalues.  
  

- Positive Semidefinite Matrices
  A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is positive semidefinite (PSD) if  
  $$
  x^T A x \geq 0 \quad \text{for all } x \in \mathbb{R}^n.
  $$
    1. $A$ is symmetric: $A = A^T$.  
    2. All eigenvalues of $A$ are non-negative: $\lambda_i \ge 0$.  
    3. If $x^T A x > 0$ for all $x \neq 0$, then $A$ is positive definite (PD).  
    4. $A$ can be diagonalized as  
      $$
      A = Q \Lambda Q^T,
      $$  
      where $Q$ is orthogonal and $\Lambda$ is a diagonal matrix with $\lambda_i \ge 0$.  
    5. $x^T A x \geq 0$ if A is PSD. 
    6. for any matrix A, $A^TA$ and $AA^T$ are PSD.

## Continuity and Lipschitz Continuity

- Continuity: A function $f : \mathbb{R}^n \to \mathbb{R}^m$ is continuous at $x_0$ if  
  $$
  \lim_{x \to x_0} f(x) = f(x_0).
  $$  
  Equivalently, for every $\varepsilon > 0$ there exists $\delta > 0$ such that  
  $$
  \|x - x_0\| < \delta \;\; \implies \;\; \|f(x) - f(x_0)\| < \varepsilon.
  $$  
  If this holds for all $x_0$, then $f$ is continuous on its domain.  

 
- Lipschitz continuity:  A function $f : \mathbb{R}^n \to \mathbb{R}^m$ is Lipschitz continuous if there exists a constant $L \geq 0$ such that  
  $$
  \|f(x) - f(y)\| \leq L \|x - y\| \quad \text{for all } x, y.
  $$  

    - The smallest such $L$ is called the Lipschitz constant.  
    - Lipschitz continuity $\implies$ continuity (but not vice versa).  
    - If $f$ is differentiable and $\|\nabla f(x)\| \leq L$ for all $x$, then $f$ is Lipschitz continuous with constant $L$.  
 
## Gradient and Hessian

- Gradient:  Let $f : \mathbb{R}^n \to \mathbb{R}$ be differentiable.  
  The gradient of $f$ at $x$ is the vector of partial derivatives:  
  $$
  \nabla f(x) =
  \begin{bmatrix}
  \frac{\partial f}{\partial x_1}(x) \\
  \frac{\partial f}{\partial x_2}(x) \\
  \vdots \\
  \frac{\partial f}{\partial x_n}(x)
  \end{bmatrix} \in \mathbb{R}^n.
  $$  

    - It points in the direction of steepest ascent of $f$.  
    - The magnitude $\|\nabla f(x)\|$ gives the rate of increase in that direction.  

- Hessian:  If $f$ is twice differentiable, the Hessian matrix at $x$ is the matrix of second-order partial derivatives:  
  $$
  \nabla^2 f(x) =
  \begin{bmatrix}
  \frac{\partial^2 f}{\partial x_1^2}(x) & \frac{\partial^2 f}{\partial x_1 \partial x_2}(x) & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n}(x) \\
  \frac{\partial^2 f}{\partial x_2 \partial x_1}(x) & \frac{\partial^2 f}{\partial x_2^2}(x) & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n}(x) \\
  \vdots & \vdots & \ddots & \vdots \\
  \frac{\partial^2 f}{\partial x_n \partial x_1}(x) & \frac{\partial^2 f}{\partial x_n \partial x_2}(x) & \cdots & \frac{\partial^2 f}{\partial x_n^2}(x)
  \end{bmatrix}.
  $$  

    - The Hessian is symmetric if $f$ is twice continuously differentiable.  
    - It describes the curvature of $f$:  
        - $\nabla^2 f(x) \succeq 0$ (PSD) $\implies f$ is locally convex.  
        - $\nabla^2 f(x) \succ 0$ (PD) $\implies f$ is locally strictly convex.  




