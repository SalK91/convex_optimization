# Mathematical Prerequisites

Before we can build convex optimisation tools, we need to review some core mathematical concepts from linear algebra and real analysis.

## Linear Algebra Essentials

- Vector spaces and norms: We work primarily in $\mathbb{R}^n$, the $n$-dimensional Euclidean space. The Euclidean norm is $\|x\|_2 = \sqrt{x^T x}$, but other norms, such as $\|x\|_1$ or $\|x\|_\infty$, are also important.

- Inner products: An inner product in $\mathbb{R}^n$ is $\langle x, y \rangle = x^T y$.

- Angle in inner product: The angle $\theta$ between two nonzero vectors $x, y \in \mathbb{R}^n$ is defined via  
  $\cos \theta = \dfrac{\langle x, y \rangle}{\|x\|_2 \, \|y\|_2}$.

- Affine sets: A set of the form  $\{x \in \mathbb{R}^n : Ax = b\}$, where $A$ is a matrix and $b$ is a vector. Affine sets are the natural generalisation of lines and planes.

- Positive semidefinite matrices (PSD): A symmetric matrix $Q$ is PSD if  $x^T Q x \geq 0$ for all $x$. Quadratic forms with PSD matrices define convex functions.

- Linear independence: A set of vectors $\{v_1, v_2, \dots, v_k\}$ is linearly independent if    $\alpha_1 v_1 + \alpha_2 v_2 + \cdots + \alpha_k v_k = 0$  
  implies $\alpha_1 = \alpha_2 = \cdots = \alpha_k = 0$.

- Outer product: Given vectors $u, v \in \mathbb{R}^n$, their outer product is   $u v^T$, which is an $n \times n$ matrix.  (Contrast with the inner product $u^T v$, which is a scalar.)

- Cauchy–Schwarz inequality: For any $x, y \in \mathbb{R}^n$, $|\langle x, y \rangle| \leq \|x\|_2 \, \|y\|_2$.  
  Equality holds iff $x$ and $y$ are linearly dependent.

- Projection onto a vector: The projection of $x$ onto $y$ (with $y \neq 0$) is  $\text{proj}_y(x) = \dfrac{\langle x, y \rangle}{\langle y, y \rangle} y$. This gives the component of $x$ in the direction of $y$.

- Subspace spanned by $k$ perpendicular (orthogonal) vectors: If $\{u_1, u_2, \dots, u_k\}$ are mutually perpendicular (orthogonal) and nonzero, then they form an orthogonal basis for their span. The subspace is $U = \text{span}\{u_1, u_2, \dots, u_k\}$,  and any $x \in U$ can be uniquely written as $x = \alpha_1 u_1 + \alpha_2 u_2 + \cdots + \alpha_k u_k$ with coefficients $\alpha_i = \dfrac{\langle x, u_i \rangle}{\langle u_i, u_i \rangle}$.


- Projection onto a subspace:  Let $U = \text{span}\{u_1, u_2, \dots, u_k\}$, where the vectors are linearly independent. If $U$ has orthonormal basis $\{q_1, q_2, \dots, q_k\}$, then the projection of $x$ onto $U$ is   $\text{proj}_U(x) = \sum_{i=1}^k \langle x, q_i \rangle q_i$. In matrix form, if $Q = [q_1 \; q_2 \; \dots \; q_k]$, then  
  $\text{proj}_U(x) = QQ^T x$.


- Projection onto a convex set:  Projecting two points on convex set is not expansive i.e. the distnce between projection of points is always less than equal to the original distance between the points.

#### Matrix Concepts

- Rank:  The rank of a matrix $A \in \mathbb{R}^{m \times n}$, denoted $\text{rank}(A)$, is the dimension of its column space (or equivalently, row space). It is the number of linearly independent columns (or rows). $\text{rank}(A) \leq \min(m,n)$.  

- Null space (kernel): The null space of $A \in \mathbb{R}^{m \times n}$ is / $
  \mathcal{N}(A) = \{x \in \mathbb{R}^n : Ax = 0\}.$ It contains all solutions to the homogeneous system $Ax=0$. Its dimension is called the *nullity* of $A$. By the *rank–nullity theorem* $\text{rank}(A) + \text{nullity}(A) = n.$

- Determinant:  For a square matrix $A \in \mathbb{R}^{n \times n}$, the determinant $\det(A)$ is a scalar with these properties:  
    - $\det(A) = 0 \iff A$ is singular (non-invertible).  
    - $\det(AB) = \det(A)\det(B)$.  
    - $\det(A^T) = \det(A)$.  
    - Geometric meaning: $|\det(A)|$ gives the volume scaling factor of the linear map $x \mapsto Ax$.

- Range (column space): The range of $A \in \mathbb{R}^{m \times n}$ is $\mathcal{R}(A) = \{Ax : x \in \mathbb{R}^n\} \subseteq \mathbb{R}^m.$ It is the span of the columns of $A$. $\dim(\mathcal{R}(A)) = \text{rank}(A)$.  


### Subspaces and Related Concepts

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

### Eigenvalues, Eigenvectors, and Symmetric Matrices

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

### Continuity and Lipschitz Continuity

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
 
### Gradient and Hessian

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


## Convex Sets, Combination, Hulls

### Convex Sets
- Convexity of sets: A set $C$ is convex if for any $x_1, x_2 \in C$ and $\theta \in [0,1]$, we have $\theta x_1 + (1-\theta) x_2 \in C$.
- Closed sets: A set is closed if it contains all its limit points. The closure of a set is the smallest closed set containing it.
- Extreme points: A point in a convex set is extreme if it cannot be expressed as a convex combination of two other distinct points in the set. For polyhedra, extreme points correspond to vertices.

 
### Convex Combination

A convex combination of $x_1, \dots, x_k$ is
$$
x = \sum_{i=1}^k \theta_i x_i, \quad \theta_i \geq 0, \quad \sum_{i=1}^k \theta_i = 1.
$$

This is simply a weighted average where weights are nonnegative and sum to 1.

### Convex Hull

The convex hull of a set $S$ is the collection of all convex combinations of points in $S$. It is the smallest convex set containing $S$.

```Geometric intuition: Imagine stretching a rubber band around the points; the enclosed region is the convex hull.```

### Cones
- A cone is a set $K$ such that if $x \in K$ and $\alpha \geq 0$, then $\alpha x \in K$.  
  In words: a cone is closed under nonnegative scalar multiplication.  
- The conic hull (or convex cone) of a set $S$ is the collection of all conic combinations of points in $S$:  

  $$
  \text{cone}(S) = \left\{ \sum_{i=1}^k \theta_i x_i \;\middle|\; x_i \in S, \; \theta_i \geq 0 \right\}.
  $$

- Cone is not a subpace but a subspace is always a cone.
- Since cone holds for all positive alphas including alphas that sum to 1. Hence a cone is a convex

### Polar Cone
- Given a cone $K \subseteq \mathbb{R}^n$, its polar cone is defined as:
  $$
  K^\circ = \{ y \in \mathbb{R}^n \mid y^T x \leq 0 \;\; \forall x \in K \}.
  $$
  i.e. dot product is less than zero.

- Intuitively, $K^\circ$ contains all vectors that form a non-acute (i.e., $\geq 90^\circ$) angle with **every** vector in $K$.




### Hyperplanes and Half-spaces
- A hyperplane is the solution set of $a^T x = b$.
- A half-space is one side of a hyperplane, defined as $a^T x \leq b$ or $a^T x \geq b$.
- These objects are convex and serve as building blocks in constraints.

Separation and Supporting Hyperplanes: One of the most powerful results in convex geometry is the separating hyperplane theorem: two disjoint convex sets can be separated by a hyperplane. For a convex set $C$ and a point $x \notin C$, there exists a hyperplane that separates $x$ from $C$. This underpins duality theory in optimisation. A supporting hyperplane touches a convex set at one or more points but does not cut through it.


## Convex Functions
A function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if its domain $\mathrm{dom}(f) \subseteq \mathbb{R}^n$ is convex and, for all $x_1, x_2 \in \mathrm{dom}(f)$ and all $\theta \in [0,1]$:

$$
f(\theta x_1 + (1-\theta)x_2) \leq \theta f(x_1) + (1-\theta) f(x_2)
$$

- Convex domain: For any $x_1, x_2 \in \mathrm{dom}(f)$, the line segment connecting them lies entirely in $\mathrm{dom}(f)$.  
- Intuition: The graph of $f$ lies below or on the straight line connecting any two points on it (“bowl-shaped”).

 
### First-order (gradient) condition: 
If $f$ is differentiable, $f$ is convex if and only if for all $x, y \in \mathrm{dom}(f)$:

  $$
  f(y) \ge f(x) + \nabla f(x)^T (y - x)
  $$

The tangent hyperplane at $x$ lies below the graph of $f$ at all points.

``` “Graph lies above all tangent planes.”  ```
 
### Second-order (Hessian) condition: 
If $f$ is twice differentiable, $f$ is convex if and only if its Hessian matrix $\nabla^2 f(x)$ is positive semidefinite for all $x \in \mathrm{dom}(f)$:

$$
\nabla^2 f(x) \succeq 0
$$

Positive semidefinite Hessian means the function curves upward or is flat in all directions, never downward.  

``` “Curvature is nonnegative in all directions.”  ```
  

### Examples of Convex Functions

1. Quadratic functions: $f(x) = \frac{1}{2} x^T Q x + b^T x + c$, where $Q \succeq 0$ (positive semidefinite).  
2. Norms: $\|x\|_p$ for $p \ge 1$.  
3. Exponential function: $f(x) = e^x$.  
4. Negative logarithm: $f(x) = -\log(x)$ on $x > 0$.  
5. Linear functions: $f(x) = a^T x + b$.  

### Strong Convexity
$f$ is $\mu$-strongly convex if  
$$
f(y) \ge f(x) + \nabla f(x)^\top (y-x) + \frac{\mu}{2}\|y-x\|^2
$$  

If twice differentiable, this is equivalent to
  $$
  \nabla^2 f(x) \succeq \mu I \quad\text{for all }x,
  $$


It guarantees a unique minimum and linear convergence for gradient-based methods. For example: Ridge/L2 regularization introduces strong convexity → more stable and faster optimization. 

``` The “bowl” is curved enough everywhere i.e. no flat regions. The steeper the curvature, the faster you slide to the bottom.```

The Extra term $\frac{\mu}{2} |y-x|^2$ ensures the function grows at least quadratically away from the tangent.

Example:

- $f(x) = x^2$ is 2-strongly convex with $\mu=2$.
- $f(x) = x^4$ is convex but not strongly convex near $x=0$ (too flat).
 
### Smoothness (L-smooth)
$$
\|\nabla f(x) - \nabla f(y)\| \le L \|x-y\|
$$  

Smoothness ensures gradients change gradually, preventing abrupt jumps. If $f$ is twice differentiable, a sufficient (and often used) equivalent condition is
  $$
  \nabla^2 f(x) \preceq L I \quad\text{for all }x,
  $$
  i.e. the largest eigenvalue of the Hessian is at most $L$. Intuition: $L$ is the maximum curvature / steepness of $f$.





It determines safe step sizes: $\alpha < 1/L$ for gradient descent. Neural networks are locally smooth; fixed-step gradient descent is justified.  
```Think of a surface that is not too steep anywhere. You can move along the slope without risk of overshooting dramatically.```

### Condition Number
$\kappa = L / \mu$  

Intuition: $\kappa$ measures the ratio between maximum and minimum curvature. 

  - High $\kappa$ → narrow, elongated valleys → gradient descent zig-zags and converges slowly.  
  - Low $\kappa$ → well-shaped valley → fast convergence.  
- ML relevance: Feature normalization, batch/layer normalization, and preconditioning improve conditioning.  
- Intuition: Condition number tells you how “twisted” the bowl is. A perfectly round bowl (low $\kappa$) is easy to slide down; a narrow steep valley (high $\kappa$) is tricky.

## Subgradients & Proximal Operators
Modern optimization in machine learning often deals with nonsmooth functions e.g., $L_1$ regularization, hinge loss in SVMs, indicator constraints. Gradients are not always defined at these nonsmooth points, so we need subgradients and proximal operators. For a differentiable convex function $f:\mathbb{R}^n \to \mathbb{R}$, the gradient $\nabla f(x)$ provides the slope for descent. But many convex functions are not differentiable everywhere:

- Absolute value: $f(x) = |x|$    (non-differentiable at $x=0$)  
- Hinge loss: $f(x) = \max(0, 1-x)$  
- $L_1$ norm: $f(x) = \|x\|_1 = \sum_i |x_i|$  

At kinks/corners, derivatives don’t exist.  

A vector $g \in \mathbb{R}^n$ is a subgradient of a convex function $f$ at point $x$ if:

$$f(y) \;\ge\; f(x) + g^\top (y-x), \quad \forall y \in \mathbb{R}^n$$

- Geometric meaning: $g$ defines a supporting hyperplane at $(x, f(x))$ that lies below the function everywhere.  
- The set of all subgradients at $x$ is called the subdifferential, written:
  $$
  \partial f(x) = \{ g \in \mathbb{R}^n \;|\; f(y) \ge f(x) + g^\top (y-x), \;\forall y \}.
  $$



### Example 1: Absolute value
Take $f(x) = |x|$.  

- If $x > 0$: $\nabla f(x) = 1$.  
- If $x < 0$: $\nabla f(x) = -1$.  
- If $x = 0$: derivative doesn’t exist. But  
  $$
  \partial f(0) = \{ g \in [-1, 1] \}.
  $$
Any slope between $-1$ and $1$ is a valid subgradient at the kink.
 Intuition: At $x=0$, instead of one tangent line, there’s a whole fan of supporting lines.



### Example 2: Hinge loss
$f(x) = \max(0, 1-x)$.  

- If $x < 1$: $\nabla f(x) = -1$.  
- If $x > 1$: $\nabla f(x) = 0$.  
- If $x = 1$:  
  $$
  \partial f(1) = [-1, 0].
  $$



### Why subgradients matter
- They generalize gradients to nonsmooth convex functions.  
- Subgradient descent update:
  $$
  x_{k+1} = x_k - \alpha_k g_k, \quad g_k \in \partial f(x_k).
  $$
- Convergence is guaranteed, though slower than gradient descent:
  - Smooth case: $O(1/k)$ rate  
  - Nonsmooth case: $O(1/\sqrt{k})$ rate  


### Proximal Operators
Nonsmooth penalties (like $L_1$ norm, indicator functions) appear frequently:  
- Lasso: $\min_x \tfrac{1}{2}\|Ax-b\|^2 + \lambda \|x\|_1$  (L1 norm is nonsmooth)
- SVM: hinge loss $\max(0, 1-y\langle w,x\rangle)$  
- Constraints: e.g., $x \in C$ for some convex set $C$

Plain gradient descent cannot directly handle the nonsmooth part.

The proximal operator of a function $g$ with step size $\alpha > 0$ is:

$$
\text{prox}_{\alpha g}(v) 
= \arg\min_x \Big( g(x) + \frac{1}{2\alpha}\|x-v\|^2 \Big).
$$

- Interpretation:  
  - Stay close to $v$ (the quadratic term)  
  - While reducing the penalty $g(x)$  

- Geometric meaning: A regularized projection of $v$ onto a region encouraged by $g$.  


### Example 1: $L_1$ norm (soft-thresholding)
Let $g(x) = \lambda \|x\|_1 = \lambda \sum_i |x_i|$.  
Then:

$$
\text{prox}_{\alpha g}(v)_i = 
\begin{cases}
v_i - \alpha\lambda, & v_i > \alpha \lambda \\
0, & |v_i| \le \alpha \lambda \\
v_i + \alpha\lambda, & v_i < -\alpha \lambda
\end{cases}
$$

This is the soft-thresholding operator:

- Shrinks small entries of $v$ to zero → sparsity.  
- Reduces magnitude of large entries but keeps their sign.  

👉 This is the key step in Lasso regression and compressed sensing.

---

### Example 2: Indicator function
Let $g(x) = I_C(x)$, where $I_C(x)=0$ if $x \in C$, and $\infty$ otherwise.  
Then:

$$
\text{prox}_{\alpha g}(v) = \Pi_C(v),
$$

the Euclidean projection of $v$ onto $C$.  

Example: if $C$ is the unit ball $\{x: \|x\|\le 1\}$, prox just normalizes $v$ if it’s outside.

---

### Example 3: Squared $\ell_2$ norm
If $g(x) = \frac{\lambda}{2}\|x\|^2$, then

$$
\text{prox}_{\alpha g}(v) = \frac{1}{1+\alpha\lambda} v.
$$

This is just a shrinkage toward the origin.

---

### Why proximal operators matter
They allow efficient algorithms for composite objectives:

$$
\min_x f(x) + g(x),
$$

where:
- $f$ is smooth (differentiable with Lipschitz gradient)  
- $g$ is convex but possibly nonsmooth  

Proximal gradient method (ISTA):
$$
x_{k+1} = \text{prox}_{\alpha g}\big(x_k - \alpha \nabla f(x_k)\big).
$$

This generalizes gradient descent by replacing the plain update with a proximal step that handles $g$.

- If $g=0$: reduces to gradient descent  
- If $f=0$: reduces to proximal operator (e.g. projection, shrinkage)  

---

## 3. Intuition Summary

- Subgradients:  
  - Generalized “slopes” for nonsmooth convex functions.  
  - At corners, we have a set of possible slopes (subdifferential).  
  - Enable subgradient descent with convergence guarantees.  

- Proximal operators:  
  - Generalized update steps for nonsmooth regularizers.  
  - Combine a gradient-like move with a “correction” that enforces structure (sparsity, constraints).  
  - Core of algorithms like ISTA, FISTA, ADMM.  

---

## 4. Big Picture in ML

- Subgradients: Let us train models with nonsmooth losses (SVM hinge loss, $L_1$).  
- Proximal operators: Let us efficiently solve regularized problems (Lasso, group sparsity, constrained optimization).  
- Intuition:  
  - Subgradient = "any slope that supports the function"  
  - Proximal = "soft move toward minimizing the nonsmooth part"  

---


### Subgradients & Proximal Operators
- Subgradient: $g$ is a subgradient if  
$$
f(y) \ge f(x)+g^\top(y-x), \quad \forall y
$$  
- Proximal operator:  
$$
\text{prox}_{\alpha g}(v) = \arg\min_x \Big(g(x) + \frac{1}{2\alpha}\|x-v\|^2\Big)
$$  
- Context: Needed for nonsmooth functions (e.g., L1-regularization, hinge loss).  
- ML relevance: SVM hinge loss, Lasso, sparse dictionary learning. Proximal methods handle shrinkage or projection efficiently.  
- Intuition:  
  - Subgradient: Like a tangent for a function that isn’t smooth—provides a direction to descend.  
  - Proximal operator: Think of it as a “soft step” toward minimizing a nonsmooth function, like gently nudging a point toward a feasible or sparse region.


## Convex Optimisation Problems

A convex optimisation problem has the form:

$$
\begin{aligned}
& \min_x \quad & f_0(x) \\
& \text{s.t.} \quad & f_i(x) \leq 0, \quad i=1, \dots, m \\
& & h_j(x) = 0, \quad j=1, \dots, p,
\end{aligned}
$$
where $f_0$ and $f_i$ are convex functions, and $h_j$ are affine. The feasible set is convex, and any local minimum is a global minimum.



## References

- S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.
- D. Bertsekas, *Nonlinear Programming*, Athena Scientific, 1999.
- A. Ben-Tal and A. Nemirovski, *Lectures on Modern Convex Optimization*, SIAM, 2001.
- Relevant research articles on log-concavity, geometric programming, and proximal methods.


