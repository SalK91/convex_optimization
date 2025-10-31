# Chapter 2: Linear Algebra Foundations

Convex optimisation is geometric. To talk about convex sets, supporting hyperplanes, projections, and quadratic forms, we need linear algebra. This chapter reviews the specific linear algebra tools we will use throughout: vector spaces, inner products, norms, projections, eigenvalues, and positive semidefinite matrices.


## 2.1 Vector spaces, subspaces, and affine sets

A vector space over $\mathbb{R}$ is a set $V$ equipped with addition and scalar multiplication satisfying the usual axioms: closure, associativity, distributivity, etc. In this book we mostly work with $V = \mathbb{R}^n$.

A subspace $S \subseteq \mathbb{R}^n$ is a subset that:

1. contains $0$,
2. is closed under addition,
3. is closed under scalar multiplication.

For example, the set of all solutions to $Ax = 0$ is a subspace, called the nullspace or kernel of $A$.

An affine set is a translated subspace. A set $A$ is affine if for any $x,y \in A$ and any $\theta \in \mathbb{R}$,  
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

 
## 2.2 Linear combinations, span, basis, dimension

Given vectors $v_1,\dots,v_k$, any vector of the form
$$
\alpha_1 v_1 + \cdots + \alpha_k v_k
$$
is a linear combination. The set of all linear combinations is called the span:
$$
\mathrm{span}\{v_1,\dots,v_k\} = \left\{ \sum_{i=1}^k \alpha_i v_i : \alpha_i \in \mathbb{R} \right\}.
$$

A list of vectors is linearly independent if no nontrivial linear combination gives $0$. A basis of a subspace $S$ is a set of linearly independent vectors whose span is $S$. The number of vectors in a basis is the dimension of $S$.

Rank and nullity facts:

- The column space of $A$ is the span of its columns. Its dimension is $\mathrm{rank}(A)$.
- The nullspace of $A$ is $\{ x : Ax = 0 \}$.
- The rank-nullity theorem** states:
$$
\mathrm{rank}(A) + \mathrm{nullity}(A) = n,
$$
where $n$ is the number of columns of $A$.

In constrained optimisation, $\mathrm{rank}(A)$ encodes the “number of independent constraints”, and the nullspace encodes feasible directions that do not violate certain constraints.

> Column Space is a set of all possible "outputs" you can create by computing $Ax$. Its answers the question does a solution $x$ even exist for $Ax = b$. Solution exists if only if vector $b$ lives inside the column space $C(A)$. 

> Null Space is all sets of "inputs" $x$ that get "squashed to zero" by A i.e. All $x$ such that $Ax = 0$. It answers the question if a solution exisits, is it the only one.If the nullspace contains non-zero vectors ($\mathrm{nullity}(A) > 0$), there are infinitely many solutions.

> Multicollinearity (ML): If one feature in your data matrix $X$ is a combination of others (e.g., $feature_3 = 2 \times feature_1 + feature_2$), this creates a non-zero vector in the nullspace. This means there are infinitely many different weight vectors $w$ that produce the exact same predictions. The model is "unidentifiable." This is why $X^T X$ becomes non-invertible, and it’s the primary motivation for using regularization (like Ridge or Lasso) to pick one "good" solution from this infinite set.

> Feasible Directions (Optimization): As you noted, in a constrained problem like $Ax = b$, the nullspace is the set of all directions $d$ you can move from a feasible point $x$ without violating the constraints. If you are at $x$ and move to $x+d$ (where $d \in N(A)$), your constraints are still met: $A(x+d) = Ax + Ad = b + 0 = b$. This tells you your "space of free movement"

## 2.3 Inner products and orthogonality

An inner product on $\mathbb{R}^n$ is a map $\langle \cdot,\cdot\rangle : \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ such that for all $x,y,z$ and all scalars $\alpha$:

1. $\langle x,y \rangle = \langle y,x\rangle$ (symmetry),
2. $\langle x+y,z \rangle = \langle x,z \rangle + \langle y,z\rangle$ (linearity in first argument),
3. $\langle \alpha x, y\rangle = \alpha \langle x, y\rangle$,
4. $\langle x, x\rangle \ge 0$ with equality iff $x=0$ (positive definiteness).

In $\mathbb{R}^n$, the standard inner product is the dot product:
$$
\langle x,y \rangle = x^\top y = \sum_{i=1}^n x_i y_i~.
$$

The inner product induces:

- length (norm): $\|x\|_2 = \sqrt{\langle x,x\rangle}$,
- angle: 
$$
\cos \theta = \frac{\langle x,y\rangle}{\|x\|\|y\|}~.
$$

Two vectors are orthogonal if $\langle x,y\rangle = 0$. A set of vectors $\{v_i\}$ is orthonormal if each $\|v_i\| = 1$ and $\langle v_i, v_j\rangle = 0$ for $i\ne j$.

More generally, an inner product endows $V$ with a geometric structure, turning it into an inner product space (and if complete, a Hilbert space). Inner products allow us to talk about orthogonality (perpendicular vectors) and orthogonal projections, and to define the all-important concept of a gradient in optimization. 

 **Geometry from the inner product:** An inner product induces a norm $\|x\| = \sqrt{\langle x,x \rangle}$ and a notion of distance $d(x,y) = \|x-y\|$. It also defines angles: $\langle x,y \rangle = 0$ means $x$ and $y$ are orthogonal. Thus, inner products generalize the geometric concepts of lengths and angles to abstract vector spaces. Many results in Euclidean geometry (like the Pythagorean theorem and law of cosines) hold in any inner product space. For example, the parallelogram law holds: $\|x+y\|^2 + \|x-y\|^2 = 2\|x\|^2 + 2\|y\|^2$.  

**The Cauchy–Schwarz inequality:** For any $x,y \in \mathbb{R}^n$:
$$
|\langle x,y\rangle| \le \|x\|\|y\|~,
$$
with equality iff $x$ and $y$ are linearly dependent Geometrically, it means the absolute inner product is maximized when $x$ and $y$ point in the same or opposite direction. 

**Examples of inner products:**

- **Standard (Euclidean) inner product:** $\langle x,y\rangle = x^\top y = \sum_i x_i y_i$. This underlies most optimization algorithms on $\mathbb{R}^n$, where $\nabla f(x)$ is defined via this inner product (so that $\langle \nabla f(x), h\rangle$ gives the directional derivative in direction $h$).  

- **Weighted inner product:** $\langle x,y\rangle_W = x^\top W y$ for some symmetric positive-definite matrix $W$. Here $\|x\|_W = \sqrt{x^\top W x}$ is a weighted length. Such inner products appear in preconditioning: by choosing $W$ cleverly, one can measure distances in a way that accounts for scaling in the problem (e.g. the Mahalanobis distance uses $W = \Sigma^{-1}$ for covariance $\Sigma$).  

- **Function space inner product:** $\langle f, g \rangle = \int_a^b f(t)\,g(t)\,dt$. This turns the space of square-integrable functions on $[a,b]$ into an inner product space (a Hilbert space, $L^2[a,b]$). In machine learning, this is the basis for kernel Hilbert spaces, where one defines an inner product between functions to lift optimization into infinite-dimensional feature spaces.  

Any vector space with an inner product has an orthonormal basis (via the Gram–Schmidt process). Gram–Schmidt is fundamental in numerical algorithms to orthogonalize vectors and is used to derive the QR decomposition: any full-rank matrix $A \in \mathbb{R}^{m\times n}$ can be factored as $A = QR$ where $Q$ has orthonormal columns and $R$ is upper triangular. This factorization is widely used in least squares and optimization because it provides a stable way to solve $Ax=b$ and to analyze subspaces. For example, for an overdetermined system ($m>n$ i.e. more equations than unknowns), $Ax=b$ has a least-squares solution $x = R^{-1}(Q^\top b)$, and for underdetermined ($m<n$), $Ax=b$ has infinitely many solutions, among which one often chooses the minimal-norm solution using the orthonormal basis of the range. 


**Applications in optimization:** Inner product geometry is indispensable in convex optimization.  

- **Gradients:** The gradient $\nabla f(x)$ is defined as the vector satisfying $f(x+h)\approx f(x) + \langle \nabla f(x), h\rangle$. Thus the inner product induces the notion of steepest ascent/descent direction (steepest descent is in direction $-\nabla f(x)$ because it minimizes the inner product with the gradient). If we changed the inner product (using a matrix $W$), the notion of gradient would change accordingly (this idea is used in natural gradient methods).  

- **Orthogonal projections:** Many algorithms require projecting onto a constraint set. For linear constraints $Ax=b$ (an affine set), the projection formula uses the inner product to find the closest point in the affine set. Projections also underpin least squares problems (solution is projection of $b$ onto $\mathrm{range}(A)$) and quadratic programs (where each iteration might involve a projection).  

- **Orthonormal representations:** Orthonormal bases (like principal components) simplify optimization by diagonalizing quadratic forms or separating variables. For instance, in PCA we use an orthonormal basis (eigenvectors) to reduce dimensionality. In iterative algorithms, working in an orthonormal basis aligned with the problem (e.g. preconditioning) can accelerate convergence.  

- **Conditioning and Gram matrix:** The inner product concept leads to the Gram matrix $G_{ij} = \langle x_i, x_j\rangle$ for a set of vectors. In machine learning, the Gram matrix (or kernel matrix) encodes similarity of features and appears in the normal equations for least squares: $X^\top X$ is a Gram matrix whose eigenvalues tell us about problem conditioning. A well-conditioned Gram matrix (no tiny eigenvalues) means the problem is nicely scaled for gradient descent, whereas ill-conditioning (some nearly zero eigenvalues) means there are directions in weight space that are very flat, slowing convergence. Techniques like feature scaling or adding regularization (ridge regression) improve the Gram matrix’s condition number and thus algorithm performance.


## 2.4 Norms and distances

A function $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}$ is a norm if for all $x,y$ and scalar $\alpha$:

1. $\|x\| \ge 0$ and $\|x\| = 0 \iff x=0$,
2. $\|\alpha x\| = |\alpha|\|x\|$ (absolute homogeneity),
3. $\|x+y\| \le \|x\| + \|y\|$ (triangle inequality).


If the vector space has an inner product, the norm $\|x\| = \sqrt{\langle x,x\rangle}$ is called the Euclidean norm (or 2-norm). But many other norms exist, each defining a different geometry.  
Common examples on $\mathbb{R}^n$:  

- $\ell_2$ norm (Euclidean): $\|x\|_2 = \sqrt{\sum_i x_i^2}$, the usual length in space.  

- $\ell_1$ norm: $\|x\|_1 = \sum_i |x_i|$, measuring taxicab distance. In $\mathbb{R}^2$, its unit ball is a diamond.  

- $\ell_\infty$ norm: $\|x\|_\infty = \max_i |x_i|$, measuring the largest coordinate magnitude. Its unit ball in $\mathbb{R}^2$ is a square.  

- General $\ell_p$ norm: $\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$ for $p\ge1$. This interpolates between $\ell_1$ and $\ell_2$, and approaches $\ell_\infty$ as $p\to\infty$. All $\ell_p$ norms are convex and satisfy the norm axioms.  

Every norm induces a metric (distance) $d(x,y) = |x-y|$ on the space. Norms thus define the shape of “balls” (sets ${x: |x|\le \text{constant}}$) and how we measure closeness. The choice of norm can significantly influence an optimization algorithm’s behavior: it affects what steps are considered small, which directions are easy to move in, and how convergence is assessed.


![Alt text](images/norms.png){: style="float:right; margin-right:15px; width:400px;"}

**Unit-ball geometry:** The shape of the unit ball ${x: |x| \le 1}$ reveals how a norm treats different directions. For example, the $\ell_2$ unit ball in $\mathbb{R}^2$ is a perfect circle, treating all directions uniformly, whereas the $\ell_1$ unit ball is a diamond with corners along the axes, indicating that $\ell_1$ treats the coordinate axes as special (those are “cheaper” directions since the ball extends further along axes, touching them at $(\pm1,0)$ and $(0,\pm1)$). The $\ell_\infty$ unit ball is a square aligned with axes, suggesting it allows more combined motion in coordinates as long as no single coordinate exceeds the limit. These shapes are illustrated below: we see the red diamond ($\ell_1$), green circle ($\ell_2$), and blue square ($\ell_\infty$) in $\mathbb{R}^2$ . The geometry of the unit ball matters whenever we regularize or constrain solutions by a norm. For instance, using an $\ell_1$ norm ball as a constraint or regularizer encourages solutions on the corners (sparse solutions), while an $\ell_2$ ball encourages more evenly-distributed changes. An $\ell_\infty$ constraint limits the maximum absolute value of any component, leading to solutions that avoid any single large entry.


**Dual norms:** Each norm $\|\cdot\|$ has a dual norm $\|\cdot\|_*$ defined by
$$
\|y\|_* = \sup_{\|x\|\le 1} x^\top y~.
$$
For example, the dual of $\ell_1$ is $\ell_\infty$, and the dual of $\ell_2$ is itself.

> Imagine the vector $x$ lives inside the original norm ball ($\|x\| \le 1$). The term $x^\top y$ is the dot product, which measures the alignment between $x$ and $y$. The dual norm $\|y\|_*$ is the maximum possible value you can get by taking the dot product of $y$ with any vector $x$ that fits inside the original norm ball.If the dual norm $\|y\|_*$ is large, it means $y$ is strongly aligned with a direction $x$ that is "small" (size $\le 1$) according to the original norm.If the dual norm is small, $y$ must be poorly aligned with all vectors $x$ in the ball.

**Norms in optimization algorithms:** Different norms define different algorithmic behaviors. For example, gradient descent typically uses the Euclidean norm for step sizes and convergence analysis, but coordinate descent methods implicitly use $\ell_\infty$ (since one coordinate move at a time is like a step in $\ell_\infty$ unit ball). Mirror descent methods use non-Euclidean norms and their duals to get better performance on certain problems (e.g. using $\ell_1$ norm for sparse problems). The norm also figures in complexity bounds: an algorithm’s convergence rate may depend on the diameter of the feasible set in the chosen norm, $D = \max_{\text{feasible}}|x - x^*|$. For instance, in subgradient methods, having a smaller $\ell_2$ diameter or $\ell_1$ diameter can improve bounds. Moreover, when constraints are given by norms (like $|x|_1 \le t$), projections and proximal operators with respect to that norm become subroutines in algorithms.

In summary, norms provide the metric backbone of optimization. They tell us how to measure progress ($|x_k - x^*|$), how to constrain solutions ($|x| \le R$), and how to bound errors. The choice of norm can induce sparsity, robustness, or other desired structure in solutions, and mastering norms and their geometry is key to understanding advanced optimization techniques.

## 2.5 Eigenvalues, eigenvectors, and positive semidefinite matrices

If $A \in \mathbb{R}^{n\times n}$ is linear, a nonzero $v$ is an eigenvector with eigenvalue $\lambda$ if

$$
Av = \lambda v~.
$$

When $A$ is symmetric ($A = A^\top$), it has:

- real eigenvalues,
- an orthonormal eigenbasis,
- a spectral decomposition

$$
A = Q \Lambda Q^\top,
$$
where $Q$ is orthonormal and $\Lambda$ is diagonal.

This is the spectral decomposition. Geometrically, a symmetric matrix acts as a scaling along $n$ orthogonal principal directions (its eigenvectors), stretching or flipping by factors given by $\lambda_i$.

> When dealing specifically with square matrices and quadratic forms (like Hessians of twice-differentiable functions), eigenvalues become central. They describe how a symmetric matrix scales vectors in different directions. Many convex optimization conditions involve requiring a matrix (Hessian or constraint curvature matrix) to be positive semidefinite, which is an eigenvalue condition.

> In optimization, the Hessian matrix of a multivariate function $f(x)$ is symmetric. Its eigenvalues $\lambda_i(\nabla^2 f(x))$ tell us the curvature along principal axes. If all eigenvalues are positive at a point, the function curves up in all directions (a local minimum if gradient is zero); if any eigenvalue is negative, there’s a direction of negative curvature (a saddle or maximum). So checking eigenvalues of Hessian is a way to test convexity/concavity locally.

**Positive semidefinite matrices:** A symmetric matrix $Q$ is positive semidefinite (PSD) if

$$
x^\top Q x \ge 0 \quad \text{for all } x~.
$$

If $x^\top Q x > 0$ for all $x\ne 0$, then $Q$ is positive definite (PD).

Why this matters: if $f(x) = \tfrac{1}{2} x^\top Q x + c^\top x + d$, then

$$
\nabla^2 f(x) = Q~.
$$

So $f$ is convex iff $Q$ is PSD. Quadratic objectives with PSD Hessians are convex; with indefinite Hessians, they are not (Boyd and Vandenberghe, 2004). This is the algebraic test for convexity of quadratic forms.


**Implications of definiteness:** If $A \succ 0$, the quadratic function $x^T A x$ is strictly convex and has a unique minimizer at $x=0$. If $A \succeq 0$, $x^T A x$ is convex but could be flat in some directions (if some $\lambda_i = 0$, those eigenvectors lie in the nullspace and the form is constant along them). In optimization, PD Hessian $\nabla^2 f(x) \succ 0$ means $f$ has a unique local (and global, if domain convex) minimum at that $x$ (since the second-order condition for optimality is satisfied strictly). PD constraint matrices in quadratic programs ensure nice properties like Slater’s condition for strong duality.

**Condition number and convergence:** For iterative methods on convex quadratics $f(x) = \frac{1}{2}x^T Q x - b^T x$, the eigenvalues of $Q$ dictate convergence speed. Gradient descent’s error after $k$ steps satisfies roughly $|x_k - x^*| \le (\frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}})^k |x_0 - x^*|$ (for normalized step). So the ratio $\frac{\lambda_{\max}}{\lambda_{\min}} = \kappa(Q)$ appears: closer to 1 (well-conditioned) means rapid convergence; large ratio (ill-conditioned) means slow, zigzagging progress. Newton’s method uses Hessian inverse, effectively rescaling by eigenvalues to 1, so its performance is invariant to $\kappa$ (locally). This explains why second-order methods shine on ill-conditioned problems: they “whiten” the curvature by dividing by eigenvalues.

**Optimization interpretation of eigenvectors:** The eigenvectors of $\nabla^2 f(x^*)$ at optimum indicate principal axes of the local quadratic approximation. Directions with small eigenvalues are flat directions where the function changes slowly (possibly requiring LARGE steps unless Newton’s method is used). Directions with large eigenvalues are steep, potentially requiring small step sizes to maintain stability if using gradient descent. Preconditioning or change of variables often aims to transform the problem so that in new coordinates the Hessian is closer to the identity (all eigenvalues ~1). For constrained problems, the Hessian of the Lagrangian (the KKT matrix) being PSD relates to second-order optimality conditions.

## 2.6 Orthogonal projections and least squares

Let $S$ be a subspace of $\mathbb{R}^n$. The orthogonal projection of a vector $b$ onto $S$ is the unique vector $p \in S$ minimising $\|b - p\|_2$. Geometrically, $p$ is the closest point in $S$ to $b$.

If $S = \mathrm{span}\{a_1,\dots,a_k\}$ and $A = [a_1~\cdots~a_k]$, then projecting $b$ onto $S$ is equivalent to solving the least-squares problem

$$
\min_x \|Ax - b\|_2^2~.
$$
The solution $x^*$ satisfies the normal equations
$$
A^\top A x^* = A^\top b~.
$$

This is our first real convex optimisation problem:

- the objective $\|Ax-b\|_2^2$ is convex,
- there are no constraints,
- we can solve it in closed form.

## 2.7 Advanced Concepts


**Operator norm:** Given a matrix (linear map) $A: \mathbb{R}^n \to \mathbb{R}^m$ and given a choice of vector norms on input and output, one can define the induced operator norm. If we use $|\cdot|_p$ on $\mathbb{R}^n$ and $|\cdot|_q$ on $\mathbb{R}^m$, the operator norm is

$$
\|A\|_{p \to q}
= \sup_{x \ne 0} \frac{\|Ax\|_q}{\|x\|_p}
= \sup_{\|x\|_p \le 1} \|Ax\|_q
$$


This gives the maximum factor by which $A$ can stretch a vector (measuring $x$ in norm $p$ and $Ax$ in norm $q$).pecial cases are common: with $p = q = 2$, $|A|_{2 \to 2}$ (often just written $|A|_2$) is the spectral norm, which equals the largest singular value of $A$ (more on singular values below).
If $p = q = 1$, $|A|_{1 \to 1}$ is the maximum absolute column sum of $A$.
If $p = q = \infty$, $|A|{\infty \to \infty}$ is the maximum absolute row sum.

Operator norms tell us the worst-case amplification of signals by $A$. In gradient descent on $f(x) = \tfrac{1}{2} x^\top A x - b^\top x$ (a quadratic form), the step size must be $\le \tfrac{2}{|A|_2}$ for convergence; here $|A|_2 = \lambda_{\max}(A)$ if $A$ is symmetric (it’s related to Hessian eigenvalues, Chapter 5). In general, controlling $|A|$ controls stability: if $|A| < 1$, the map brings vectors closer (contraction mapping), important in fixed-point algorithms.

**Singular Value Decomposition (SVD):** Any matrix $A \in \mathbb{R}^{m\times n}$ can be factored as

$$
A = U \Sigma V^\top
$$


where $U \in \mathbb{R}^{m\times m}$ and $V \in \mathbb{R}^{n\times n}$ are orthogonal matrices (their columns are orthonormal bases of $\mathbb{R}^m$ and $\mathbb{R}^n$, respectively), and $\Sigma$ is an $m\times n$ diagonal matrix with nonnegative entries $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ on the diagonal. The $\sigma_i$ are the singular values of $A$. Geometrically, $A$ sends the unit ball in $\mathbb{R}^n$ to an ellipsoid in $\mathbb{R}^m$ whose principal semi-axes lengths are the singular values and directions are the columns of $V$ (mapped to columns of $U$). The largest singular value $\sigma_{\max} = |A|_2$ is the spectral norm. The smallest (if $n \le m$, $\sigma{\min}$ of those $n$) indicates how $A$ contracts the least – if $\sigma_{\min} = 0$, $A$ is rank-deficient.

The SVD is a fundamental tool for analyzing linear maps in optimization: it reveals the condition number $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ (when $\sigma_{\min}>0$), which measures how stretched the map is in one direction versus another. High condition number means ill-conditioning: some directions in $x$-space hardly change $Ax$ (flat curvature), making it hard for algorithms to progress uniformly. Low condition number means $A$ is close to an orthogonal scaling, which is ideal. SVD is also used for dimensionality reduction: truncating small singular values gives the best low-rank approximation of $A$ (Eckart–Young theorem), widely used in PCA and compressive sensing. In convex optimization, many second-order methods or constraint eliminations use eigen or singular values to simplify problems.

**Low-rank structure:** The rank of $A$ equals the number of nonzero singular values. If $A$ has rank $r \ll \min(n,m)$, it means $A$ effectively operates in a low-dimensional subspace. This often can be exploited: the data or constraints have some latent low-dimensional structure. Many convex optimization techniques (like nuclear norm minimization) aim to produce low-rank solutions by leveraging singular values. Conversely, if an optimization problem’s data matrix $A$ is low-rank, one can often compress it (via SVD) to speed up computations or reduce variables.

**Operator norm in optimization:** Operator norms also guide step sizes and preconditioning. As noted, for a quadratic problem $f(x) = \frac{1}{2}x^TQx - b^Tx$, the Hessian is $Q$ and gradient descent converges if $\alpha < 2/\lambda_{\max}(Q)$. Preconditioning aims to transform $Q$ into one with a smaller condition number by multiplying by some $P$ (like using $P^{-1}Q$) — effectively changing the norm in which we measure lengths, so the operator norm becomes smaller. In first-order methods for general convex $f$, the Lipschitz constant of $\nabla f$ (which often equals a spectral norm of a Hessian or Jacobian) determines convergence rates.

**Summary of spectral properties:**

- The **spectral norm** $|A|_2 = \sigma_{\max}(A)$ quantifies the largest stretching. It determines stability and step sizes.

- The smallest singular value $\sigma_{\min}$ (if $A$ is tall full-rank) tells if $A$ is invertible and how sensitive the inverse is. If $\sigma_{\min}$ is tiny, small changes in output cause huge changes in solving $Ax=b$.

- The **condition number** $\kappa = \sigma_{\max}/\sigma_{\min}$ is a figure of merit for algorithms: gradient descent iterations needed often scale with $\kappa$ (worse conditioning = slower). Regularization like adding $\mu I$ increases $\sigma_{\min}$, thereby reducing $\kappa$ and accelerating convergence (at the expense of bias).

- **Nuclear norm** (sum of singular values) and **spectral norm** often appear in optimization as convex surrogates for rank and as constraints to limit the operator’s impact.

In machine learning, one often whitens data (via SVD of the covariance) to improve conditioning, or uses truncated SVD to compress features. In sum, understanding singular values and operator norms equips us to diagnose and improve algorithmic performance for convex optimization problems.