# 2. Inner Product Spaces and Orthogonality

Once we have vector spaces, the next ingredient is a way to measure angles and lengths. An inner product on a real vector space $V$ is a function $\langle \cdot,\cdot \rangle: V \times V \to \mathbb{R}$ that is symmetric, bilinear, and positive-definite. For any $x,y,z \in V$ and scalar $\alpha$:  

- Symmetry: $\langle x, y \rangle = \langle y, x \rangle$
- Linearity: $\langle \alpha x + y, z \rangle = \alpha \langle x, z \rangle + \langle y, z \rangle$.  
- Positive-definiteness: $\langle x, x \rangle \ge 0$, with equality only if $x = 0$.  

The canonical example is the Euclidean inner product on $\mathbb{R}^n$: $\langle x, y \rangle = x^\top y = \sum_{i=1}^n x_i y_i$. This yields the familiar length $\|x\|_2 = \sqrt{x^\top x}$ and angle via $\cos\theta = \frac{\langle x,y\rangle}{\|x\|\|y\|}$. More generally, an inner product endows $V$ with a geometric structure, turning it into an inner product space (and if complete, a Hilbert space). Inner products allow us to talk about orthogonality (perpendicular vectors) and orthogonal projections, and to define the all-important concept of a gradient in optimization. 

**Geometry from the inner product:** An inner product induces a norm $\|x\| = \sqrt{\langle x,x \rangle}$ and a notion of distance $d(x,y) = \|x-y\|$. It also defines angles: $\langle x,y \rangle = 0$ means $x$ and $y$ are orthogonal. Thus, inner products generalize the geometric concepts of lengths and angles to abstract vector spaces. Many results in Euclidean geometry (like the Pythagorean theorem and law of cosines) hold in any inner product space. For example, the parallelogram law holds: $\|x+y\|^2 + \|x-y\|^2 = 2\|x\|^2 + 2\|y\|^2$.  

**Cauchy–Schwarz inequality:** Perhaps the most important inequality in an inner product space is Cauchy–Schwarz:


$$\|\langle x, y \rangle\| \le \|x\| \|y\|$$

with equality if and only if $x$ and $y$ are linearly dependent. Geometrically, it means the absolute inner product is maximized when $x$ and $y$ point in the same or opposite direction. Cauchy–Schwarz is ubiquitous in optimization, providing bounds on cosine similarity, error estimates, and is the basis for many other inequalities (like Hölder’s inequality in Chapter 11). For example, it ensures that projecting one vector onto another (or onto a subspace) cannot increase its length.  

**Examples of inner products:**

- **Standard (Euclidean) inner product:** $\langle x,y\rangle = x^\top y = \sum_i x_i y_i$. This underlies most optimization algorithms on $\mathbb{R}^n$, where $\nabla f(x)$ is defined via this inner product (so that $\langle \nabla f(x), h\rangle$ gives the directional derivative in direction $h$).  

- **Weighted inner product:** $\langle x,y\rangle_W = x^\top W y$ for some symmetric positive-definite matrix $W$. Here $\|x\|_W = \sqrt{x^\top W x}$ is a weighted length. Such inner products appear in preconditioning: by choosing $W$ cleverly, one can measure distances in a way that accounts for scaling in the problem (e.g. the Mahalanobis distance uses $W = \Sigma^{-1}$ for covariance $\Sigma$).  

- **Function space inner product:** $\langle f, g \rangle = \int_a^b f(t)\,g(t)\,dt$. This turns the space of square-integrable functions on $[a,b]$ into an inner product space (a Hilbert space, $L^2[a,b]$). In machine learning, this is the basis for kernel Hilbert spaces, where one defines an inner product between functions to lift optimization into infinite-dimensional feature spaces.  

**Orthogonality and orthonormal bases:** Two vectors $x,y$ are orthogonal if $\langle x,y\rangle=0$. A set of vectors $\{q_1,\dots,q_k\}$ is orthonormal if each $q_i$ has unit length and they are mutually orthogonal: $\langle q_i, q_j\rangle = \delta_{ij}$ (1 if $i=j$, else 0). Orthonormal vectors are by definition linearly independent. If an orthonormal set spans a subspace $W$, it is the orthonormal basis of that subspace. Orthonormal bases greatly simplify computations because coordinates decouple. Any vector $x$ can be projected onto $W$ easily: the projection onto $W$ is

$$P_W(x)=\sum_{i=1}^k \langle x,q_i\rangle q_i$$

which is the closest vector in $W$ to $x$. Importantly, $x - P_W(x)$ is orthogonal to $W$, and $P_W$ is the linear map that is idempotent ($P_W(P_W(x))=P_W(x)$). In optimization, projected gradient descent uses this operation to remain feasible: if we need $x_{k+1}$ to lie in a subspace $W$, we take a step $x_k - \alpha \nabla f(x_k)$ and then project back to $W$. 

Any vector space with an inner product has an orthonormal basis (via the Gram–Schmidt process). Gram–Schmidt is fundamental in numerical algorithms to orthogonalize vectors and is used to derive the QR decomposition: any full-rank matrix $A \in \mathbb{R}^{m\times n}$ can be factored as $A = QR$ where $Q$ has orthonormal columns and $R$ is upper triangular. This factorization is widely used in least squares and optimization because it provides a stable way to solve $Ax=b$ and to analyze subspaces. For example, for an overdetermined system ($m>n$ i.e. more equations than unknowns), $Ax=b$ has a least-squares solution $x = R^{-1}(Q^\top b)$, and for underdetermined ($m<n$), $Ax=b$ has infinitely many solutions, among which one often chooses the minimal-norm solution using the orthonormal basis of the range. 


**Applications in optimization:** Inner product geometry is indispensable in convex optimization.  

- **Gradients:** The gradient $\nabla f(x)$ is defined as the vector satisfying $f(x+h)\approx f(x) + \langle \nabla f(x), h\rangle$. Thus the inner product induces the notion of steepest ascent/descent direction (steepest descent is in direction $-\nabla f(x)$ because it minimizes the inner product with the gradient). If we changed the inner product (using a matrix $W$), the notion of gradient would change accordingly (this idea is used in natural gradient methods).  

- **Orthogonal projections:** Many algorithms require projecting onto a constraint set. For linear constraints $Ax=b$ (an affine set), the projection formula uses the inner product to find the closest point in the affine set. Projections also underpin least squares problems (solution is projection of $b$ onto $\mathrm{range}(A)$) and quadratic programs (where each iteration might involve a projection).  

- **Orthonormal representations:** Orthonormal bases (like principal components) simplify optimization by diagonalizing quadratic forms or separating variables. For instance, in PCA we use an orthonormal basis (eigenvectors) to reduce dimensionality. In iterative algorithms, working in an orthonormal basis aligned with the problem (e.g. preconditioning) can accelerate convergence.  

- **Conditioning and Gram matrix:** The inner product concept leads to the Gram matrix $G_{ij} = \langle x_i, x_j\rangle$ for a set of vectors. In machine learning, the Gram matrix (or kernel matrix) encodes similarity of features and appears in the normal equations for least squares: $X^\top X$ is a Gram matrix whose eigenvalues tell us about problem conditioning. A well-conditioned Gram matrix (no tiny eigenvalues) means the problem is nicely scaled for gradient descent, whereas ill-conditioning (some nearly zero eigenvalues) means there are directions in weight space that are very flat, slowing convergence. Techniques like feature scaling or adding regularization (ridge regression) improve the Gram matrix’s condition number and thus algorithm performance.
