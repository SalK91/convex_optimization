When dealing specifically with square matrices and quadratic forms (like Hessians of twice-differentiable functions), eigenvalues become central. They describe how a symmetric matrix scales vectors in different directions. Many convex optimization conditions involve requiring a matrix (Hessian or constraint curvature matrix) to be positive semidefinite, which is an eigenvalue condition.

**Eigenvalues and eigenvectors:** For a square matrix $A \in \mathbb{R}^{n\times n}$, an eigenvector $v \neq 0$ and scalar eigenvalue $\lambda$ satisfy $A v = \lambda v$. In general $A$ might be defective or complex eigenvalues might appear, but for symmetric matrices $A = A^T$, a beautiful theory simplifies things: all eigenvalues are real, and there is an orthonormal basis of $\mathbb{R}^n$ consisting of eigenvectors of $A$. Thus we can write

$$
A = Q \Lambda Q^\top
$$

with $Q$ orthogonal and $\Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_n)$. This is the spectral decomposition. Geometrically, a symmetric matrix acts as a scaling along $n$ orthogonal principal directions (its eigenvectors), stretching or flipping by factors given by $\lambda_i$.

In optimization, the Hessian matrix of a multivariate function $f(x)$ is symmetric. Its eigenvalues $\lambda_i(\nabla^2 f(x))$ tell us the curvature along principal axes. If all eigenvalues are positive at a point, the function curves up in all directions (a local minimum if gradient is zero); if any eigenvalue is negative, there’s a direction of negative curvature (a saddle or maximum). So checking eigenvalues of Hessian is a way to test convexity/concavity locally.

**Positive semidefinite matrices:** A symmetric matrix $A$ is positive semidefinite (PSD), written $A \succeq 0$, if $x^T A x \ge 0$ for all $x$. It’s positive definite (PD), $A \succ 0$, if $x^T A x > 0$ for all $x\neq0$. In terms of eigenvalues, $A \succeq 0$ iff all $\lambda_i \ge 0$, and $A \succ 0$ iff all $\lambda_i > 0$. PSD matrices generalize the notion of a nonnegative scalar. They define convex quadratic forms: $x^T A x$ is a convex function of $x$ if $A$ is PSD. In fact, a twice-differentiable function $f$ is convex on a region if and only if its Hessian is PSD at every point in that region. So Hessian eigenvalues $\lambda_{\min}(\nabla^2 f) \ge 0$ for all points is a certificate of convexity.

Some important properties and examples:

- Identity matrix $I$ has all eigenvalues 1, is PD. Scaling $I$ by $\alpha>0$ yields eigenvalues $\alpha$, still PD.

- Diagonal matrices have eigenvalues equal to their diagonal entries (so easy to check PSD by seeing if all diagonal are nonnegative, provided matrix is diagonal in some basis).

- Covariance matrices (like $X^T X$ or the Gram matrix $G$) are PSD: for any $z$, $z^T X^T X z = |Xz|^2 \ge 0$. In ML, covariance being PSD corresponds to variance being nonnegative.

- Laplacian matrices in graphs are PSD, which connects to convex quadratic energies.

**Implications of definiteness:** If $A \succ 0$, the quadratic function $x^T A x$ is strictly convex and has a unique minimizer at $x=0$. If $A \succeq 0$, $x^T A x$ is convex but could be flat in some directions (if some $\lambda_i = 0$, those eigenvectors lie in the nullspace and the form is constant along them). In optimization, PD Hessian $\nabla^2 f(x) \succ 0$ means $f$ has a unique local (and global, if domain convex) minimum at that $x$ (since the second-order condition for optimality is satisfied strictly). PD constraint matrices in quadratic programs ensure nice properties like Slater’s condition for strong duality.

**Condition number and convergence:** For iterative methods on convex quadratics $f(x) = \frac{1}{2}x^T Q x - b^T x$, the eigenvalues of $Q$ dictate convergence speed. Gradient descent’s error after $k$ steps satisfies roughly $|x_k - x^*| \le (\frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}})^k |x_0 - x^*|$ (for normalized step). So the ratio $\frac{\lambda_{\max}}{\lambda_{\min}} = \kappa(Q)$ appears: closer to 1 (well-conditioned) means rapid convergence; large ratio (ill-conditioned) means slow, zigzagging progress. Newton’s method uses Hessian inverse, effectively rescaling by eigenvalues to 1, so its performance is invariant to $\kappa$ (locally). This explains why second-order methods shine on ill-conditioned problems: they “whiten” the curvature by dividing by eigenvalues.

**Optimization interpretation of eigenvectors:** The eigenvectors of $\nabla^2 f(x^*)$ at optimum indicate principal axes of the local quadratic approximation. Directions with small eigenvalues are flat directions where the function changes slowly (possibly requiring LARGE steps unless Newton’s method is used). Directions with large eigenvalues are steep, potentially requiring small step sizes to maintain stability if using gradient descent. Preconditioning or change of variables often aims to transform the problem so that in new coordinates the Hessian is closer to the identity (all eigenvalues ~1). For constrained problems, the Hessian of the Lagrangian (the KKT matrix) being PSD relates to second-order optimality conditions.

In summary, eigenvalues and definiteness connect algebra to geometry: they tell us if a quadratic bowl opens upward (convex) or downward, how elongated or skewed that bowl is, and whether an optimization problem has a unique solution. Much of convex optimization theory, especially duality and KKT conditions, assumes convexity which can often be verified via positive semidefiniteness of certain matrices (Hessians or constraint Hessians in second-order cone programs, etc.). We often constrain matrices to be PSD in semidefinite programming, which is essentially convex optimization in the space of eigenvalues.