# Eigenvalues, Symmetric Matrices, and PSD/PD Matrices

Eigenvalues and positive semidefinite/definite matrices play a central role in convex optimization. They provide insight into curvature, conditioning, step scaling, and uniqueness of solutions. This chapter develops the theory of eigenpairs, diagonalization of symmetric matrices, and properties of PSD/PD matrices, with direct applications to optimization algorithms and machine learning.

## Eigenpairs and Diagonalization of Symmetric Matrices

Let $A \in \mathbb{R}^{n \times n}$. A scalar $\lambda \in \mathbb{R}$ is an eigenvalue of $A$ if there exists a nonzero vector $v \in \mathbb{R}^n$ such that

$$
A v = \lambda v.
$$

The vector $v$ is called an eigenvector associated with $\lambda$.

For symmetric matrices $A = A^\top$, several important properties hold:

- All eigenvalues are real: $\lambda_i \in \mathbb{R}$  
- Eigenvectors corresponding to distinct eigenvalues are orthogonal  
- $A$ can be diagonalized by an orthogonal matrix $Q$:  

$$
A = Q \Lambda Q^\top,
$$

where $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$ and $Q^\top Q = I$.

Geometric intuition: Symmetric matrices act as stretching or compressing along orthogonal directions, where the eigenvectors indicate the principal directions and eigenvalues the scaling factors.

In optimization, the eigenvalues of the Hessian $\nabla^2 f(x)$ determine curvature along different directions. Positive eigenvalues indicate convexity along that direction, while negative eigenvalues indicate concavity.

## Positive Semidefinite and Positive Definite Matrices

A symmetric matrix $A$ is positive semidefinite (PSD) if

$$
x^\top A x \ge 0 \quad \forall x \in \mathbb{R}^n,
$$

and **positive definite (PD)** if

$$
x^\top A x > 0 \quad \forall x \neq 0.
$$

Equivalently:

- PSD: all eigenvalues $\lambda_i \ge 0$  
- PD: all eigenvalues $\lambda_i > 0$

Properties:

- $A \succeq 0 \implies$ quadratic form $x^\top A x$ is convex  
- $A \succ 0 \implies$ quadratic form is strictly convex with a unique minimizer

In convex optimization, PD Hessians guarantee unique minimizers and enable Newton-type methods with reliable step scaling. PSD matrices appear in quadratic programming and covariance matrices in statistics and machine learning.

## Spectral Connections to Optimization

1. Step size selection: For gradient descent on $f(x) = \frac{1}{2} x^\top A x - b^\top x$, the largest eigenvalue $\lambda_{\max}(A)$ determines the maximum safe step size $\alpha \le 2/\lambda_{\max}(A)$.  
2. Conditioning: The condition number $\kappa(A) = \lambda_{\max}/\lambda_{\min}$ controls the convergence rate of gradient descent and other iterative methods.  
3. Curvature analysis: Eigenvectors of the Hessian indicate directions of fastest or slowest curvature, guiding preconditioning and variable rescaling.  
4. Low-rank approximations: PSD matrices can be truncated along small eigenvalues to reduce dimensionality in ML applications such as PCA.

 