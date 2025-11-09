# Numerical Linear Algebra for Convex Optimization

Numerical linear algebra is the computational foundation of convex optimization.
Every modern optimization algorithm — from Newton’s method to interior-point or proximal algorithms — ultimately requires solving a structured linear system:
$$
H x = b,
$$
where $H$ may represent a Hessian, a normal equations matrix, or a KKT (Karush–Kuhn–Tucker) system.

In practice, we never compute $H^{-1}$ directly. Instead, we exploit matrix factorizations and structure to solve such systems efficiently and stably.

 
## 1. Why Linear Algebra Matters in Convex Optimization

At each iteration of a convex optimization algorithm, we must solve one or more linear systems:

* Newton’s method:
  $$
  \nabla^2 f(x_k), \\ \Delta x = -\nabla f(x_k)
  $$

* Interior-point methods (KKT systems):

$$
\begin{bmatrix}
H & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\[3pt] \Delta \lambda
\end{bmatrix}
=
\begin{bmatrix}
-r_d \\[3pt] -r_p
\end{bmatrix}
$$


* Least-squares problems: $A^T A x = A^T b$

Solving these systems dominates computation time. The stability, speed, and scalability of a convex solver depend on the numerical linear algebra techniques used.

## 2. The Matrix Factorization Toolbox

Matrix factorizations decompose a matrix into simpler pieces, exposing its structure.
They enable efficient triangular solves instead of direct inversion.

| Factorization | Applies To                  | Form                | Common Use                        | Key Notes                       |
| - |  | - |  | - |
| LU        | Any nonsingular matrix      | $A = L U$           | General linear systems            | Requires pivoting for stability |
| QR        | Any (rectangular) matrix    | $A = Q R$           | Least-squares                     | Orthogonal, stable              |
| Cholesky  | Symmetric positive definite | $A = L L^T$         | SPD systems, normal equations     | Fastest for SPD                 |
| $LDL^T$      | Symmetric indefinite        | $A = L D L^T$       | KKT systems                       | Handles indefiniteness          |
| Eigen     | Symmetric/Hermitian         | $A = Q \Lambda Q^T$ | Curvature, convexity checks       | Diagonalizes $A$                |
| SVD       | Any matrix                  | $A = U \Sigma V^T$  | Rank, conditioning, pseudoinverse | Most stable, expensive          |

Each factorization corresponds to a *numerically preferred strategy* for certain classes of problems.

 
## 3. LU Factorization — *The General-Purpose Workhorse*

Form:
$$
A = P L U
$$
where $P$ is a permutation matrix ensuring stability.

* Used for: General linear systems, nonsymmetric matrices.
* Cost: $\approx \tfrac{2}{3}n^3$ (dense).
* Stability: Requires partial pivoting ($PA=LU$) to avoid numerical blow-up.

Example use case:

* Solving KKT systems in linear programming (LP simplex tableau).
* Small dense systems with no symmetry or SPD property.

Note: For symmetric systems, LU wastes work (duplicate storage and computation). Prefer Cholesky or $LDL^T$.

 
## 4. QR Factorization — *Orthogonal and Stable*

Form:
$$
A = Q R, \quad Q^T Q = I, \ R \text{ upper triangular.}
$$

* Used for: Least-squares problems
  $$
  \min_x |A x - b|_2^2.
  $$
  Instead of forming normal equations ($A^T A x = A^T b$), we solve:
  $$
  R x = Q^T b.
  $$
* Stability: Orthogonal transformations preserve the 2-norm, making QR backward stable.

Example use cases:

* Linear regression via least squares.
* ADMM and proximal steps with overdetermined systems.
* Orthogonal projections in signal processing.

Variants:

* Householder QR: numerically robust, used in LAPACK.
* Rank-revealing QR (RRQR): detects rank deficiency robustly.

 
## 5. Cholesky Factorization — *Fastest for SPD Systems*

Form:
$$
A = L L^T, \quad L \text{ lower triangular.}
$$
Applicable when $A$ is symmetric positive definite (SPD) — common in convex problems.

Why it’s central:
Convexity ensures $A \succeq 0$.
For strictly convex problems, $A \succ 0$ and Cholesky is the most efficient and stable method.

Cost: $\tfrac{1}{3}n^3$ operations — half of LU.

Example use cases:

* Newton’s method on unconstrained convex functions.
* Solving normal equations $A^T A x = A^T b$.
* QP subproblems and ridge regression.

Implementation detail:
No pivoting needed for SPD matrices. Sparse versions (e.g., CHOLMOD) use fill-reducing orderings (AMD, METIS).

 
## 6. LDLᵀ Factorization — *For Indefinite Symmetric Systems*

Form:
$$
A = L D L^T,
$$
where $D$ is block diagonal (1×1 or 2×2 blocks), and $L$ is unit lower triangular.

Used when $A$ is symmetric but not SPD (e.g., KKT systems).

Example use cases:

* Interior-point methods for QPs and SDPs:
  $$
  \begin{bmatrix}
  Q & A^T \\ A & 0
  \end{bmatrix}
  \begin{bmatrix}
  \Delta x \\ \Delta \lambda
  \end{bmatrix} =
  \begin{bmatrix}
  r_1 \\ r_2
  \end{bmatrix}.
  $$

* Equality-constrained least-squares.
* Sparse symmetric indefinite systems in primal-dual algorithms.

Algorithmic note:
Uses Bunch–Kaufman pivoting to maintain numerical stability.
In practice, LDLᵀ is used with sparse reordering and partial elimination.


## 7. Block Systems and the Schur Complement

Many KKT or structured systems naturally appear in block form:
$$
\begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2
\end{bmatrix} =
\begin{bmatrix}
b_1 \\ b_2
\end{bmatrix}.
$$

Assuming $A_{11}$ is invertible:

1. Eliminate $x_1$:
   $$
   x_1 = A_{11}^{-1}(b_1 - A_{12}x_2)
   $$
2. Substitute into the second block:
   $$
   (A_{22} - A_{21}A_{11}^{-1}A_{12})x_2 = b_2 - A_{21}A_{11}^{-1}b_1
   $$

The matrix
$$
S = A_{22} - A_{21}A_{11}^{-1}A_{12}
$$
is the Schur complement of $A_{11}$ in $A$.


### Schur Complement in Optimization

* Reduces high-dimensional KKT systems to smaller systems in dual variables.
* Preserves symmetry and often positive definiteness.
* Foundation of block elimination and reduced Hessian methods.

Example use cases:

* Interior-point Newton systems (eliminate $\Delta x$ to get a system in $\Delta \lambda$).
* Partial elimination in sequential quadratic programming (SQP).
* Covariance conditioning and Gaussian marginalization.

Numerical caution: Never form $A_{11}^{-1}$ explicitly — use triangular solves via Cholesky or LU.

 
## 8. Block Elimination Algorithm

Given a nonsingular $A_{11}$:

1. Compute $A_{11}^{-1}A_{12}$ and $A_{11}^{-1}b_1$ by solving triangular systems.
2. Form $S = A_{22} - A_{21}A_{11}^{-1}A_{12}$, $\tilde{b} = b_2 - A_{21}A_{11}^{-1}b_1$.
3. Solve $Sx_2 = \tilde{b}$.
4. Recover $x_1 = A_{11}^{-1}(b_1 - A_{12}x_2)$.

Used in block Gaussian elimination, especially when the system has clear hierarchical structure.

Example use case:

* Partitioned least-squares with fixed and variable parameters.
* Constrained optimization where some variables can be analytically eliminated.

 
## 9. Structured Plus Low-Rank Matrices

Suppose we need to solve:
$$
(A + BC)x = b,
$$
where:

* $A \in \mathbb{R}^{n \times n}$ is structured or easily invertible (e.g., diagonal or sparse),
* $B \in \mathbb{R}^{n \times p}$, $C \in \mathbb{R}^{p \times n}$ are low rank.

This situation arises when updating an existing system with a small modification.

 
### Block Reformulation

Introduce $y = Cx$, yielding:

$$
\begin{bmatrix}
A & B \\ C & -I
\end{bmatrix}
\begin{bmatrix}
x \\ y
\end{bmatrix}
=

\begin{bmatrix}
b \\ 0
\end{bmatrix}.
$$

Block elimination gives:
$$
(I + C A^{-1} B)y = C A^{-1} b,
\quad
x = A^{-1}(b - By).
$$

 

### Matrix Inversion Lemma (Woodbury Identity)

If $A$ and $A + BC$ are nonsingular:
$$
(A + BC)^{-1} = A^{-1} - A^{-1}B(I + C A^{-1}B)^{-1}C A^{-1}.
$$

Example use cases:

* Kalman filters / Bayesian updates: covariance updates with rank-1 corrections.
* Ridge regression / kernel methods: low-rank updates to $(X^T X + \lambda I)^{-1}$.
* Active-set QP: efficiently reusing factorization when constraints are added or removed.

Numerical note: Avoid explicit inversion; use solves with $A$ and small dense matrices.

 

## 10. Conditioning, Stability, and Sparsity

### Conditioning

* Condition number: $\kappa(A) = |A||A^{-1}|$ measures sensitivity to perturbations.
* High $\kappa(A)$ ⇒ round-off errors amplified ⇒ ill-conditioning.
* Regularization (adding $\lambda I$) improves numerical stability.

### Stability

* Orthogonal transformations (QR, SVD) are backward stable.
* LU needs partial pivoting.
* LDLᵀ needs symmetric pivoting (Bunch–Kaufman).
* Cholesky is stable for SPD matrices.

### Sparsity and Fill-In

* Large convex solvers exploit sparse Cholesky / LDLᵀ.
* Fill-reducing orderings (AMD, METIS) minimize new nonzeros.
* Symbolic factorization determines the pattern before numeric factorization.


## 11. Iterative Solvers and Preconditioning

For large-scale problems (e.g., machine learning, PDE-constrained optimization), direct factorizations are infeasible.

### Common Iterative Methods

| Method               | For                  | Description                                         |
| -- | -- |  |
| CG               | SPD systems          | Uses matrix–vector products; converges in ≤ n steps |
| MINRES / SYMMLQ  | Symmetric indefinite | Handles KKT and saddle-point systems                |
| GMRES / BiCGSTAB | Nonsymmetric         | General-purpose Krylov solvers                      |

### Preconditioning

Preconditioners $M \approx A^{-1}$ improve convergence:

* Jacobi (diagonal): $M = \text{diag}(A)^{-1}$
* Incomplete Cholesky (IC) or Incomplete LU (ILU): approximate factorization
* Block preconditioners: use Schur complement approximations for KKT systems

Example use case:

* Solving large sparse Newton systems in logistic regression or LASSO via CG with IC preconditioner.
* Interior-point methods for large LPs using MINRES with block-diagonal preconditioning.

 
## 12. Eigenvalue and SVD Decompositions

### Eigenvalue Decomposition

$$
A = Q \Lambda Q^T, \quad Q^T Q = I.
$$
Reveals curvature, stability, and definiteness:

* Convexity ⇔ $\Lambda \ge 0$.
* Used in semidefinite programming (SDP) and spectral analysis.

### Singular Value Decomposition (SVD)

$$
A = U \Sigma V^T,
$$
with $\Sigma = \text{diag}(\sigma_i) \ge 0$.

Applications:

* Rank and condition number estimation ($\kappa(A) = \sigma_{\max}/\sigma_{\min}$).
* Low-rank approximation ($A_k = U_k \Sigma_k V_k^T$).
* Pseudoinverse: $A^+ = V \Sigma^{-1} U^T$.
* Convex relaxations: nuclear-norm minimization (matrix completion).

 

## 13. Computational Complexity Summary

| Factorization | Dense Cost               | Notes                                           |
| - |  | -- |
| LU            | $\frac{2}{3}n^3$         | Needs pivoting                                  |
| Cholesky      | $\frac{1}{3}n^3$         | Fastest for SPD                                 |
| QR            | $\approx \frac{2}{3}n^3$ | Stable, more memory                             |
| LDLᵀ          | $\approx \frac{2}{3}n^3$ | For indefinite                                  |
| SVD           | $\approx \frac{4}{3}n^3$ | Most accurate                                   |
| CG / MINRES   | Variable                 | Depends on condition number and preconditioning |

Sparse systems reduce cost to roughly $O(n^{1.5})$–$O(n^2)$ depending on fill-in.

 
## 14. Example Applications Overview

| Problem Type              | Typical Matrix           | Solver / Factorization | Example                              |
| - |  | - |  |
| Unconstrained Newton step | SPD Hessian              | Cholesky               | Convex quadratic, ridge regression   |
| Equality-constrained QP   | Symmetric indefinite KKT | LDLᵀ                   | Interior-point QP solver             |
| Overdetermined LS         | Rectangular $A$          | QR                     | Linear regression, ADMM subproblem   |
| KKT block system          | Block-symmetric          | Schur complement       | Primal-dual method                   |
| Low-rank correction       | $A + U U^T$              | Woodbury               | Kalman filter, online update         |
| Rank-deficient system     | Any                      | SVD                    | Matrix completion, regularization    |
| Large-scale Hessian       | SPD                      | CG + preconditioner    | Logistic regression, large ML models |

