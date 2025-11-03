# Matrix Factorizations in Convex Optimization

Matrix factorizations are the numerical backbone of convex optimization algorithms.  
Whether solving Newton steps, least-squares problems, or KKT systems, every convex solver eventually reduces to solving a linear system of the form:
\[
H x = b,
\]
where \(H\) might represent a Hessian, normal equations matrix, or a KKT system.

Directly computing \(H^{-1}\) is never done in practice — it’s unstable, slow, and wasteful.  
Instead, solvers use matrix factorizations, which decompose \(H\) into structured components that make systems easier to solve, more numerically stable, and computationally efficient.

---

## 1. Why Factorizations Matter

In convex optimization, almost every second-order or constrained algorithm requires repeatedly solving linear systems:
- Newton’s method: \(\nabla^2 f(x_k) \Delta x = -\nabla f(x_k)\)
- Interior-point method: KKT systems coupling primal and dual variables
- Least-squares: \(A^T A x = A^T b\)
- Dual updates: systems involving \(A Q^{-1} A^T\)

The key challenges:
- \(H\) can be large, sparse, ill-conditioned, or indefinite.
- Numerical stability and efficiency depend entirely on choosing the right factorization.

---

## 2. Overview of Major Factorizations

| Factorization | Applies To | Form | Use Case | Notes |
|----------------|-------------|------|-----------|-------|
| LU | Any square (nonsingular) matrix | \(A = L U\) | General linear systems | Most basic; doesn’t exploit symmetry |
| QR | Any matrix (square or rectangular) | \(A = Q R\) | Least squares, orthogonal projections | Numerically stable but expensive |
| Cholesky | Symmetric positive definite (SPD) | \(A = L L^T\) | QP Hessians, least-squares normal equations | Fastest, most stable for SPD |
| LDLᵀ | Symmetric indefinite | \(A = L D L^T\) | KKT systems, equality-constrained QPs | Generalization of Cholesky |
| Eigen | Symmetric (or Hermitian) | \(A = Q \Lambda Q^T\) | Spectral analysis, SDP decomposition | Diagonalizes curvature |
| SVD | Any matrix | \(A = U \Sigma V^T\) | Rank analysis, pseudoinverse, low-rank solvers | Most accurate but expensive |

Each decomposition expresses the matrix in a structured way, revealing properties such as stability, conditioning, and rank.

---

## 3. LU Factorization — *The general-purpose workhorse*

Form:  
\[
A = L U,
\]
where \(L\) is lower triangular, \(U\) is upper triangular.

Use: General square linear systems, \(A x = b\).

Why in convex optimization:  
Used when the problem matrix is not symmetric or not guaranteed positive definite (e.g., linear equality constraints in augmented systems).

Pros
- Works for any nonsingular matrix.  
- Straightforward to implement.

Cons
- Does not exploit symmetry — double the work of symmetric factorizations.  
- Can be unstable without pivoting.

Solver context:  
LP simplex tableau, nonsymmetric KKT systems, or linearized subproblems.

---

## 4. QR Factorization — *Orthogonal and stable*

Form:  
\[
A = Q R, \quad Q^T Q = I, \quad R \text{ upper triangular.}
\]

Use: Solving least-squares problems
\[
\min_x \|A x - b\|_2^2.
\]

Instead of forming normal equations \(A^T A x = A^T b\) (which squares the condition number), we solve:
\[
R x = Q^T b,
\]
which is far more numerically stable.

Why in convex optimization:
- Used in least-squares or fitting subproblems inside proximal or ADMM steps.  
- Essential when \(A\) is rectangular (overdetermined systems).

Pros
- Very stable (orthogonal transformations preserve norms).  
- No explicit squaring of conditioning.

Cons
- ~2× slower than Cholesky.  
- Higher memory footprint.

---

## 5. Cholesky Factorization — *Fastest for positive definite systems*

Form:  
\[
A = L L^T, \quad L \text{ lower triangular.}
\]

Applies to: Symmetric positive definite matrices (SPD).  
This includes Hessians of convex functions and normal equations in least-squares.

Why it’s central to convex optimization:
- Convex problems have positive semidefinite curvature → the Hessian or KKT submatrix is SPD.  
- Cholesky is the fastest and most stable factorization for SPD matrices.

Typical uses:
- Newton steps in unconstrained convex problems.
- Solving normal equations \(A^T A x = A^T b\).
- Covariance and regularization matrices in QP and ridge regression.

Pros
- Exploits symmetry → half the computation of LU.  
- Stable (no pivoting needed).  
- Simple forward–backward substitution.

Cons
- Fails if matrix not SPD (e.g., indefinite KKT systems).

---

## 6. LDLᵀ Factorization — *Handles indefinite symmetric systems*

Form:  
\[
A = L D L^T,
\]
where \(D\) is block diagonal (1×1 or 2×2 blocks), \(L\) lower triangular.

Why it’s important:
Many convex solvers (interior-point, equality-constrained QPs, SDPs) produce symmetric but indefinite KKT systems:
\[
\begin{bmatrix}
Q & A^T \\
A & 0
\end{bmatrix}.
\]
Cholesky fails here (negative pivots), but LDLᵀ succeeds.

Advantages
- Exploits symmetry.  
- Works with indefinite matrices.  
- Enables sparse reordering for large-scale problems.

Used in:  
Interior-point solvers, primal–dual methods, equality-constrained Newton systems.

---

## 7. Eigenvalue Decomposition — *Revealing curvature*

Form:  
\[
A = Q \Lambda Q^T, \quad Q^T Q = I.
\]

Use: For symmetric (or Hermitian) matrices, especially Hessians or covariance matrices.

Why it matters in convex optimization:
- Checks convexity (\(A \succeq 0 \iff \Lambda \ge 0\)).  
- Used in semidefinite programming (SDP): constraints like \(X \succeq 0\) rely on eigenvalue decompositions.  
- Provides spectral insight for preconditioning and regularization.

Pros
- Fully diagonalizes the problem (reveals curvature).  
- Conceptually clean.

Cons
- Expensive (\(O(n^3)\)).  
- Not used iteratively in large solvers — mainly for analysis or small SDPs.

---

## 8. Singular Value Decomposition (SVD) — *The most general and stable*

Form:  
\[
A = U \Sigma V^T,
\]
where \(U, V\) are orthogonal, \(\Sigma\) is diagonal with nonnegative singular values.

Why it’s powerful:
- Works for *any* matrix (rectangular, rank-deficient, ill-conditioned).  
- Gives rank, conditioning, and pseudoinverse:
  \[
  A^+ = V \Sigma^{-1} U^T.
  \]

Used in:
- Low-rank convex optimization.  
- Nuclear-norm minimization (matrix completion).  
- Preconditioning and numerical conditioning analysis.

Pros
- Extremely stable, handles degeneracy gracefully.  
- Reveals numerical rank and conditioning.

Cons
- Most expensive factorization (\(O(m n^2)\)).  
- Used selectively, not every iteration.

---

## 9. Which Factorization to Use in Convex Optimization

| Context | System Type | Recommended Factorization | Reason |
|----------|--------------|----------------------------|--------|
| Unconstrained convex problem | SPD Hessian | Cholesky | Fastest, stable, symmetric |
| Equality-constrained QP | Symmetric indefinite KKT | LDLᵀ | Handles indefinite blocks |
| General LP/QP Newton system | Sparse symmetric indefinite | LDLᵀ with reordering | Exploits sparsity, preserves symmetry |
| Least-squares / regression | Rectangular \(A\) | QR | Stable for overdetermined systems |
| Poorly conditioned / rank-deficient | Any | SVD | Safest, reveals rank and conditioning |
| Non-symmetric systems | General \(A\) | LU | Works for all invertible matrices |

---

## 10. Why Convex Solvers Favor Cholesky and LDLᵀ

Convex problems typically produce symmetric matrices — the Hessian and KKT systems come from derivatives of convex functions or constraints.

- Cholesky is ideal for positive definite systems (smooth convex Hessians, normal equations).  
- LDLᵀ handles indefinite but symmetric systems (common in KKT matrices).  
- Both preserve symmetry, halve computation, and allow sparse reordering — crucial for large-scale optimization.

In short:
> Convex optimization solvers rely on Cholesky and LDLᵀ because convexity guarantees symmetry, and these factorizations exploit that structure for speed and stability.

---

### 💡 Summary: Key Takeaways

- Matrix factorizations transform hard linear algebra problems into easy triangular ones.  
- Cholesky and LDLᵀ are the backbone of modern convex solvers — efficient, symmetric, and numerically stable.  
- QR is preferred for least-squares; SVD for robustness; LU for generality.  
- Choosing the right factorization balances speed, stability, and structure.

---

### References
- Boyd & Vandenberghe, *Convex Optimization* (Ch. 4, Appendix C)  
- Nocedal & Wright, *Numerical Optimization* (Ch. 2–3)  
- Trefethen & Bau, *Numerical Linear Algebra* (Ch. 20–25)

