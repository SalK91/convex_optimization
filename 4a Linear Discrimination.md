## 1. Linear Discrimination (LP Feasibility)

### Problem Setup
- **Variables**: $(a,b) \in \mathbb{R}^{n+1}$
- **Constraints**:
  - $a^T x_i - b \geq 1$
  - $a^T y_j - b \leq -1$

### Convexity
- Each constraint is affine in $(a,b)$.
- Affine inequalities define convex half-spaces.
- Intersection of half-spaces = convex polyhedron.
- No objective → pure **LP feasibility**.

**Type**: Convex LP feasibility problem.


## 2. Robust Linear Discrimination (Hard-Margin SVM)

### Problem
$$
\min \tfrac{1}{2}\|a\|_2^2 \quad 
\text{s.t. } a^T x_i - b \geq 1, \; a^T y_j - b \leq -1.
$$

### Convexity
- **Objective**: $\tfrac{1}{2}\|a\|_2^2$ is convex quadratic (strictly convex in $a$).
- **Constraints**: Affine $\Rightarrow$ convex feasible set.

**Type**: Convex quadratic program (QP).

 
## 3. Soft-Margin SVM

### Problem
$$
\min_{a,b,\xi} \; \tfrac{1}{2}\|a\|_2^2 + C \sum_i \xi_i
$$
subject to
$$
y_i(a^T z_i - b) \geq 1 - \xi_i, \quad \xi_i \geq 0.
$$

### Convexity
- **Objective**: Sum of convex quadratic ($\|a\|^2$) and linear ($\sum_i \xi_i$).
- **Constraints**: Affine in $(a,b,\xi)$.
- Feasible set = intersection of half-spaces (convex).

**Type**: Convex quadratic program (QP).

 
## 4. Hinge Loss Formulation

### Problem
$$
\min_{a,b} \; \tfrac{1}{2}\|a\|_2^2 + C \sum_i \max(0, 1 - y_i(a^T z_i - b)).
$$

### Convexity
- $\tfrac{1}{2}\|a\|_2^2$: convex quadratic.
- Inside hinge: $1 - y_i(a^T z_i - b)$ is affine.
- $\max(0, \text{affine})$ = convex function.
- Sum of convex functions = convex.

**Type**: Unconstrained convex optimization problem.

 
## 5. Dual SVM Problem

### Problem
$$
\max_{\alpha} \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j z_i^T z_j
$$
subject to
$$
\sum_i \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C.
$$

### Convexity
- Quadratic form has negative semidefinite Hessian (concave).
- Maximization of concave function over convex set → convex optimization.

**Type**: Convex quadratic program in dual variables.

 
## 6. Nonlinear Discrimination with Kernels

### Problem (Dual with Kernel)
$$
\max_{\alpha} \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(z_i,z_j)
$$
subject to
$$
\sum_i \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C.
$$

### Convexity
- If $K$ is positive semidefinite (Mercer kernel), quadratic form is convex.
- Maximization remains convex program.

**Type**: Convex QP with kernel matrix.

---

## 7. Quadratic Discrimination

### Problem
Classifier: $f(z) = z^T P z + q^T z + r$ with variables $(P,q,r)$.

Constraints:
- $x_i^T P x_i + q^T x_i + r \geq 1$
- $y_j^T P y_j + q^T y_j + r \leq -1$

### Convexity
- $x_i^T P x_i = \mathrm{Tr}(P x_i x_i^T)$, affine in $P$.
- Constraints affine in $(P,q,r)$.
- If additional constraint $P \succeq 0$, this is convex (semidefinite cone).

**Type**: LP feasibility or SDP (semidefinite program).

---

## 8. Polynomial Feature Maps

### Setup
- Map $z \mapsto F(z)$ with monomials up to degree $d$.
- Classifier: $f(z) = \theta^T F(z)$.

### Convexity
- Constraints: $\theta^T F(x_i) \geq 1$, affine in $\theta$.
- Margin maximization objective: $\|\theta\|^2$, convex quadratic.

**Type**: LP feasibility or convex QP.

---

## 9. Summary of Convex Structures

- **LP feasibility**: Linear separation.  
- **QP**: Hard-margin and soft-margin SVM.  
- **Unconstrained convex problem**: Hinge loss.  
- **Dual SVM**: Convex QP in dual variables.  
- **Kernel SVM**: Convex QP with PSD kernel.  
- **Quadratic/Polynomial discrimination**: LP or SDP, depending on constraints.  

---
