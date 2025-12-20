# Chapter 17: Canonical Problems in Convex Optimization

Convex optimization encompasses a wide range of problem classes.  Despite their diversity, many real-world models reduce to a few canonical forms — each with characteristic geometry, structure, and algorithms.

 
## Hierarchy of Canonical Problems

Convex programs form a nested hierarchy:

$$
\text{LP} \subseteq \text{QP} \subseteq \text{SOCP} \subseteq \text{SDP}.
$$

Each inclusion represents an extension of representational power — from linear to quadratic, to conic, and finally to semidefinite constraints.  
Separately, Geometric Programs (GPs) and Maximum Likelihood Estimators (MLEs) form additional convex families after suitable transformations.

| Class | Canonical Form | Key Condition | Typical Algorithms | ML / Applied Examples |
|--------|----------------|----------------|--------------------|---------------------|
| LP | $\min_x c^\top x$ s.t. $A x=b,\,x\ge0$ | Linear constraints | Simplex, Interior-point | Resource allocation, Chebyshev regression |
| QP | $\min_x \tfrac12 x^\top Q x + c^\top x$ s.t. $A x\le b$ | $Q\succeq0$ | Interior-point, Active-set, CG | Ridge, SVM, Portfolio optimization |
| QCQP | $\min_x \tfrac12 x^\top P_0 x + q_0^\top x$ s.t. $\tfrac12 x^\top P_i x + q_i^\top x \le0$ | All $P_i\succeq0$ | Interior-point, SOCP reformulation | Robust regression, trust-region |
| SOCP | $\min_x f^\top x$ s.t. $\|A_i x + b_i\|_2 \le c_i^\top x + d_i$ | Cone constraints | Conic interior-point | Robust least-squares, risk limits |
| SDP | $\min_X \mathrm{Tr}(C^\top X)$ s.t. $\mathrm{Tr}(A_i^\top X)=b_i$, $X\succeq0$ | Matrix PSD constraint | Interior-point, low-rank first-order | Covariance estimation, control |
| GP | $\min_{x>0} f_0(x)$ s.t. $f_i(x)\le1,\,g_j(x)=1$ | Log-convex after $y=\log x$ | Log-transform + IPM | Circuit design, power control |
| MLE / GLM | $\min_x -\sum_i \log p(b_i|a_i^\top x)+\mathcal{R}(x)$ | Log-concave likelihood | Newton, L-BFGS, Prox / SGD | Logistic regression, Poisson GLMs |

 
## Linear Programming (LP)

Form

$$
\min_x c^\top x \quad \text{s.t. } A x=b,\, x\ge0
$$

Geometry: Feasible region = polyhedron; optimum = vertex.  
Applications: Resource allocation, shortest path, flow, scheduling.  
Algorithms:

1. Simplex: walks along edges (vertex-based).  
2. Interior-point: moves through the interior using log barriers.  
3. Decomposition: exploits block structure for large LPs.

 
## Quadratic Programming (QP)

Form

$$
\min_x \tfrac12 x^\top Q x + c^\top x 
\quad \text{s.t. } A x \le b,\, F x = g, \quad Q\succeq0
$$

Geometry: Objective = ellipsoids; feasible = polyhedron.  
Examples: Ridge regression, Markowitz portfolio, SVM.  
Algorithms:  
- Interior-point (smooth path).  
- Active-set (edge-following).  
- Conjugate Gradient for large unconstrained QPs.  
- First-order methods for massive $n$.

 
## Quadratically Constrained QP (QCQP)

Form

$$
\min_x \tfrac12 x^\top P_0x + q_0^\top x
\quad \text{s.t. } \tfrac12 x^\top P_i x + q_i^\top x + r_i \le 0
$$

Convex if all $P_i\succeq0$.  
Geometry: Intersection of ellipsoids and half-spaces.  
Applications: Robust control, filter design, trust-region.  
Algorithms: Interior-point (convex case), SOCP / SDP reformulations.

 
## Second-Order Cone Programming (SOCP)

Form

$$
\min_x f^\top x
\quad \text{s.t. } 
\|A_i x + b_i\|_2 \le c_i^\top x + d_i,\;
F x = g
$$

Interpretation: Linear objective, norm-bounded constraints.  
Applications: Robust regression, risk-aware portfolio, engineering design.  
Algorithms: Conic interior-point; scalable ADMM variants.  
Special case: Any QP or norm constraint can be written as an SOCP.

 
## Semidefinite Programming (SDP)

Form

$$
\min_X \mathrm{Tr}(C^\top X)
\quad \text{s.t. } \mathrm{Tr}(A_i^\top X)=b_i,\; X\succeq0
$$

Meaning: Variable = PSD matrix $X$; constraints = affine.  
Geometry: Feasible region = intersection of affine space with PSD cone.  
Applications: Control synthesis, combinatorial relaxations, covariance estimation, matrix completion.  
Algorithms: Interior-point for moderate $n$; low-rank proximal / Frank–Wolfe for large-scale.

 
## Geometric Programming (GP)

Original form

$$
\min_{x>0} f_0(x)
\quad \text{s.t. } f_i(x)\le1,\; g_j(x)=1
$$

where $f_i$ are posynomials and $g_j$ monomials.  

Log transformation: With $y=\log x$, the problem becomes convex in $y$.  
Applications: Circuit sizing, communication power control, resource allocation.  
Solvers: Convert to convex form → interior-point or primal-dual methods.

 
## Likelihood-Based Convex Models (MLE and GLMs)

General form

$$
\min_x -\sum_i \log p(b_i|a_i^\top x) + \mathcal{R}(x)
$$

Examples

| Noise Model | Objective | Equivalent Problem |
|--------------|------------|--------------------|
| Gaussian | $\|A x - b\|_2^2$ | Least squares |
| Laplacian | $\|A x - b\|_1$ | Robust regression (LP) |
| Bernoulli | $\sum_i \log(1+e^{-y_i a_i^\top x})$ | Logistic regression |
| Poisson | $\sum_i [a_i^\top x - y_i\log(a_i^\top x)]$ | Poisson GLM |

Algorithms  
- Newton or IRLS (small–medium).  
- Quasi-Newton / L-BFGS (moderate).  
- Proximal or SGD (large-scale).

 
## Solver Selection Summary

| Problem Type | Convex Form | Key Solvers | ML Examples |
|---------------|-------------|--------------|--------------|
| LP | Linear | Simplex, Interior-point | Minimax regression |
| QP | Quadratic | Interior-point, CG, Active-set | Ridge, SVM |
| QCQP | Quadratic + constraints | IPM, SOCP / SDP reformulation | Robust regression |
| SOCP | Cone | Conic IPM, ADMM | Robust least-squares |
| SDP | PSD cone | Interior-point, low-rank FW | Covariance, Max-cut relaxations |
| GP | Log-convex | Log-transform + IPM | Power allocation |
| MLE / GLM | Log-concave | Newton, L-BFGS, Prox-SGD | Logistic regression |

 