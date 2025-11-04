# Chapter 15: Canonical Problems in Convex Optimization

The table below gives canonical convex problem classes, with special cases listed *inside* their parent class to avoid duplication.  
Key hierarchy: LP ⊂ QP ⊂ SOCP ⊂ SDP (all are conic programs).

| Problem Type | Canonical Form | Special Cases  Examples | Typical Applications | Common Algorithms |
|---|---|---|---|---|
| Linear Program | $\displaystyle \min_x\; c^T x  \\ \text{s.t.}\; A x = b,\; x \ge 0$ |  - | Resource allocation, scheduling, routing, relaxations | Simplex  Dual simplex, Interior-point (barrier), Decomposition & first-order for very large-scale |
| Quadratic Program (convex if $Q \succeq 0$) | $\displaystyle \min_x\; \tfrac12 x^T Q x + c^T x \\ \text{s.t.}\; A x \le b,\; F x = g$ | Least Squares $\min\|Ax-b\|_2^2$  <br>  Ridge:$\min\|Ax-b\|_2^2+\lambda\|x\|_2^2$ <br> LASSO (via variable splitting → QP <br> Box-QP <br> Trust-region (QP with ball) | Portfolio optimization, MPC, regression & sparse estimation, large-scale ML fitting | Interior-point, Active-set, ProjectedProximal gradient, Conjugate gradient (unconstrained), Coordinate descentADMM (sparsestructured) |
| Second-Order Cone Program | $\displaystyle \min_x\; f^T x \\ \text{s.t.}\; \|A_i x + b_i\|_2 \le c_i^T x + d_i,\; F x = g$ | Robust least squares <br> Norm constraints ($\ell_2$) <br> Quadratic constraints representable as SOC <br> Chebyshev center <br> Portfolio with varianceVaR surrogates | Robust regression, antennabeamforming, control, risk-constrained finance | Conic interior-point (barrier), Primal-dual first-order splitting for large-scale |
| Semidefinite Program | $\displaystyle \min_X\; \mathrm{Tr}(C^T X) \\ \text{s.t.}\; \mathrm{Tr}(A_i^T X)=b_i,\; X \succeq 0$ | QCQP relaxations; Max-cutGoemans–Williamson; Covariance selection; Lyapunovcontrol synthesis; Sum-of-squares (SOS) relaxations | Control & systems, combinatorial relaxations, covariancecorrelation modeling, quantum information | Interior-point (medium-scale), Low-rank first-order (proxFrank–Wolfe), ADMMsplitting |
| Quadratically Constrained QP (convex if all $P_i \succeq 0$; otherwise generally NP-hard) | $\displaystyle \min_x\; \tfrac12 x^T P_0 x + q_0^T x + d_0 \\ \text{s.t.}\ \tfrac12 x^T P_i x + q_i^T x + d_i \le 0$ | Trust-region subproblems; Ellipsoidal constraints; Robust fitting; Many admit SOCPSDP reformulations | Robust control, beamforming, filter design, robust estimation | Interior-point (convex case), SDPSOCP relaxations, Convex–concave proceduresheuristics |
| GP (Geometric Program) (convex in log-space) | $\displaystyle \min_x\; f_0(x)\ \text{s.t.}\ f_i(x)\le 1,\ g_j(x)=1$ (posynomial $f_i$, monomial $g_j$) | After $y=\log x$: convex; posynomial circuit sizing; powerenergy scaling laws; resource trade-offs | CircuitIC design, communication & power control, engineering tuning | Log-transform → Interior-point; Primal-dual methods; First-order for large sparse instances |
| MLE  GLM (Likelihood-based convex models) | $\displaystyle \min_x\; -\sum_i \log p(b_i\mid a_i^T x)\ +\ \mathcal{R}(x)$ where $\mathcal{R}$ is convex (e.g., $\ell_1,\ell_2$) | Classification: Logisticsoftmax (GLMs); SVM as convex risk minimization (hinge loss); PoissonExponential family GLMs; Elastic-net regularized GLMs | Predictive modeling, classification, count modeling, calibration | Newton  (L-)BFGS, Acceleratedproximal gradient, Coordinate descent, Stochastic variants (SGDSAGASVRG), ADMM |


## Linear Programming (LP)

A standard LP has a linear objective and linear constraints:

$$
minimize_{x}\; c^T x
\quad\text{s.t.}\;
A x = b,\; x \ge 0.
$$

with decision variable $x\in\mathbb R^n$. Geometrically, the feasible set is a polyhedron (intersection of halfspaces), and optimal solutions lie at extreme points (vertices) of this polyhedron. Intuitively, imagine a flat objective plane being “pushed” until it first touches the polyhedron at a vertex.


### Algorithms: 
At a high level, LP algorithms fall into three families:

1. Vertex-based (Simplex)
2. Interior-based (Barrier  IPM)
3. Decomposition and First-Order (Large-Scale  Structured)

Each family explores the polyhedral geometry of the feasible set in a different way.


#### 1. Simplex Method — *Walking the edges of the polyhedron*

The Simplex method moves from vertex to vertex (corner to corner) of the feasible polyhedron, improving the objective value at each step.  
Intuitively, the LP feasible region is a *polyhedron* (a high-dimensional “flat-sided” shape) and the linear objective defines a *tilted plane* sweeping across it. The optimum is always attained at a vertex, so Simplex efficiently hops between adjacent vertices along edges until it reaches the best one.

#### 2. Interior-Point Methods — *Gliding through the interior*

Interior-point methods (IPMs) take a fundamentally different route:  
Instead of crawling along edges, they glide through the *interior* of the feasible region, guided by a barrier function that prevents them from hitting the boundaries.

They solve a sequence of barrier problems of the form:
\[
\min_x\; c^T x - \mu \sum_i \log(x_i)
\quad \text{s.t.}\; A x = b,
\]
where the logarithmic term keeps \(x_i > 0\).  

As the barrier parameter \(\mu \to 0\), the iterates approach the true boundary optimum along a *central path*.  Each step requires solving a Newton system that couples all variables, making IPMs especially efficient for dense, moderately sized LPs.

#### 3. Decomposition Methods — *Divide and conquer for large-scale structure*

When an LP is too large to solve as a single system, or its constraint matrix \(A\) has *block structure*, decomposition methods exploit separability by breaking the problem into smaller subproblems that can be solved independently.

The basic idea:  

- Identify coupling constraints or variables that link otherwise independent subsystems.  
- Solve each subsystem separately, and coordinate them via *dual variables* or *cutting planes* until consistency and optimality are achieved.

Decomposition turns “one huge LP” into “many small LPs talking to each other.”


### Applications
LPs model many resource-allocation and network-flow problems (e.g.\ transportation, scheduling, blending). They also capture piecewise-linear loss minimization. For example, the minimax (Chebyshev) regression $\min_x\max_i|a_i^T x - b_i|$ can be written as an LP. LPs appear in operations research, control (robust linear design), and in approximations of more complex problems.

## Quadratic Programming (QP)

A convex QP has a quadratic objective and linear constraints:

$$
minimize_{x}\;\tfrac12 x^T Q x + c^T x
\quad\text{s.t.}\;
A x = b,\;x\ge0,
$$

where $Q\succeq0$ (positive semidefinite) makes the problem convex. Here $x\in\mathbb R^n$. Equivalently, one may have inequality constraints $A x\le b$. Level sets of the objective are ellipsoids (quadratic bowls), so the optimum may lie on a boundary or in the interior of the feasible set.

Intuition & geometry: The positive-definite part of $Q$ defines an “ellipsoidal” bowl of the objective. At optimum, the gradient $\nabla(\tfrac12 x^T Q x + c^T x) = Qx+c$ is orthogonal to the active constraint surfaces (KKT stationarity). Unlike LP, an unconstrained QP’s minimizer is at $x=-Q^{-1}c$ if $Q\succ0$. With constraints, the optimum occurs where the ellipsoid just touches the feasible set (which is a polyhedron or affine subspace).

Applications: QPs model many smooth convex problems. A classic example is Markowitz portfolio optimization (minimize variance $x^T\Sigma x$ plus a linear return term). QPs arise in ridge regression (least-squares with $\ell_2$ penalty), support-vector machines (SVMs with quadratic soft-margin), and model-predictive control (convex quadratic cost). Any least-squares problem with linear constraints is a QP. Many robust or regularized designs (elastic net regression, norm-constrained classification) also lead to QPs.

Example Risk-Adjusted Quadratic Program (Mean–Variance Portfolio Optimization)

We want to allocate portfolio weights ( x \in \mathbb{R}^n ) across ( n ) assets.

* $\mu \in \mathbb{R}^n$: expected returns of each asset
* $\Sigma \in \mathbb{R}^{n \times n}$: covariance matrix of returns (symmetric, positive semidefinite)
* $x$: portfolio weights (fractions of capital invested in each asset)

The mean–variance trade-off says we want to:

* maximize expected return $\mu^T x$,
* minimize risk (variance) $x^T \Sigma x$.

Combining them gives the standard convex QP form:

\begin{aligned}
\min_x \quad & \tfrac{1}{2} x^T \Sigma x - \lambda \mu^T x \
\text{s.t.} \quad & \mathbf{1}^T x = 1, \
& x \ge 0,
\end{aligned}


where $\lambda > 0$ is the risk–return trade-off parameter.


* The quadratic term $\tfrac{1}{2} x^T \Sigma x$ penalizes portfolio variance (risk).
* The linear term $-\lambda \mu^T x$ rewards expected return.
* The equality constraint $\mathbf{1}^T x = 1$ ensures all capital is invested (weights sum to 1).
* The inequality $x \ge 0$ enforces no short selling.


### Algorithms
#### 1. Interior-Point Methods — *Smooth paths through the interior*

Interior-point methods extend the same idea used for LPs.  
They replace inequality constraints with barrier terms, e.g.
\[
\min_x\; \tfrac{1}{2}x^T Qx + c^T x - \mu \sum_i \log(s_i)
\quad \text{s.t.}\; A x + s = b,\; s > 0,
\]
where the logarithmic barrier keeps \(s_i > 0\).

Each iteration solves a Newton system derived from the Karush–Kuhn–Tucker (KKT) conditions, coupling primal and dual variables. The method follows a *central path* toward the optimum as \(\mu \to 0\).


#### 2. Active-Set Methods — *Walking along faces*

Active-set methods generalize the Simplex idea to QPs.

1. Guess which constraints are active (tight) at the solution.  
2. Solve the resulting equality-constrained QP:
   \[
   \min_x\; \tfrac{1}{2}x^T Qx + c^T x
   \quad \text{s.t.}\; A_{\text{active}} x = b_{\text{active}}.
   \]
3. Check which inactive constraints become violated; update the working set and repeat.
Each iteration moves along the *face* of the feasible region, turning when constraints become active or inactive.

#### 3. Newton’s Method — *One-shot solution when unconstrained*

If there are no constraints, the QP reduces to:
\[
\min_x\; \tfrac{1}{2}x^T Qx + c^T x.
\]

Setting the gradient to zero gives:
\[
Qx + c = 0 \quad \Rightarrow \quad x^* = -Q^{-1} c.
\]

Since the Hessian \(Q\) is constant, Newton’s method reaches the optimum in one iteration.


#### 4. Conjugate Gradient (CG) — *Iterative linear solver for large problems*

When \(Q\) is large and sparse, inverting or factorizing it is too expensive. Instead, Conjugate Gradient (CG) solves \(Qx = -c\) iteratively using only matrix–vector products. If \(Q = A^T A\), the QP corresponds to a least-squares problem:
\[
\min_x \|A x - b\|_2^2,
\]
and CG can efficiently find the minimizer without forming \(A^T A\).

#### 5. Accelerated Gradient Methods — *First-order scalability*

For extremely large QPs with simple constraints (\(x \ge 0\) or box bounds), first-order methods are used:
\[
x_{k+1} = \Pi_{\mathcal{C}}(x_k - \alpha (Qx_k + c)),
\]
where \(\Pi_{\mathcal{C}}\) projects onto the feasible region.

## Quadratically Constrained QP

$$
minimize_{x}\;\tfrac12x^T P_0 x + q_0^T x + r_0
\quad\text{s.t.}\quad
\tfrac12x^T P_i x + q_i^T x + r_i \le 0.
$$

where each $P_i\in\mathbb R^{n\times n}$ and $x\in\mathbb R^n$. If all $P_i\succeq0$, the feasible set is convex, and the QCQP is convex. (If any $P_i$ is indefinite, the problem is generally nonconvex and NP-hard.) A special case is QP (no quadratic constraints). A common simpler form is a single quadratic (ellipsoidal) constraint, as in the trust-region problem.

Intuition & geometry: Convex QCQPs describe intersections of “ellipsoidal” regions (and possibly affine sets). For example, the constraint $\tfrac12x^TPx + q^Tx + r\le0$ defines a (possibly unbounded) ellipsoid or paraboloid region. Geometrically, a convex QCQP feasible set can be an ellipsoid, paraboloid, or their intersections. The optimum lies where the objective’s ellipsoidal level set first touches this intersection. When $P_0\succeq0$, the objective contours are convex (ellipsoids); if $P_i\succ0$, constraint boundaries are convex surfaces.

Applications: QCQPs appear in robust design and engineering. For instance, in robust beamforming or filter design one often imposes constraints on quadratic forms of $x$. Sensor network localization and trust-region subproblems are QCQPs. Robust linear regression against ellipsoidal noise uncertainty is a QCQP. Any problem with norms or quadratic inequalities (e.g.\ $|Fx-c|_2^2\le d^2$) can be written as a QCQP.

### Algorithms

Convex Quadratically Constrained Quadratic Programs (QCQPs) where all \(P_i \succeq 0\) can be solved efficiently using interior-point methods, which provide polynomial-time convergence.  
Because QCQPs can be expressed as second-order cone programs (SOCPs) or more generally as semidefinite programs (SDPs), modern conic solvers (e.g., MOSEK, SCS, SDPT3) handle these problems robustly for moderate dimensions.


## Second-Order Cone Programming (SOCP)

$$
minimize_{x}\; f^T x
\quad\text{s.t.}\quad
\|A_i x + b_i\|_2 \le c_i^T x + d_i,\;
F x = g.
$$

where each $|A_i x + b_i|_2\le c_i^T x + d_i$ is a second-order cone constraint. Equivalently, each constraint says the affine function $(x,t)\mapsto (A_i x + b_i, c_i^T x + d_i)$ lies in the norm cone {${(u,t)\mid|u|_2\le t}\$}. Thus the feasible set is the intersection of affine spaces and cones (a convex cone itself).

Intuition & geometry: Each second-order constraint carves out a convex “cone” in $(x,t)$-space. Geometrically, a 2-norm constraint $|u|\le t$ forms a (rotated) quadratic cone. The feasible set is the intersection of these cones with any affine constraints. For example, $|x|_2 \le 1$ is a unit ball (an ellipsoid), which is a simple special case of an SOCP. SOCPs generalize linear constraints ($\ell_1$ norm cones) and some QCQPs.

Applications: SOCPs are widely used in engineering and finance. Examples include robust least-squares (using Huber or $\ell_2$ penalties), design of filters and antennas, truss or structural design under stress norms, and certain portfolio models with risk measures. In statistics and ML, enforcing $|w|_2\le R$ is an SOCP constraint. Many chance-constrained or variance-based optimization problems lead to SOCPs. Notably, portfolio optimization with a value-at-risk (VaR) constraint can be cast as an SOCP
en.wikipedia.org


Algorithms: Interior-point methods are the standard solvers for SOCPs. Off-the-shelf conic solvers (e.g.\ MOSEK, ECOS) efficiently handle moderate-size SOCPs. Compared to SDP, SOCPs are usually faster to solve for similar problem sizes. First-order or decomposition methods (e.g.\ ADMM) can scale to larger SOCPs if needed.

## Geometric Programming (GP)

Original (nonconvex) form:

$$
minimize_{x>0}\; f_0(x)
\quad\text{s.t.}\;
f_i(x)\le1,\quad g_j(x)=1,
$$

where each $f_i$ is a posynomial (a sum of monomials) and each $g_j$ is a monomial. In coordinates, a monomial has the form $c,x_1^{a_1}x_2^{a_2}\cdots x_n^{a_n}$ with $c>0$, and a posynomial is a sum of such terms.

Intuition & convexity: Though GPs are not convex in the original variables, they can be made convex by the change of variables $y_i=\log x_i$ and taking logs of functions. Under this log–log transformation, each monomial becomes an affine function of $y$, and each posynomial becomes a log-sum-exp (convex) function. Thus every GP can be converted to a convex problem (a log-sum-exp minimization). Geometrically, GPs operate on positive variables with multiplicative relationships; in the log-domain, they become standard convex programs.

Applications: GPs have many applications in engineering design. For example, circuit and analog IC component sizing (transistors, amplifiers) is often cast as a GP. Other examples include aircraft design, power system design, and network flow with multiplicative constraints. In statistics, certain inference problems (e.g. fitting log-linear models) can be written as GPs. Any problem where costs and constraints are products of powers of variables (posynomials) is a candidate for geometric programming
en.wikipedia.org


Algorithms: The usual approach is to transform the GP into a convex form and apply convex solvers. After $y=\log x$, one solves a convex program using interior-point methods on the log-transformed problem. Off-the-shelf convex solvers (e.g.\ CVXOPT, MOSEK) now directly support GPs by performing this transformation internally. Specialized GP toolkits (e.g.\ GPkit) also exist. For sequential design, one can iteratively solve approximate GPs. Because GPs become convex, KKT conditions and duality apply after transformation.

## Maximum Likelihood and Generalized Linear Models (GLM)

### 1. Maximum Likelihood Estimation (MLE)

* Parametric distribution estimation: Choose from a family of densities $p_x(y)$, indexed by a parameter $x$ (often denoted $\theta$).
* We take $p_x(y) = 0$ for invalid values of $x$.
* The likelihood function is $p_x(y)$ as a function of $x$.
* The log-likelihood function is $l(x) = \log p_x(y)$.

Maximum Likelihood Estimation (MLE):
$$
\hat{x} = \arg\max_x p_x(y) = \arg\max_x l(x)
$$

This is often a convex optimization problem when $\log p_x(y)$ is concave in $x$ for fixed $y$.

 
### 2. Linear Measurements with IID Noise

Consider the linear model:
$$
y_i = a_i^T x + \nu_i, \quad i = 1, \ldots, m
$$

where:

* $x \in \mathbb{R}^n$: unknown parameter vector,
* $\nu_i$: IID measurement noise with density $p(z)$,
* $y_i$: observed measurements.

Then the joint density is:
$$
p_x(y) = \prod_{i=1}^m p(y_i - a_i^T x)
$$

and the log-likelihood becomes:
$$
l(x) = \sum_{i=1}^m \log p(y_i - a_i^T x)
$$

The MLE is any $x$ that maximizes $l(x)$.

 
### 3. Common Noise Models and Corresponding MLEs

#### (a) Gaussian Noise $\mathcal{N}(0, \sigma^2)$

$$p(z) = (2\pi\sigma^2)^{-1/2} e^{-z^2 / (2\sigma^2)}$$

$$l(x) = -\frac{m}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^m (a_i^T x - y_i)^2$$

MLE: Least-squares solution
$$
\hat{x} = \arg\min_x |Ax - y|_2^2
$$

 
#### (b) Laplacian Noise

$$p(z) = \frac{1}{2a} e^{-|z|/a}$$

$$l(x) = -m\log(2a) - \frac{1}{a} \sum_{i=1}^m |a_i^T x - y_i|$$

MLE: $\ell_1$-norm solution (least absolute deviations)
$$
\hat{x} = \arg\min_x |Ax - y|_1
$$

This is convex but non-smooth; it can be solved via LP, proximal gradient, or robust convex solvers.

#### (c) Uniform Noise on $[-a, a]$

$$
p(z) =
\begin{cases}
\frac{1}{2a}, & |z| \le a \
0, & \text{otherwise}
\end{cases}
$$

$$
l(x) =
\begin{cases}
-m \log(2a), & |a_i^T x - y_i| \le a, \ \forall i \
-\infty, & \text{otherwise}
\end{cases}
$$

MLE: Any $x$ satisfying
$$
|a_i^T x - y_i| \le a, \quad i = 1, \ldots, m
$$

This defines a feasible region (a convex polyhedron).


### 4. Logistic Regression (Bernoulli Likelihood)

For binary labels $y \in {0,1}$:
$$
p = \Pr(y = 1 \mid u) = \frac{\exp(a^T u + b)}{1 + \exp(a^T u + b)}
$$
where $a, b$ are parameters and $u \in \mathbb{R}^n$ are observed features.

For $m$ samples $(u_i, y_i)$, the log-likelihood is:
$$
l(a,b) = \sum_{i=1}^m \big[ y_i(a^T u_i + b) - \log(1 + e^{a^T u_i + b}) \big]
$$

and the negative log-likelihood (to minimize) is:
$$
L(a,b) = -l(a,b) = \sum_{i=1}^m \big[ \log(1 + e^{a^T u_i + b}) - y_i(a^T u_i + b) \big]
$$

This function is convex and smooth in $a, b$.

 
### 5. Softmax (Multiclass Logistic Regression)

For $K$ classes with one-hot encoded labels $y_i$ and parameters $x_k$ for each class:
$$
p(y_i = k \mid a_i) = \frac{e^{a_i^T x_k}}{\sum_{j=1}^K e^{a_i^T x_j}}
$$

Negative log-likelihood:
$$
L({x_k}) = -\sum_{i,k} y_{ik}(a_i^T x_k) + \sum_i \log\Big(\sum_{j=1}^K e^{a_i^T x_j}\Big)
$$

Convex in all $x_k$.

### Algorithms for MLE and GLMs

- LS/Gaussian: Closed-form or CG/Newton as above.
- ℓ₁ regression (Laplace): Convert to LP or use interior-point, or first-order methods (subgradient/proximal) for large-scale. 
- Logistic/softmax: Use Newton or quasi-Newton (L-BFGS) when moderate-sized, since the loss is smooth. Accelerated gradient or stochastic gradient (SGD/Adam) is common for large datasets. With $\ell_2$-regularization the problem is strongly convex, ensuring fast convergence. With $\ell_1$-regularization one uses proximal-gradient or coordinate descent to handle the non-smooth part. 
- GLMs: Iteratively Reweighted Least Squares (IRLS, a Newton method) often solves canonical-link GLMs efficiently.



## Support Vector Machines

$$
\min_w\;\tfrac12\|w\|_2^2 + C\sum_i \max(0,1 - y_i w^T a_i).
$$

The SVM primal problem is a QP of the form $, which is convex but non-smooth. This QP is typically solved via its dual or by specialized coordinate-descent.

