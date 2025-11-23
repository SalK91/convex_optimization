# Chapter 12: Algorithms for Convex Optimization
In the previous chapters, we built the mathematical foundations of convex optimization: convex sets, convex functions, gradients, subgradients, KKT conditions, and duality. Now we answer the practical question: How do we actually solve convex optimization problems in practice?

This chapter now serves as the algorithmic backbone of the book. It bridges theoretical convex analysis (Chapters 3–11) with the practical numerical methods that solve those problems. Each algorithm here can be seen as a computational lens on a convex geometry concept — gradients as supporting planes, Hessians as curvature maps, and proximal maps as projection operators. Later chapters (13–15) extend these ideas to constrained, stochastic, and large-scale environments.

 
## 12.1 Problem classes vs method classes

Different convex problems call for different algorithmic structures. Here is the broad landscape:

| Problem Type | Typical Formulation | Representative Methods | Examples |
|---------------|--------------------|-------------------------|-----------|
| Smooth, unconstrained | $\min_x f(x)$, convex and differentiable | Gradient descent, Accelerated gradient, Newton | Logistic regression, least squares |
| Smooth with simple constraints | $\min_x f(x)$ s.t. $x \in \mathcal{X}$ (box, ball, simplex) | Projected gradient | Constrained regression, probability simplex |
| Composite convex (smooth + nonsmooth) | $\min_x f(x) + R(x)$ | Proximal gradient, coordinate descent | Lasso, Elastic Net, TV minimization |
| General constrained convex | $\min f(x)$ s.t. $g_i(x) \le 0, h_j(x)=0$ | Interior-point, primal–dual methods | LP, QP, SDP, SOCP |


     

## 12.2 First-order methods: Gradient descent

We solve
$$
\min_x f(x),
$$
where $f$ is convex, differentiable, and (ideally) $L$-smooth: its gradient is Lipschitz with constant $L$, meaning
$$
\|\nabla f(x) - \nabla f(y)\|_2 \le L \|x-y\|_2 \quad \text{for all } x,y.
$$
> Smoothness lets us control step sizes.

Gradient descent iterates
$$
x_{k+1} = x_k - \alpha_k \nabla f(x_k),
$$
where $\alpha_k>0$ is the step size (also called learning rate in machine learning). Typical choices:

- constant $\alpha_k = 1/L$ when $L$ is known,
- backtracking line search when $L$ is unknown,
- diminishing step sizes in some settings.



> Derivation: 

> Around $x_t$, we can approximate $f$ using its Taylor expansion:

> $$
f(x) \approx f(x_t) + \langle \nabla f(x_t), x - x_t \rangle.
$$


> We assume $f$ behaves approximately like its tangent plane near $x_t$.  But tf we were to minimize just this linear model, we would move infinitely far in the direction of steepest descent $-\nabla f(x_t)$, which is not realistic or stable. This motivates adding a locality restriction: we trust the linear approximation near $x_t$, not globally. To prevent taking arbitrarily large steps, we add a quadratic penalty for moving away from $x_t$:

> $$
f(x) \approx f(x_t) + \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{2\eta} \|x - x_t\|^2,
$$

> where $\eta > 0$ is the learning rate or step size.

> - The linear term pulls $x$ in the steepest descent direction.
> - The quadratic term acts like a trust region, discouraging large deviations from $x_t$.
> - $\eta$ trades off aggressive progress vs stability:
>     - Small $\eta$ → cautious updates.
>     - Large $\eta$ → bold updates (risk of divergence).


> We define the next iterate as the minimizer of the surrogate objective:

> $$
x_{t+1} = \arg\min_{x \in \mathcal{X}} \Big[ f(x_t) + \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{2\eta} \|x - x_t\|^2 \Big].
$$

> Ignoring the constant term $f(x_t)$ and differentiating w.r.t. $x$:

> $$
\nabla f(x_t) + \frac{1}{\eta}(x - x_t) = 0
$$

> Solving:

> $$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

Convergence: For convex, $L$-smooth $f$, gradient descent with a suitable fixed step size satisfies
$$
f(x_k) - f^\star = O\!\left(\frac{1}{k}\right),
$$
where $f^\star$ is the global minimum. This $O(1/k)$ sublinear rate is slow compared to second-order methods, but each step is extremely cheap: you only need $\nabla f(x_k)$.

When to use gradient descent:

- High-dimensional smooth convex problems (e.g. large-scale logistic regression).
- You can compute gradients cheaply.
- You only need moderate accuracy.
- Memory constraints rule out storing or factoring Hessians.



 
## 12.3 Accelerated first-order methods

Plain gradient descent has an $O(1/k)$ rate for smooth convex problems. Remarkably, we can do better — and in fact, provably optimal — by adding *momentum*.

### 12.3.1 Nesterov acceleration
Nesterov’s accelerated gradient method modifies the update using a momentum-like extrapolation. One common form of Nesterov acceleration uses two sequences $x_k$ and $y_k$:

1. Maintain two sequences $x_k$ and $y_k$.
2. Take a gradient step from $y_k$:
   $$
   x_{k+1} = y_k - \alpha \nabla f(y_k).
   $$
3. Extrapolate:
   $$
   y_{k+1} = x_{k+1} + \beta_k (x_{k+1} - x_k).
   $$

The extra momentum term $\beta_k (x_{k+1}-x_k)$ uses past iterates to “look ahead” and can significantly accelerate convergence.

Convergece: For smooth convex $f$, accelerated gradient achieves
$$
f(x_k) - f^\star = O\!\left(\frac{1}{k^2}\right),
$$
which is *optimal* for any algorithm that uses only gradient information and not higher derivatives.


- Acceleration is effective for well-behaved smooth convex problems.
- It can be more sensitive to step size and noise than plain gradient descent.
- Variants such as FISTA apply acceleration in the composite setting $f + R$.

> The convergence of gradient descent depends strongly on the geometry of the level sets of the objective function. When these level sets are poorly conditioned—that is, highly anisotropic or elongated (not spherical) the gradient directions tend to oscillate across narrow valleys, leading to zig-zag behavior and slow convergence. In contrast, when the level sets are well-conditioned (approximately spherical), gradient descent progresses efficiently toward the minimum. Thus, the efficiency of gradient-based methods is governed by how aspherical (anisotropic) the level sets are, which is directly related to the condition number of the Hessian.

## 12.4 Steepest Descent Method
 
The steepest descent method generalizes gradient descent by depending on the choice of norm used to measure step size or direction. It finds the direction of *maximum decrease* of the objective function under a unit norm constraint.


> The norm defines the “geometry” of optimization.cGradient descent is steepest descent under the Euclidean norm. Changing the norm changes what “steepest” means, and can greatly affect convergence, especially for ill-conditioned or anisotropic problems.The norm in steepest descent determines the geometry of the descent and choosing an appropriate norm effectively makes the level sets of the function more rounded (more isotropic), which greatly improves convergence.
  
At a point $x$, and for a chosen norm $|\cdot|$:

$$
\Delta x_{\text{nsd}} = \arg\min_{|v| = 1} \nabla f(x)^T v
$$

This defines the normalized steepest descent direction — the unit-norm direction that yields the most negative directional derivative (i.e., the steepest local decrease of $f$).

* $\Delta x_{\text{nsd}}$: normalized steepest descent direction
* $\Delta x_{\text{sd}}$: unnormalized direction (scaled by the gradient norm)


For small steps $v$,
$$
f(x + v) \approx f(x) + \nabla f(x)^T v.
$$
The term $\nabla f(x)^T v$ describes how fast $f$ increases in direction $v$.
To decrease $f$ most rapidly, we pick $v$ that minimizes this inner product — subject to $|v| = 1$.

* The result depends on which norm we use to measure the “size” of $v$.
* The corresponding dual norm $|\cdot|_*$ determines how we measure the gradient’s magnitude.

Thus, the steepest descent direction always aligns with the negative gradient, but it is scaled and shaped according to the geometry induced by the chosen norm.


## 12.4.1. Mathematical Properties

### (a) Normalized direction

$$
\Delta x_{\text{nsd}} = \arg\min_{|v|=1} \nabla f(x)^T v
$$
→ unit vector with the most negative directional derivative.

### (b) Unnormalized direction

$$
\Delta x_{\text{sd}} = |\nabla f(x)| , \Delta x*{\text{nsd}}
$$
This gives the actual direction and magnitude used in updates.

### (c) Key identity

$$
\nabla f(x)^T \Delta x_{\text{sd}} = -|\nabla f(x)|_*^2
$$
The directional derivative equals the negative squared dual norm of the gradient.



### 12.4.2. The Steepest Descent Method

The iterative update rule is:
$$
x_{k+1} = x_k + t_k , \Delta x_{\text{sd}},
$$
where $t_k > 0$ is a step size (from line search or a fixed rule).

* For the Euclidean norm, this reduces to ordinary gradient descent.
* For other norms, it adapts the search direction to the geometry of the problem.

Convergence: Similar to gradient descent — linear for general convex functions, potentially faster when level sets are well-conditioned.



### 12.4.3. Role of the Norm and Its Influence

The choice of norm determines:

1. The shape of the unit ball ${v : |v| \le 1}$,
2. The direction of steepest descent, since the minimization is constrained by that shape,
3. The dual norm $|\nabla f(x)|_*$ that measures the gradient’s size.

Different norms yield different “geometries” of descent:

| Norm                        | Unit Ball Shape | Dual Norm         | Effect on Direction                         |
|  |  | -- | - |
| $\ell_2$                    | Circle / sphere | $\ell_2$          | Direction is opposite to gradient           |
| $\ell_1$                    | Diamond         | $\ell_\infty$     | Moves along coordinate of largest gradient  |
| $\ell_\infty$               | Square          | $\ell_1$          | Moves opposite to sum of all gradient signs |
| Quadratic $(x^T P x)^{1/2}$ | Ellipsoid       | Weighted $\ell_2$ | Scales direction by preconditioner $P^{-1}$ |

Thus, the norm defines how “distance” and “steepness” are perceived, shaping how the algorithm moves through the landscape of $f(x)$.

### (a) Euclidean Norm $|v|_2$

$$
\Delta x_{\text{nsd}} = -\frac{\nabla f(x)}{|\nabla f(x)|*2},
\quad
\Delta x*{\text{sd}} = -\nabla f(x)
$$

This is standard gradient descent.
The direction is exactly opposite the gradient, and steps are isotropic (same scaling in all directions).



### (b) Quadratic Norm $|v|_P = (v^T P v)^{1/2}$, with $P \succ 0$

Here, $P$ defines an ellipsoidal metric.
The dual norm is $|y|_* = (y^T P^{-1} y)^{1/2}$.

$$
\Delta x_{\text{sd}} = -P^{-1}\nabla f(x)
$$

This corresponds to preconditioned gradient descent, where $P$ rescales directions to counter anisotropy in level sets.

Interpretation:

* If $P$ approximates the Hessian, this becomes Newton’s method.
* If $P$ is diagonal, it acts like an adaptive step size per coordinate.



### (c) $\ell_1$-Norm

$$
\Delta x_{\text{nsd}} = -e_i, \quad i = \arg\max_j \left|\frac{\partial f}{\partial x_j}\right|
$$

and

$$
\Delta x_{\text{sd}} = -|\nabla f(x)|_\infty e_i
$$

The step moves along the coordinate with the largest gradient component, resembling a coordinate descent update.

Geometric intuition:
The $\ell_1$-unit ball is a diamond; its corners align with coordinate axes, so the steepest direction is along one axis at a time.



* In $\ell_2$-norm: the unit ball is a circle → the steepest direction is exactly opposite the gradient.
* In $\ell_1$-norm: the unit ball is a diamond → the steepest direction points to a corner (one coordinate).
* In quadratic norms: the unit ball is an ellipsoid → the steepest direction follows the metric-adjusted gradient.

Hence, the norm defines the geometry of what “steepest” means.


## 12.5 Conjugate Gradient Method — Fast Optimization for Quadratic Objectives

Gradient descent can be painfully slow when the level sets of the objective are long and skinny an indication that the Hessian has very different curvature in different directions (poor conditioning). The Conjugate Gradient (CG) method fixes this without forming or inverting the Hessian. It exploits the exact structure of quadratic functions to build advanced search directions that incorporate curvature information at almost no extra cost.

CG is a *first-order* method that behaves like a *second-order* method for quadratics.


For a quadratic objective function:

$$
f(x) = \tfrac12 x^\top A x - b^\top x 
$$

with $A \succ 0$, the level sets are ellipses shaped by the eigenvalues of $A$. If $A$ is ill-conditioned, these ellipses are highly elongated. Gradient descent follows the steepest Euclidean descent direction, which points perpendicular to level sets. On elongated ellipses, this produces a zig-zag path that wastes many iterations.

CG replaces the steepest-descent directions with conjugate directions. Two nonzero vectors $p_i, p_j$ are said to be A-conjugate if

$$
p_i^\top A p_j = 0.
$$

This is orthogonality measured in the geometry induced by the Hessian $A$. Why is this useful?

- Moving along an A-conjugate direction eliminates error components associated with a different eigen-direction of $A$.
- Once you minimize along a conjugate direction, you never need to correct that direction again.
- After $n$ mutually A-conjugate directions, all curvature directions are resolved → exact solution.

In contrast, gradient descent repeatedly re-corrects previous progress.


Algorithm (Linear CG): We solve the quadratic minimization problem or, equivalently, the linear system $Ax = b$. Let

$$
r_0 = b - A x_0, \qquad p_0 = r_0.
$$

For $k = 0,1,2,\dots$:

1. Step size
   $$
   \alpha_k = \frac{r_k^\top r_k}{p_k^\top A p_k}.
   $$

2. Update iterate
   $$
   x_{k+1} = x_k + \alpha_k p_k.
   $$

3. Update residual (negative gradient)
   $$
   r_{k+1} = r_k - \alpha_k A p_k.
   $$

4. Direction scaling
   $$
   \beta_k = \frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k}.
   $$

5. New conjugate direction
   $$
   p_{k+1} = r_{k+1} + \beta_k p_k.
   $$

Stop when $\|r_k\|$ is below tolerance.

Every new direction $p_{k+1}$ is constructed to be A-conjugate to all previous ones, and this is preserved automatically by the recurrence.

Why CG Is Fast: For an $n$-dimensional quadratic, CG solves the problem in at most $n$ iterations in exact arithmetic. In practice, due to floating-point errors and finite precision, it converges much earlier, typically in $O(\sqrt{\kappa})$ iterations, where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number. The convergence bound in the A-norm is:

$$
\|x_k - x^\star\|_A \le 
2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k 
\|x_0 - x^\star\|_A.
$$

This is dramatically better than the $O(1/k)$ rate of gradient descent.


CG is ideal when:

- The problem is a quadratic or a linear system with symmetric positive definite (SPD) matrix $A$.
- $A$ is large and sparse or available as a matrix–vector product.
- You cannot form or store $A^{-1}$ or even the full matrix $A$.
- You want a Hessian-aware method but cannot afford Newton’s method.

Typical scenarios:

| Application | Why CG fits |
|------------|-------------|
| Large linear systems $A x = b$ | Only requires $A p$, not factorization. |
| Ridge regression | Normal equations form an SPD matrix. |
| Kernel ridge regression | Solves $(K+\lambda I)\alpha = y$ efficiently. |
| Newton steps in ML | Inner solver for Hessian systems without forming Hessian. |
| PDEs and scientific computing | Sparse SPD matrices, ideal for CG. |


Assumptions Required for CG: To guarantee correctness of *linear CG*, we require:

- $A$ is symmetric
- $A$ is positive definite
- Objective is strictly convex quadratic
- Arithmetic is exact (for the finite-step guarantee)

If the function is *not* quadratic or Hessian is not SPD, use Nonlinear CG, which generalizes the idea but loses finite-step guarantees.


Practical Notes:

- You only need matrix–vector products $Ap$.  
- Storage cost is $O(n)$.  
- Preconditioning (replacing the system with $M^{-1} A$) improves conditioning and accelerates convergence dramatically.  
- Periodic re-orthogonalization can help in long runs with floating-point drift.


> CG is the optimal descent method for quadratic objectives:  it constructs Hessian-aware conjugate directions that efficiently resolve curvature, giving Newton-like speed while requiring only gradient-level operations.



## 12.6 Newton’s method and second-order methods

First-order methods (like gradient descent) only use gradient information. Newton’s method, in contrast, incorporates curvature information from the Hessian to take steps that better adapt to the local geometry of the function. This often leads to much faster convergence near the optimum.


From Chapter 3, the second-order Taylor approximation of $f(x)$ around a point $x_k$ is:

$$
f(x_k + d)
\approx
f(x_k)
+ \nabla f(x_k)^\top d
+ \tfrac{1}{2} d^\top \nabla^2 f(x_k) d.
$$

If we temporarily trust this quadratic model, we can choose $d$ to minimize the right-hand side. Differentiating with respect to $d$ and setting to zero gives:

$$
\nabla^2 f(x_k) \, d_{\text{newton}} = - \nabla f(x_k).
$$

Hence, the Newton step is:

$$
d_{\text{newton}} = - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k),
\quad
x_{k+1} = x_k + d_{\text{newton}}.
$$


This step aims directly at the stationary point of the local quadratic model. When the iterates are sufficiently close to the true minimizer of a strictly convex $f$, Newton’s method achieves quadratic convergence—dramatically faster than the $O(1/k)$ or $O(1/k^2)$ rates typical of first-order algorithms.

However, far from the minimizer the quadratic model may be inaccurate, the Hessian may be indefinite, or the step may be unreasonably large. For stability, Newton’s method is almost always paired with a line search or trust-region strategy that adjusts step length based on how well the model predicts actual decrease.

### Solving the Newton System
Each iteration requires solving

$$
H \,\Delta x = -g,
\qquad
H = \nabla^2 f(x), \;\; g = \nabla f(x).
$$

If $H$ is symmetric positive definite, a Cholesky factorization

$$
H = L L^\top
$$

allows efficient and numerically stable solution via two triangular solves:

1. $L y = -g$
2. $L^\top \Delta x_{\text{nt}} = y$

This avoids forming $H^{-1}$ explicitly.


The Newton decrement:

$$
\lambda(x) = \|L^{-1} g\|_2
$$

gauges proximity to the optimum and provides a natural stopping criterion: $\lambda(x)^2/2 < \varepsilon$.

Computationally, the dominant cost is solving the Newton system. For dense, unstructured problems this costs $\approx (1/3)n^3$ operations, though sparsity or structure can reduce this dramatically. Because of this cost, Newton’s method is most appealing for problems of moderate dimension or for situations where Hessian systems can be solved efficiently using sparse linear algebra or matrix–free iterative methods.


### Gauss–Newton Method

The Gauss–Newton method is a specialization of Newton’s method for nonlinear least squares problems

$$
f(x) = \tfrac12 \| r(x) \|^2,
$$

where $r(x)$ is a vector of residual functions and a nonlinear function of $x$ and $J$ is its Jacobian. Newton’s Hessian decomposes as

$$
\nabla^2 f(x) = J^\top J \;+\; \sum_i r_i(x)\, \nabla^2 r_i(x).
$$

The second term involves the curvature of the residuals. When $r(x)$ is approximately linear near the optimum, this term is small. Gauss–Newton drops it, giving the approximation

$$
\nabla^2 f(x) \approx J^\top J,
$$

leading to the Gauss–Newton step:

$$
(J^\top J)\, \Delta = -J^\top r.
$$

Thus each iteration reduces to solving a (potentially large but structured) least-squares system, avoiding full Hessians entirely. The Levenberg–Marquardt method adds a damping term,

$$
(J^\top J + \lambda I)\, \Delta = -J^\top r,
$$

which interpolates smoothly between  

- gradient descent (large $\lambda$), and  
- Gauss–Newton (small $\lambda$).

Damping improves robustness when the Jacobian is rank-deficient or when the neglected second-order terms are not negligible Gauss–Newton and Levenberg–Marquardt are highly effective when the residuals are nearly linear—common in curve fitting, bundle adjustment, and certain layerwise training procedures in deep learning—yielding fast convergence without the expense of full second derivatives.



### Quasi-Newton methods

When computing or storing the Hessian is too expensive, we can build low-rank approximations of $\nabla^2 f(x_k)$ or its inverse. These methods use gradient information from previous steps to estimate curvature.

The most famous examples are:

- BFGS (Broyden–Fletcher–Goldfarb–Shanno)  
- DFP (Davidon–Fletcher–Powell)  
- L-BFGS (Limited-memory BFGS) — for very large-scale problems.

Quasi-Newton methods (BFGS, L-BFGS) build inverse-Hessian approximations from gradient differences, achieving superlinear convergence with low memory. They maintain many of Newton’s fast local convergence properties, but with per-iteration costs similar to first-order methods. For instance, BFGS maintains an approximation $B_k \approx \nabla^2 f(x_k)^{-1}$ updated via gradient and step differences:

$$
B_{k+1} = B_k + \frac{(s_k^\top y_k + y_k^\top B_k y_k)}{(s_k^\top y_k)^2} s_k s_k^\top
- \frac{B_k y_k s_k^\top + s_k y_k^\top B_k}{s_k^\top y_k},
$$

where $s_k = x_{k+1} - x_k$ and $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$.

These methods achieve superlinear convergence in practice, making them popular for large smooth optimization problems.



When to use Newton or quasi-Newton methods:

- You need high-accuracy solutions.  
- The problem is smooth and reasonably well-conditioned.  
- The dimension is moderate, or Hessian systems can be solved efficiently (e.g., using sparse linear algebra).  

For large, ill-conditioned, or nonsmooth problems, first-order or proximal methods (Chapter 10) are typically more suitable.


 

## 12.8 Constraints and nonsmooth terms: projection and proximal methods

In practice, most convex objectives are not just “nice smooth $f(x)$”. They often have:

- constraints $x \in \mathcal{X}$,
- nonsmooth regularisers like $\|x\|_1$,
- penalties that encode robustness or sparsity (Chapter 6).

Two core ideas handle this: projected gradient and proximal gradient.

### 12.8.1 Projected gradient descent

Setting: Minimise convex, differentiable $f(x)$ subject to $x \in \mathcal{X}$, where $\mathcal{X}$ is a simple closed convex set (Chapter 4).

Algorithm:

1. Gradient step:
   $$
   y_k = x_k - \alpha \nabla f(x_k).
   $$
2. Projection:
   $$
   x_{k+1}
   =
   \Pi_{\mathcal{X}}(y_k)
   :=
   \arg\min_{x \in \mathcal{X}} \|x - y_k\|_2^2~.
   $$

Interpretation:

- You take an unconstrained step downhill,
- then you “snap back” to feasibility by Euclidean projection.

Examples of $\mathcal{X}$ where projection is cheap:

- A box: $l \le x \le u$ (clip each coordinate).
- The probability simplex $\{x \ge 0, \sum_i x_i = 1\}$ (there are fast projection routines).
- An $\ell_2$ ball $\{x : \|x\|_2 \le R\}$ (scale down if needed).

Projected gradient is the constrained version of gradient descent. It maintains feasibility at every iterate.

### 12.8.2 Proximal gradient (forward–backward splitting)

Setting: Composite convex minimisation
$$
\min_x \; F(x) := f(x) + R(x),
$$
where:

- $f$ is convex, differentiable, with Lipschitz gradient,
- $R$ is convex, possibly nonsmooth.

Typical choices of $R(x)$:

- $R(x) = \lambda \|x\|_1$ (sparsity),
- $R(x) = \lambda \|x\|_2^2$ (ridge),
- $R(x)$ is the indicator function of a convex set $\mathcal{X}$, i.e. $R(x)=0$ if $x \in \mathcal{X}$ and $+\infty$ otherwise — this encodes a hard constraint.

Define the proximal operator of $R$:
$$
\mathrm{prox}_{\alpha R}(y)
=
\arg\min_x
\left(
R(x) + \frac{1}{2\alpha} \|x-y\|_2^2
\right).
$$

Proximal gradient method:

1. Gradient step on $f$:
   $$
   y_k = x_k - \alpha \nabla f(x_k).
   $$
2. Proximal step on $R$:
   $$
   x_{k+1} = \mathrm{prox}_{\alpha R}(y_k).
   $$

This is also called forward–backward splitting: “forward” = gradient step, “backward” = prox step.

#### Interpretation:
- The prox step “handles” the nonsmooth or constrained part exactly.
- For $R(x)=\lambda \|x\|_1$, $\mathrm{prox}_{\alpha R}$ is soft-thresholding, which promotes sparsity in $x$.  
  This is the heart of $\ell_1$-regularised least-squares (LASSO) and many sparse recovery problems.
- For $R$ as an indicator of $\mathcal{X}$, $\mathrm{prox}_{\alpha R} = \Pi_\mathcal{X}$, so projected gradient is a special case of proximal gradient.

This unifies constraints and regularisation.

#### When to use proximal / projected gradient
- High-dimensional ML/statistics problems.
- Objectives with $\ell_1$, group sparsity, total variation, hinge loss, or indicator constraints.
- You can evaluate $\nabla f$ and compute $\mathrm{prox}_{\alpha R}$ cheaply.
- You don’t need absurdly high accuracy, but you do need scalability.

This is the standard tool for modern large-scale convex learning problems.


## 12.9 Penalties, barriers, and interior-point methods

So far we’ve assumed either:

- simple constraints we can project onto,
- or nonsmooth terms we can prox.

What if the constraints are general convex inequalities $g_i(x)\le0$: Enter penalty methods, barrier methods, and (ultimately) interior-point methods.

### 12.9.1 Penalty methods

Turn constrained optimisation into unconstrained optimisation by adding a penalty for violating constraints. Suppose we want
$$
\min_x f(x)
\quad \text{s.t.} \quad g_i(x) \le 0,\ i=1,\dots,m.
$$

A penalty method solves instead
$$
\min_x \; f(x) + \rho \sum_{i=1}^m \phi(g_i(x)),
$$
where:

- $\phi(r)$ is $0$ when $r \le 0$ (feasible),
- $\phi(r)$ grows when $r>0$ (infeasible),
- $\rho > 0$ is a penalty weight.

As $\rho \to \infty$, infeasible points become extremely expensive, so minimisers approach feasibility.  

This is conceptually simple and is sometimes effective, but:

- choosing $\rho$ is tricky,
- very large $\rho$ can make the landscape ill-conditioned and hard for gradient/Newton to solve.

### Algorithm: Basic Penalty Method (Quadratic or General Penalization)

Goal:  Solve  
$$
\min_x f(x) \quad \text{s.t. } g_i(x) \le 0,\; i=1,\dots,m.
$$

Penalty formulation:  
$$
F_\rho(x) = f(x) + \rho \sum_{i=1}^m \phi(g_i(x)),
$$
where  

- $\phi(r) = 0$ if $r \le 0$,  
- $\phi(r)$ grows when $r>0$ (e.g., $\phi(r)=\max\{0,r\}^2$),  
- $\rho > 0$ is the penalty weight.

Inputs:  

- objective $f(x)$  
- constraints $g_i(x)$  
- penalty function $\phi$  
- initial point $x_0$  
- initial penalty parameter $\rho_0 > 0$  
- penalty update factor $\gamma > 1$  
- tolerance $\varepsilon$


Procedure:

1. Choose $x_0$, $\rho_0 > 0$.  
2. For $k = 0, 1, 2, \dots$:  
      1. Solve the penalized subproblem  $x_{k+1} = \arg\min_x F_{\rho_k}(x)$ using Newton’s method, gradient descent, quasi-Newton, etc.  
       2. Check feasibility / stopping:  If $\max_i g_i(x_{k+1}) \le \varepsilon, \quad   \|x_{k+1} - x_k\| \le \varepsilon$  stop and return $x_{k+1}$.  
      3. Increase penalty parameter  $\rho_{k+1} = \gamma\, \rho_k$   with typical $\gamma \in [5,10]$.  
3. End.


### 12.9.2 Barrier methods

Penalty methods penalise violation *after* you cross the boundary. Barrier methods make it impossible to even touch the boundary. For inequality constraints $g_i(x) \le 0$, define the logarithmic barrier
$$
b(x) = - \sum_{i=1}^m \log(-g_i(x)).
$$
This is finite only if $g_i(x) < 0$ for all $i$, i.e. $x$ is strictly feasible. As you approach the boundary $g_i(x)=0$, $b(x)$ blows up to $+\infty$.

We then solve, for a sequence of increasing parameters $t$:
$$
\min_x \; F_t(x) := t f(x) + b(x),
$$
subject to strict feasibility $g_i(x)<0$.

As $t \to \infty$, minimisers of $F_t$ approach the true constrained optimum. The path of minimisers $x^*(t)$ is called the central path.

Key points:

- $F_t$ is smooth on the interior of the feasible region.
- We can apply Newton’s method to $F_t$.
- Each Newton step solves a linear system involving the Hessian of $F_t$, so the inner loop looks like a damped Newton method.
- Increasing $t$ tightens the approximation; we “home in” on the boundary of feasibility.


### Algorithm: Barrier Method (Logarithmic Barrier / Interior Approximation)

Goal: Solve the constrained problem  
$$
\min_x f(x) \quad \text{s.t. } g_i(x) \le 0,\; i=1,\dots,m.
$$

Logarithmic barrier:  
$$
b(x) = -\sum_{i=1}^m \log\!\big(-g_i(x)\big),
$$
defined only for strictly feasible points $g_i(x)<0$.

Barrier subproblem:  
$$
F_t(x) = t\, f(x) + b(x),
$$
where $t>0$ is the barrier parameter.

As $t \to \infty$, minimizers of $F_t$ approach the constrained optimum.

 Inputs:  

- objective $f(x)$  
- inequality constraints $g_i(x)$  
- barrier function $b(x)$  
- strictly feasible starting point $x_0$ ($g_i(x_0) < 0$)  
- initial barrier parameter $t_0 > 0$  
- barrier growth factor $\mu > 1$ (often $\mu = 10$)  
- tolerance $\varepsilon$

 
Procedure:

1. Choose strictly feasible $x_0$, and pick $t_0 > 0$.  
2. For $k = 0,1,2,\dots$:  
    1. Centering step (inner loop):  Solve the barrier subproblem  $
      x_{k+1} = \arg\min_x F_{t_k}(x)
      \quad\text{with} g_i(x)<0. $  Typically use Newton’s method (damped) on $F_{t_k}$.  Stop when the Newton decrement satisfies  $\lambda(x_{k+1})^2/2 \le \varepsilon$
      2. Optimality / stopping test:    If  $\frac{m}{t_k} \le \varepsilon,$
      then $x_{k+1}$ is an $\varepsilon$-approximate solution of the original constrained problem; stop and return $x_{k+1}$.  
      3. Increase barrier parameter:  $t_{k+1} = \mu\, t_k,$   which tightens the approximation and moves closer to the boundary.  
3. End.
 
### 12.9.3 Interior-point methods

Interior-point methods combine barrier functions with Newton’s method to solve general convex programs:

- They maintain strict feasibility throughout.
- Each iteration solves a Newton system for the barrier-augmented objective.
- They naturally generate primal–dual pairs and duality gap estimates.
- Under standard assumptions (e.g., Slater’s condition), they converge in a predictable number of iterations.

Interior-point methods are the foundation of modern solvers for LP, QP, SOCP, and SDP. They are more expensive per iteration than first-order methods but converge in far fewer steps and achieve high accuracy.

### Algorithm: Primal–Dual Interior-Point Method (for convex inequality constraints)

We consider the problem
$$
\min_x\; f(x) \quad \text{s.t. } g_i(x) \le 0,\; i=1,\dots,m.
$$

Introduce Lagrange multipliers $\lambda \ge 0$. The KKT conditions are
$$
\begin{aligned}
\nabla f(x) + \sum_i \lambda_i \nabla g_i(x) &= 0, \\
g_i(x) &\le 0, \\
\lambda_i &\ge 0, \\
\lambda_i\, g_i(x) &= 0.
\end{aligned}
$$

Interior-point methods enforce the relaxed condition
$$
\lambda_i\, g_i(x) = -\frac{1}{t},
$$
which keeps iterates strictly feasible.

 
### Inputs
- objective $f(x)$  
- inequality constraints $g_i(x)$  
- initial primal point $x_0$ with $g_i(x_0)<0$  
- initial dual variable $\lambda_0 > 0$  
- initial barrier parameter $t_0 > 0$  
- growth factor $\mu > 1$  
- tolerance $\varepsilon$



### Procedure

1. Choose strictly feasible $x_0$, positive $\lambda_0$, and $t_0$.

2. For $k = 0,1,2,\dots$:

      (a) Form the perturbed KKT system.  Solve for the Newton direction $(\Delta x, \Delta \lambda)$:

      $$
   \begin{bmatrix}
   \nabla^2 f(x) + \sum_i \lambda_i \nabla^2 g_i(x) & \nabla g(x) \\
   \text{diag}(\lambda)\,\nabla g(x)^\top & \text{diag}(g(x))
   \end{bmatrix}
   \begin{bmatrix}
   \Delta x \\
   \Delta \lambda
   \end{bmatrix}
   =
   -
   \begin{bmatrix}
   \nabla f(x) + \sum_i \lambda_i \nabla g_i(x) \\
   \lambda \circ g(x) + \tfrac{1}{t}\mathbf{1}
   \end{bmatrix}.
   $$

      (b) Line search to keep strict feasibility. Choose the maximum $\alpha\in(0,1]$ such that:
      
      - $g_i(x + \alpha \Delta x) < 0$,
      - $\lambda + \alpha \Delta \lambda > 0$.

      (c) Update: $x \leftarrow x + \alpha \Delta x,
   \qquad  \lambda \leftarrow \lambda + \alpha \Delta \lambda.$

      (d) Check duality gap: $\text{gap} = - g(x)^\top \lambda$ If $\text{gap} \le \varepsilon$, stop.

      (e) Increase barrier parameter $t \leftarrow \mu t.$

3. Return $x$.




## 12.10 Choosing the right method in practice


Case A. Smooth, unconstrained, very high dimensional.  
Example: logistic regression on millions of samples.  
Use: gradient descent or (better) accelerated gradient.  
Why: cheap iterations, easy to implement, scales.  
 
Case B. Smooth, unconstrained, moderate dimensional, need high accuracy.  
Example: convex nonlinear fitting with well-behaved Hessian.  
Use: Newton or quasi-Newton.  
Why: quadratic (or near-quadratic) convergence near optimum.  
 
Case C. Convex with simple feasible set $x \in \mathcal{X}$ (box, ball, simplex).  
Use: projected gradient.  
Why: projection is easy, maintains feasibility at each step.  
 
Case D. Composite objective $f(x) + R(x)$ where $R$ is nonsmooth (e.g. $\ell_1$, indicator of a constraint set).  
Use: proximal gradient.  
Why: prox handles nonsmooth/constraint part exactly each step.  
 
Case E. General convex program with inequalities $g_i(x)\le 0$.  
Use: interior-point methods.  
Why: they solve smooth barrier subproblems via Newton steps and give primal–dual certificates through KKT and duality (Chapters 7–8).  
  