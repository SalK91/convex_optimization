# Chapter 12: Algorithms for Convex Optimization
In the previous chapters, we built the mathematical foundations of convex optimization — convex sets, convex functions, gradients, subgradients, KKT conditions, and duality. Now we answer the practical question: How do we actually solve convex optimization problems in practice?

This chapter surveys the main algorithmic families for convex optimization — how they work, what problems they solve, and how they connect to learning and modeling principles.  

 
## 12.1 Problem classes vs method classes

Different convex problems call for different algorithmic structures.  
Here is the broad landscape:

| Problem Type | Typical Formulation | Representative Methods | Examples |
|---------------|--------------------|-------------------------|-----------|
| Smooth, unconstrained | $\min_x f(x)$, convex and differentiable | Gradient descent, Accelerated gradient, Newton | Logistic regression, least squares |
| Smooth with simple constraints | $\min_x f(x)$ s.t. $x \in \mathcal{X}$ (box, ball, simplex) | Projected gradient | Constrained regression, probability simplex |
| Composite convex (smooth + nonsmooth) | $\min_x f(x) + R(x)$ | Proximal gradient, coordinate descent | Lasso, Elastic Net, TV minimization |
| General constrained convex | $\min f(x)$ s.t. $g_i(x) \le 0, h_j(x)=0$ | Interior-point, primal–dual methods | LP, QP, SDP, SOCP |


     

## 12.2 First-order methods: Gradient descent

### 12.2.1 Setting
We solve
$$
\min_x f(x),
$$
where $f$ is convex, differentiable, and (ideally) $L$-smooth: its gradient is Lipschitz with constant $L$, meaning
$$
\|\nabla f(x) - \nabla f(y)\|_2 \le L \|x-y\|_2 \quad \text{for all } x,y.
$$
Smoothness lets us control step sizes.

### 12.2.2 Algorithm
Gradient descent iterates
$$
x_{k+1} = x_k - \alpha_k \nabla f(x_k),
$$
where $\alpha_k>0$ is the step size (also called learning rate in machine learning). A common choice is a constant $\alpha_k = 1/L$ when $L$ is known, or a backtracking line search when it is not.


> Derivation: Around $x_t$, we approximate $f$ using its Taylor expansion:

> $$
f(x) \approx f(x_t) + \langle \nabla f(x_t), x - x_t \rangle.
$$


> - We assume $f$ behaves approximately like its tangent plane near $x_t$.  
> - If we were to minimize just this linear model, we would move infinitely far in the direction of steepest descent $-\nabla f(x_t)$, which is not realistic or stable.

> This motivates adding a locality restriction — we trust the linear approximation near $x_t$, not globally. To prevent taking arbitrarily large steps, we add a quadratic penalty for moving away from $x_t$:

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


### 12.2.3 Geometric meaning
From Chapter 3, the first-order Taylor model is
$$
f(x + d) \approx f(x) + \nabla f(x)^\top d.
$$
This is minimised (under a step length constraint) by taking $d$ in the direction $-\nabla f(x)$. So gradient descent is just “take a cautious step downhill”.

### 12.2.4 Convergence
For convex, $L$-smooth $f$, gradient descent with a suitable fixed step size satisfies
$$
f(x_k) - f^\star = O\!\left(\frac{1}{k}\right),
$$
where $f^\star$ is the global minimum. This $O(1/k)$ sublinear rate is slow compared to second-order methods, but each step is extremely cheap: you only need $\nabla f(x_k)$.

### 12.2.5 When to use gradient descent
- Problems with millions of variables (large-scale ML).
- You can afford many cheap iterations.
- You only have access to gradients (or stochastic gradients).
- You do not need very high precision.

Gradient descent is the baseline first-order method. But we can do better.

 
## 12.3 Accelerated first-order methods

Plain gradient descent has an $O(1/k)$ rate for smooth convex problems. Remarkably, we can do better — and in fact, provably optimal — by adding *momentum*.

### 12.3.1 Nesterov acceleration
Nesterov’s accelerated gradient method modifies the update using a momentum-like extrapolation. One common presentation is:

1. Maintain two sequences $x_k$ and $y_k$.
2. Take a gradient step from $y_k$:
   $$
   x_{k+1} = y_k - \alpha \nabla f(y_k).
   $$
3. Extrapolate:
   $$
   y_{k+1} = x_{k+1} + \beta_k (x_{k+1} - x_k).
   $$

The extra $\beta_k$ term “looks ahead,” helping the method exploit curvature better than plain gradient descent.

### 12.3.2 Optimal first-order rate
For smooth convex $f$, accelerated gradient achieves
$$
f(x_k) - f^\star = O\!\left(\frac{1}{k^2}\right),
$$
which is *optimal* for any algorithm that uses only gradient information and not higher derivatives. In other words, you cannot beat $O(1/k^2)$ in the worst case using only first-order oracle calls.


### 12.3.3 When to use acceleration
- Same setting as gradient descent (large-scale smooth convex problems),
- but you want to converge in fewer iterations.
- You can tolerate a little more instability/parameter tuning (acceleration can overshoot if step sizes are not chosen carefully).

Acceleration is the default upgrade from vanilla gradient descent in many smooth convex machine learning problems.


The convergence of gradient descent depends strongly on the geometry of the level sets of the objective function. When these level sets are poorly conditioned—that is, highly anisotropic or elongated (not spherical)—the gradient directions tend to oscillate across narrow valleys, leading to zig-zag behavior and slow convergence.

In contrast, when the level sets are well-conditioned (approximately spherical), gradient descent progresses efficiently toward the minimum. Thus, the efficiency of gradient-based methods is governed by how aspherical (anisotropic) the level sets are, which is directly related to the condition number of the Hessian.

## 12.4 Steepest Descent Method
 
The steepest descent method generalizes gradient descent by depending on the choice of norm used to measure step size or direction. It finds the direction of *maximum decrease* of the objective function under a unit norm constraint.


> The norm defines the “geometry” of optimization.
> Gradient descent is steepest descent under the Euclidean norm.
> Changing the norm changes what “steepest” means, and can greatly affect convergence, especially for ill-conditioned or anisotropic problems.
 
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

## 12.5 Conjugate Gradient Method — Efficient Optimization for Quadratic Objectives

Gradient descent can be slow when the objective’s level sets are highly elongated —  
a symptom of ill-conditioning in the Hessian.  
For quadratic functions of the form

$$
f(x) = \tfrac{1}{2} x^\top A x - b^\top x, \quad A \succ 0,
$$

plain gradient descent takes many small steps along shallow directions of $A$.

The Conjugate Gradient (CG) method accelerates convergence dramatically for such problems — it exploits the structure of the quadratic and uses curvature-aware search directions without explicitly forming or inverting the Hessian.


### Problem Setup

Minimize a strictly convex quadratic:

$$
\min_x f(x) = \tfrac{1}{2} x^\top A x - b^\top x, \quad A \in \mathbb{R}^{n \times n}, \; A \succ 0.
$$

This is equivalent to solving the linear system

$$
A x = b.
$$


### Algorithm (Linear CG)

Given an initial $x_0$, define the residual $r_0 = b - A x_0$  
and the initial direction $p_0 = r_0$.

For $k = 0, 1, 2, \dots$ until convergence:

1. Compute step size  
   $$
   \alpha_k = \frac{r_k^\top r_k}{p_k^\top A p_k}.
   $$
2. Update the iterate  
   $$
   x_{k+1} = x_k + \alpha_k p_k.
   $$
3. Update the residual  
   $$
   r_{k+1} = r_k - \alpha_k A p_k.
   $$
4. Compute the new direction coefficient  
   $$
   \beta_k = \frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k}.
   $$
5. Update the direction  
   $$
   p_{k+1} = r_{k+1} + \beta_k p_k.
   $$

Terminate when $\|r_k\|$ is below tolerance $\varepsilon$.


### Geometric Intuition

Each search direction $p_k$ is $A$-conjugate to the previous ones:

$$
p_i^\top A p_j = 0 \quad \text{for } i \ne j.
$$

That means successive steps explore independent curvature directions of the quadratic. The residual $r_k$ (the negative gradient) becomes orthogonal to all previous directions, so the method never re-searches the same subspace.

As a result, in exact arithmetic, CG finds the exact minimizer in at most $n$ steps.


### Convergence Properties

- For SPD $A$, CG converges monotonically to the minimizer $x^\star = A^{-1} b$.
- The rate depends on the condition number $\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}}$:
  $$
  \|x_k - x^\star\|_A \le 2 \left( \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} \right)^k \|x_0 - x^\star\|_A.
  $$
- Preconditioning (using $M^{-1}A$ with well-chosen $M$) further improves convergence.


### Machine Learning Context

In ML, CG is widely used for large-scale convex quadratic subproblems:

| Application | Formulation | Notes |
|--------------|--------------|-------|
| Ridge regression | $\min_x \|A x - b\|_2^2 + \lambda\|x\|_2^2$ | Normal equations are SPD; CG avoids explicit inversion. |
| Kernel ridge regression | $(K + \lambda I)\alpha = y$ | CG solves this efficiently without forming full $K^{-1}$. |
| Linear least squares | $\min_x \tfrac{1}{2}\|A x - b\|^2$ | Equivalent to solving $A^\top A x = A^\top b$. |
| Large-scale Newton steps | Solve $\nabla^2 f(x_k)p = -\nabla f(x_k)$ | CG acts as an inner solver for the Newton direction. |



### Practical Notes

- CG requires only matrix–vector products with $A$, not explicit storage.  
  It’s ideal when $A$ is large, sparse, or implicitly defined.
- Sensitive to rounding errors; residual re-orthogonalization may be needed for long runs.
- Preconditioners (Jacobi, incomplete Cholesky, etc.) can drastically reduce iterations.

### Comparison Summary

| Method | Memory | Curvature Use | Convergence | Typical Use |
|--------|---------|----------------|--------------|--------------|
| Gradient Descent | $O(n)$ | None | $O(1/k)$ | General smooth convex |
| Newton’s Method | $O(n^2)$ | Full Hessian | Quadratic (local) | Small/medium convex |
| Conjugate Gradient | $O(n)$ | Implicit (via $A$-conjugacy) | Fast linear / finite-step | Large quadratic systems |

-

### Key Insight

> The Conjugate Gradient method is the exact gradient method for quadratic objectives  
> that automatically builds curvature information through orthogonalized directions,  
> achieving Newton-like efficiency without forming the Hessian.


## 12.6 Newton’s method and second-order methods

First-order methods (like gradient descent) only use gradient information. Newton’s method, in contrast, incorporates curvature information from the Hessian to take steps that better adapt to the local geometry of the function. This often leads to much faster convergence near the optimum.



### 12.6.1 Local quadratic model

From Chapter 3, the second-order Taylor approximation of $f(x)$ around a point $x_k$ is:

$$
f(x_k + d)
\approx
f(x_k)
+ \nabla f(x_k)^\top d
+ \tfrac{1}{2} d^\top \nabla^2 f(x_k) d.
$$

If we temporarily trust this quadratic model, we can choose $d$ to minimize the right-hand side.  
Differentiating with respect to $d$ and setting to zero gives:

$$
\nabla^2 f(x_k) \, d_{\text{newton}} = - \nabla f(x_k).
$$

Hence, the Newton step is:

$$
d_{\text{newton}} = - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k),
\quad
x_{k+1} = x_k + d_{\text{newton}}.
$$

This step points toward the minimizer of the local quadratic model, and near the true minimizer, Newton’s method exhibits quadratic convergence.



### 12.6.2 Convergence behaviour

- Near the minimiser of a strictly convex, twice-differentiable $f$, Newton’s method converges quadratically: roughly, the number of correct digits doubles every iteration.  
- This is dramatically faster than the $O(1/k)$ or $O(1/k^2)$ rates typical of first-order methods — but only once the iterates enter the basin of attraction.  
- Far from the minimiser, Newton’s method can behave erratically or even diverge.  
  To stabilise it, we typically pair it with a line search or trust region strategy to control step size.



### 12.6.3 Implementation

The main computational effort in each iteration lies in evaluating derivatives and solving the Newton system:

$$
H \, \Delta x = -g,
$$

where

$$
H = \nabla^2 f(x), \quad g = \nabla f(x).
$$

#### Solving via Cholesky factorization

If $H$ is symmetric and positive definite, we can efficiently solve this system using a Cholesky factorization:

$$
H = L L^{\top},
$$

where $L$ is lower triangular.  
The Newton step is then:

$$
\Delta x_{\text{nt}} = -L^{-\top} L^{-1} g.
$$

This involves two triangular solves:

1. $L y = -g$
2. $L^{\top} \Delta x_{\text{nt}} = y$

This avoids explicitly computing $H^{-1}$ and ensures numerical stability.

#### Newton decrement

A useful measure of progress is the Newton decrement:

$$
\lambda(x) = \| L^{-1} g \|_2,
$$

which approximates how far we are from the optimum.  
A common stopping criterion is $\lambda(x)^2 / 2 < \varepsilon$.



### 12.6.4 Computational cost

Each Newton step requires solving a linear system involving $\nabla^2 f(x_k)$, which costs about as much as factoring the Hessian (or an approximation).

- For an unstructured, dense Hessian, Cholesky factorization requires approximately $(1/3) n^3$ floating-point operations.  
- If $H$ is sparse, banded, or has special structure, the cost can be much lower.  
- Because of this cubic scaling, Newton’s method is most attractive for medium-scale problems where high accuracy is required.



### 12.6.5 Why convexity helps

If $f$ is convex, then $\nabla^2 f(x_k)$ is positive semidefinite (Chapter 5).  
This has two important implications:

- The local quadratic model is bowl-shaped, so the Newton direction points toward a minimiser.  
- Regularised Newton steps (e.g. using $H + \mu I$ for small $\mu > 0$) are guaranteed to be descent directions and behave predictably.



### 12.6.6 Quasi-Newton methods

When computing or storing the Hessian is too expensive, we can build low-rank approximations of $\nabla^2 f(x_k)$ or its inverse.  
These methods use gradient information from previous steps to estimate curvature.

The most famous examples are:

- BFGS (Broyden–Fletcher–Goldfarb–Shanno)  
- DFP (Davidon–Fletcher–Powell)  
- L-BFGS (Limited-memory BFGS) — for very large-scale problems.

Quasi-Newton methods (BFGS, L-BFGS) build inverse-Hessian approximations from gradient differences, achieving superlinear convergence with low memory

They maintain many of Newton’s fast local convergence properties, but with per-iteration costs similar to first-order methods.

For instance, BFGS maintains an approximation $B_k \approx \nabla^2 f(x_k)^{-1}$ updated via gradient and step differences:

$$
B_{k+1} = B_k + \frac{(s_k^\top y_k + y_k^\top B_k y_k)}{(s_k^\top y_k)^2} s_k s_k^\top
- \frac{B_k y_k s_k^\top + s_k y_k^\top B_k}{s_k^\top y_k},
$$

where $s_k = x_{k+1} - x_k$ and $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$.

These methods achieve superlinear convergence in practice, making them popular for large smooth optimization problems.



### 12.6.7 When to use Newton or quasi-Newton methods

Use Newton or quasi-Newton methods when:

- You need high-accuracy solutions.  
- The problem is smooth and reasonably well-conditioned.  
- The dimension is moderate, or Hessian systems can be solved efficiently (e.g., using sparse linear algebra).  

For large, ill-conditioned, or nonsmooth problems, first-order or proximal methods (Chapter 10) are typically more suitable.




> Newton-Raphson: The Newton step solves $\nabla^2 f(x_k) p_k=-\nabla f(x_k)$ and updates $x_{k+1}=x_k+p_k$ with line search or trust-region safeguards. Complexity hinges on solving linear systems; use sparse Cholesky, conjugate gradients with preconditioning, or low-rank structure to scale. For generalized linear models, iteratively reweighted least squares converges in few iterations, but regularization and damping are needed when data are nearly separable.

> Gauss-Newton: For nonlinear least squares $f(x)=\tfrac12\|r(x)\|^2$, the Gauss–Newton approximation uses $H\approx J^\top J$ where $J$ is the Jacobian of $r$. Solve $(J^\top J)\Delta=-J^\top r$ to get a step; Levenberg–Marquardt adds damping $(J^\top J+\lambda I)\Delta=-J^\top r$ interpolating between gradient and Gauss–Newton. Effective for residual models where second-order residual terms are small; widely used in curve fitting and some deep learning layerwise updates.

## 12.7 Constraints and nonsmooth terms: projection and proximal methods

In practice, most convex optimization problems are not purely smooth.  
They often include:

- Constraints: $x \in \mathcal{X}$,
- Nonsmooth regularisers: such as $\|x\|_1$,
- Penalties: promoting robustness or sparsity (see Chapter 6).

Two core strategies handle such settings:

1. Projected gradient methods — where we project each iterate back into the feasible set $\mathcal{X}$.  
2. Proximal gradient methods — which generalize projection to handle nonsmooth but structured terms.

These methods extend the ideas of gradient and Newton updates to the broader world of constrained and composite optimization.

### 12.7.2 Convergence behaviour
- Near the minimiser of a strictly convex, twice-differentiable $f$, Newton’s method converges quadratically: roughly, the number of correct digits doubles every iteration.
- This is dramatically faster than $O(1/k)$ or $O(1/k^2)$, but only once you’re in the “basin of attraction.”
- Far from the minimiser, Newton can misbehave, so we pair it with a line search or trust region.

### 12.7.3 Computational cost
Each Newton step requires solving a linear system involving $\nabla^2 f(x_k)$, which costs about as much as factoring the Hessian (or an approximation). This is expensive in very high dimensions, which is why Newton is most attractive for medium-scale problems where high accuracy matters.

### 12.7.4 Why convexity helps
If $f$ is convex, then $\nabla^2 f(x_k)$ is positive semidefinite (Chapter 5). This means:

- The quadratic model is bowl-shaped, so the Newton step makes sense.
- Regularised Newton steps (adding a multiple of the identity to the Hessian) behave very predictably.

### 12.7.5 Quasi-Newton
When Hessians are too expensive, we can build low-rank approximations of $\nabla^2 f(x_k)$ or its inverse. Famous examples include BFGS and L-BFGS. These methods keep much of Newton’s fast local convergence but with per-iteration cost closer to first-order methods.

### 12.7.6 When to use Newton / quasi-Newton
- You need high-accuracy solutions.
- The problem is smooth and reasonably well-conditioned.
- The dimension is moderate, or Hessian systems can be solved efficiently (e.g. via sparse linear algebra).


 

## 12.8 Constraints and nonsmooth terms: projection and proximal methods

In practice, most convex objectives are not just “nice smooth $f(x)$”. They often have:

- constraints $x \in \mathcal{X}$,
- nonsmooth regularisers like $\|x\|_1$,
- penalties that encode robustness or sparsity (Chapter 6).

Two core ideas handle this: projected gradient and proximal gradient.

### 12.8.1 Projected gradient descent

Setting:  
Minimise convex, differentiable $f(x)$ subject to $x \in \mathcal{X}$, where $\mathcal{X}$ is a simple closed convex set (Chapter 4).

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

Setting:  
Composite convex minimisation
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

What if the constraints are general convex inequalities $g_i(x)\le0$?  
Enter penalty methods, barrier methods, and (ultimately) interior-point methods.

### 12.9.1 Penalty methods

Turn constrained optimisation into unconstrained optimisation by adding a penalty for violating constraints.

Suppose we want
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

Penalty methods are closely linked to robust formulations and Huber-like losses: you replace a hard requirement by a soft cost. This is exactly what you do in robust regression and in $\epsilon$-insensitive / Huber losses (see Section 9.7).

### 12.9.2 Barrier methods

Penalty methods penalise violation *after* you cross the boundary. Barrier methods make it impossible to even touch the boundary.

For inequality constraints $g_i(x) \le 0$, define the logarithmic barrier
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

This is the core idea of interior-point methods.

### 12.9.3 Interior-point methods in practice

Interior-point methods:

- Are globally convergent for convex problems under mild assumptions (Slater’s condition; see Chapter 8).
- Solve a series of smooth, strictly feasible subproblems.
- Use Newton-like steps to update primal (and, implicitly, dual) variables.
- Produce both primal and dual iterates — so they naturally produce a duality gap, which certifies how close you are to optimality (Chapter 8).

Interior-point methods are the engine behind modern general-purpose convex solvers for:

- linear programs (LP),
- quadratic programs (QP),
- second-order cone programs (SOCP),
- semidefinite programs (SDP).

They give high-accuracy answers and KKT-based optimality certificates. They are more expensive per iteration than gradient methods, but need far fewer iterations, and they handle fully general convex constraints.


Summary: Penalty vs Barrier vs Interior-Point

| Method | Feasibility During Iteration | Mechanism | Typical Behavior |
|--||||
| Penalty | May violate constraints | Adds large penalty outside feasible region | Easy to implement but can be ill-conditioned |
| Barrier | Stays strictly feasible | Adds infinite cost near constraint boundary | Smooth approximation to constrained problem |
| Interior-Point | Always feasible (uses barrier) | Solves a sequence of barrier problems with increasing precision | Follows central path to true optimum |



## 12.10 Choosing the right method in practice

Let’s summarise the chapter in the form of a decision guide.

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
  