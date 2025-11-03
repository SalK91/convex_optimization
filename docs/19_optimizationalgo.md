# Chapter 12: Algorithms for Convex Optimization
In Chapters 2–8 we built the mathematics of convex optimization: linear algebra (Chapter 2), gradients and Hessians (Chapter 3), convex sets (Chapter 4), convex functions (Chapter 5), subgradients (Chapter 6), KKT conditions (Chapter 7), and duality (Chapter 8). 

Now we answer the practical question:

How do we actually solve convex optimization problems in practice?

This chapter develops the major algorithmic families used to solve convex problems. Our goal is not only to describe each method, but to explain:

- what class of problem it solves,
- what information it needs (gradient, Hessian, projection, etc.),
- when you should use it,
- how it connects to the modelling choices you make.

 
## 9.1 Problem classes vs method classes

Before we dive into algorithms, we need a map. Different algorithms are natural for different convex problem structures.

### 9.1.1 Smooth unconstrained convex minimisation
We want
$$
\min_x f(x),
$$
where $f:\mathbb{R}^n \to \mathbb{R}$ is convex and differentiable.

Typical methods:

- Gradient descent (first-order),
- Accelerated gradient,
- Newton / quasi-Newton (second-order).

Information required:

- $\nabla f(x)$, sometimes $\nabla^2 f(x)$.

### 9.1.2 Smooth convex minimisation with simple constraints
We want
$$
\min_x f(x)
\quad \text{s.t.} \quad x \in \mathcal{X},
$$
where $\mathcal{X}$ is a “simple” closed convex set such as a box, a norm ball, or a simplex (Chapter 4).


#### Practical Examples of Simple Constraints

| Constraint Type | Explanation | Example | Meaning |
|------------------|--------------|----------|----------|
| Box | Each variable is bounded independently within lower and upper limits. | \( 0 \le x_i \le 1 \) | Parameters are restricted to a fixed range (e.g., pixel intensities, control limits). |
| Norm Ball | All feasible points lie within a fixed radius from a center under some norm. | \( \|x - x_0\|_2 \le 1 \) | Keeps the solution close to a reference point — controls total magnitude or deviation. |
| Simplex | Nonnegative variables that sum to one. | \( x_i \ge 0,\ \sum_i x_i = 1 \) | Represents valid probability distributions or normalized weights (e.g., portfolio allocations). |


Typical method:

- Projected gradient descent, which alternates a gradient step and Euclidean projection back to $\mathcal{X}$.


Information required:

- $\nabla f(x)$,
- the ability to compute $\Pi_\mathcal{X}(y) = \arg\min_{x \in \mathcal{X}} \|x-y\|_2^2$ efficiently.

### 9.1.3 Composite convex minimisation (smooth + nonsmooth)
We want
$$
\min_x \; F(x) := f(x) + R(x),
$$
where $f$ is convex and differentiable with Lipschitz gradient, and $R$ is convex but possibly nonsmooth (Chapter 6).  
Examples:

- $f(x)=\|Ax-b\|_2^2$, $R(x)=\lambda\|x\|_1$ (LASSO),
- $R(x)$ is the indicator of a convex set, enforcing a hard constraint.

Typical method:

- Proximal gradient / forward–backward splitting,
- Projected gradient as a special case.

Information required:

- $\nabla f(x)$,
- the proximal operator of $R$.

### 9.1.4 General convex programs with inequality constraints
We want
$$
\begin{array}{ll}
\text{minimise} & f(x) \\
\text{subject to} & g_i(x) \le 0,\quad i=1,\dots,m, \\
& h_j(x) = 0,\quad j=1,\dots,p,
\end{array}
$$
where $f$ and $g_i$ are convex, $h_j$ are affine.  

Typical method:

- Interior-point (barrier) methods.

Information required:

- Gradients and Hessians of the barrier-augmented objective,
- ability to solve linear systems arising from Newton steps.


### 9.1.5 The moral
There is no single “best” algorithm.  
There is a best algorithm for the structure you have.

- First-order methods scale to huge problems but converge relatively slowly.
- Newton and interior-point methods converge extremely fast in iterations but each iteration is more expensive (they solve linear systems involving Hessians).
- Proximal methods are designed for nonsmooth regularisers and constraints that appear everywhere in statistics and machine learning.
- Interior-point methods are the workhorse for general convex programs (including linear programs, quadratic programs, conic programs) and deliver high-accuracy solutions with strong certificates of optimality 


## 9.2 First-order methods: Gradient descent

### 9.2.1 Setting
We solve
$$
\min_x f(x),
$$
where $f$ is convex, differentiable, and (ideally) $L$-smooth: its gradient is Lipschitz with constant $L$, meaning
$$
\|\nabla f(x) - \nabla f(y)\|_2 \le L \|x-y\|_2 \quad \text{for all } x,y.
$$
Smoothness lets us control step sizes.

### 9.2.2 Algorithm
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
> - If we were to minimize just this linear model, we would move **infinitely far** in the direction of **steepest descent** $-\nabla f(x_t)$, which is not realistic or stable.

> This motivates adding a **locality restriction** — we trust the linear approximation **near** $x_t$, not globally. To prevent taking arbitrarily large steps, we add a quadratic penalty for moving away from $x_t$:

> $$
f(x) \approx f(x_t) + \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{2\eta} \|x - x_t\|^2,
$$

> where $\eta > 0$ is the **learning rate** or **step size**.

> - The linear term pulls $x$ in the steepest descent direction.
> - The quadratic term acts like a **trust region**, discouraging large deviations from $x_t$.
> - $\eta$ trades off **aggressive progress** vs **stability**:
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


### 9.2.3 Geometric meaning
From Chapter 3, the first-order Taylor model is
$$
f(x + d) \approx f(x) + \nabla f(x)^\top d.
$$
This is minimised (under a step length constraint) by taking $d$ in the direction $-\nabla f(x)$. So gradient descent is just “take a cautious step downhill”.

### 9.2.4 Convergence
For convex, $L$-smooth $f$, gradient descent with a suitable fixed step size satisfies
$$
f(x_k) - f^\star = O\!\left(\frac{1}{k}\right),
$$
where $f^\star$ is the global minimum. This $O(1/k)$ sublinear rate is slow compared to second-order methods, but each step is extremely cheap: you only need $\nabla f(x_k)$.

### 9.2.5 When to use gradient descent
- Problems with millions of variables (large-scale ML).
- You can afford many cheap iterations.
- You only have access to gradients (or stochastic gradients).
- You do not need very high precision.

Gradient descent is the baseline first-order method. But we can do better.

 
## 9.3 Accelerated first-order methods

Plain gradient descent has an $O(1/k)$ rate for smooth convex problems. Remarkably, we can do better — and in fact, provably optimal — by adding *momentum*.

### 9.3.1 Nesterov acceleration
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

### 9.3.2 Optimal first-order rate
For smooth convex $f$, accelerated gradient achieves
$$
f(x_k) - f^\star = O\!\left(\frac{1}{k^2}\right),
$$
which is *optimal* for any algorithm that uses only gradient information and not higher derivatives. In other words, you cannot beat $O(1/k^2)$ in the worst case using only first-order oracle calls.


### 9.3.3 When to use acceleration
- Same setting as gradient descent (large-scale smooth convex problems),
- but you want to converge in fewer iterations.
- You can tolerate a little more instability/parameter tuning (acceleration can overshoot if step sizes are not chosen carefully).

Acceleration is the default upgrade from vanilla gradient descent in many smooth convex machine learning problems.

 
## 9.4 Newton’s method and second-order methods

First-order methods only use gradient information. Newton’s method uses curvature (the Hessian) to take smarter steps.

### 9.4.1 Local quadratic model
From Chapter 3, the second-order Taylor approximation at $x_k$ is
$$
f(x_k + d)
\approx
f(x_k)
+ \nabla f(x_k)^\top d
+ \tfrac{1}{2} d^\top \nabla^2 f(x_k) d.
$$

If we (temporarily) trust this model, we choose $d$ to minimise the RHS. Differentiating w.r.t. $d$ and setting to zero gives the Newton step:
$$
\nabla^2 f(x_k) \, d_{\text{newton}}
= - \nabla f(x_k).
$$
So
$$
d_{\text{newton}} = - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k),
\quad
x_{k+1} = x_k + d_{\text{newton}}.
$$

### 9.4.2 Convergence behaviour
- Near the minimiser of a strictly convex, twice-differentiable $f$, Newton’s method converges quadratically: roughly, the number of correct digits doubles every iteration.
- This is dramatically faster than $O(1/k)$ or $O(1/k^2)$, but only once you’re in the “basin of attraction.”
- Far from the minimiser, Newton can misbehave, so we pair it with a line search or trust region.

### 9.4.3 Computational cost
Each Newton step requires solving a linear system involving $\nabla^2 f(x_k)$, which costs about as much as factoring the Hessian (or an approximation). This is expensive in very high dimensions, which is why Newton is most attractive for medium-scale problems where high accuracy matters.

### 9.4.4 Why convexity helps
If $f$ is convex, then $\nabla^2 f(x_k)$ is positive semidefinite (Chapter 5). This means:

- The quadratic model is bowl-shaped, so the Newton step makes sense.
- Regularised Newton steps (adding a multiple of the identity to the Hessian) behave very predictably.

### 9.4.5 Quasi-Newton
When Hessians are too expensive, we can build low-rank approximations of $\nabla^2 f(x_k)$ or its inverse. Famous examples include BFGS and L-BFGS. These methods keep much of Newton’s fast local convergence but with per-iteration cost closer to first-order methods.

### 9.4.6 When to use Newton / quasi-Newton
- You need high-accuracy solutions.
- The problem is smooth and reasonably well-conditioned.
- The dimension is moderate, or Hessian systems can be solved efficiently (e.g. via sparse linear algebra).


 

## 9.5 Constraints and nonsmooth terms: projection and proximal methods

In practice, most convex objectives are not just “nice smooth $f(x)$”. They often have:

- constraints $x \in \mathcal{X}$,
- nonsmooth regularisers like $\|x\|_1$,
- penalties that encode robustness or sparsity (Chapter 6).

Two core ideas handle this: projected gradient and proximal gradient.

### 9.5.1 Projected gradient descent

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

### 9.5.2 Proximal gradient (forward–backward splitting)

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


## 9.6 Penalties, barriers, and interior-point methods

So far we’ve assumed either:

- simple constraints we can project onto,
- or nonsmooth terms we can prox.

What if the constraints are general convex inequalities $g_i(x)\le0$?  
Enter penalty methods, barrier methods, and (ultimately) interior-point methods.

### 9.6.1 Penalty methods

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

### 9.6.2 Barrier methods

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

### 9.6.3 Interior-point methods in practice

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


**Summary: Penalty vs Barrier vs Interior-Point**

| Method | Feasibility During Iteration | Mechanism | Typical Behavior |
|--------|------------------------------|------------|------------------|
| Penalty | May violate constraints | Adds large penalty outside feasible region | Easy to implement but can be ill-conditioned |
| Barrier | Stays strictly feasible | Adds infinite cost near constraint boundary | Smooth approximation to constrained problem |
| Interior-Point | Always feasible (uses barrier) | Solves a sequence of barrier problems with increasing precision | Follows central path to true optimum |



## 9.8 Choosing the right method in practice

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
  