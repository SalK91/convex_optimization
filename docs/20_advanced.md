# Chapter 10: Advanced Large-Scale and Structured Methods

In Chapter 9 we focused on “classical convex solvers”: gradient methods, accelerated methods, Newton and quasi-Newton methods, projected/proximal methods, and interior-point methods. Those are the canonical tools of convex optimisation.

This chapter moves one step further.

Here we study methods that:
- exploit **problem structure** (sparsity, separability, block structure),
- scale to extremely high dimensions,
- or are widely used in practice for machine learning and signal processing — including in problems that are not convex.

Some of these methods were first analysed in the convex setting (often with strong guarantees), and then adopted — sometimes recklessly — in the nonconvex world (training neural nets, matrix factorisation, etc.). You’ll absolutely see them in modern optimisation and ML code.

We’ll cover:
1. Coordinate (block) descent,
2. Stochastic gradient and mini-batch methods,
3. ADMM (Alternating Direction Method of Multipliers),
4. Proximal coordinate / coordinate proximal variants,
5. Majorization–minimization and iterative reweighted schemes.

Throughout we’ll emphasise:
- When they are provably correct for convex problems,
- Why people also use them in nonconvex problems.

---

## 10.1 Coordinate descent and block coordinate descent

### 10.1.1 Idea

Instead of updating **all** coordinates of $x$ at once using a full gradient or Newton direction, we update **one coordinate (or one block of coordinates)** at a time, holding the others fixed.

Suppose we want to minimise a convex function
$$
\min_x F(x),
$$
and write $x = (x_1, x_2, \dots, x_p)$ in coordinates or blocks.  
Coordinate descent cycles through $i = 1,2,\dots,p$ and solves (or approximately solves)
$$
x_i^{(k+1)}
=
\arg\min_{z} \; F\big(x_1^{(k+1)}, \dots, x_{i-1}^{(k+1)}, z, x_{i+1}^{(k)}, \dots, x_p^{(k)}\big).
$$

In other words: update coordinate $i$ by optimising over just that coordinate (or block), treating the rest as constants.

### 10.1.2 Why this can be fast

- Each subproblem is often 1D (or low-dimensional), so it may have a closed form.
- For problems with separable structure — e.g. sums over features, or regularisers like $\|x\|_1 = \sum_i |x_i|$ — the coordinate update is extremely cheap.
- You never form the full gradient or solve a large linear system; you just operate on pieces.

This is especially attractive in high dimensions (millions of features), where a full Newton step would be absurdly expensive.

### 10.1.3 Convergence in convex problems

For many convex, continuously differentiable problems with certain regularity (e.g. strictly convex objective, or convex plus separable nonsmooth terms), cyclic coordinate descent is guaranteed to converge to the global minimiser. There are also randomized versions that pick a coordinate uniformly at random, which often give cleaner expected-rate guarantees.

For $\ell_1$-regularised least squares, i.e.
$$
\min_x \; \tfrac12 \|Ax - b\|_2^2 + \lambda \|x\|_1,
$$
each coordinate update becomes a scalar soft-thresholding step — so coordinate descent becomes an extremely efficient sparse regression solver.

### 10.1.4 Block coordinate descent

When coordinates are naturally grouped (for example, $x$ is really $(x^{(1)}, x^{(2)}, \dots)$ where each $x^{(j)}$ is a vector of parameters for a submodule or layer), we generalise to **block coordinate descent**. Each step solves
$$
x^{(j)} \leftarrow \arg\min_{z} F(\dots, z, \dots)\,.
$$

Block coordinate descent is the backbone of many alternating minimisation schemes in signal processing, matrix factorisation, dictionary learning, etc.

### 10.1.5 Use in nonconvex problems

Even when $F$ is not convex, people still run block coordinate descent (under names like “alternating minimisation” or “alternating least squares”), because:

- each block subproblem might be convex even if the joint problem isn’t,
- it is easy to implement,
- it often works “well enough” in practice.

You see this in low-rank matrix factorisation (recommender systems), where you fix all user factors and update item factors, then swap. There are no global guarantees in general (no convexity), but empirically it converges to useful solutions.

So:  
- In convex settings → provable global convergence.  
- In nonconvex settings → heuristic that often finds acceptable stationary points.

---

## 10.2 Stochastic gradient and mini-batch methods

### 10.2.1 Full gradient vs stochastic gradient

In Chapter 9, gradient descent uses the full gradient $\nabla f(x)$ at each step. In large-scale learning problems, $f$ is almost always an average over data:
$$
f(x) = \frac{1}{N} \sum_{i=1}^N \ell_i(x),
$$
where $\ell_i$ is the loss on sample $i$.

Computing $\nabla f(x)$ exactly costs $O(N)$ per step, which is huge.

**Stochastic Gradient Descent (SGD)** replaces $\nabla f(x)$ with an unbiased estimate. At each iteration we:

1. Sample $i$ uniformly from $\{1,\dots,N\}$,
2. Use $g_k = \nabla \ell_i(x_k)$,
3. Update
   $$
   x_{k+1} = x_k - \alpha_k g_k.
   $$

This is extremely cheap: one data point (or a small mini-batch) per step.

### 10.2.2 Convergence in convex problems

For convex problems, with diminishing step sizes $\alpha_k$, SGD converges to the global optimum in expectation, and more refined analyses show $O(1/\sqrt{k})$ suboptimality rates for general convex Lipschitz losses, improving to $O(1/k)$ in strongly convex smooth cases with appropriate averaging.

That is slower (per iteration) than deterministic gradient descent in theory, but each iteration is *much* cheaper. So SGD wins in wall-clock time for huge $N$.

### 10.2.3 Momentum, Adam, RMSProp (nonconvex practice, convex roots)

In modern machine learning, methods like momentum SGD, Adam, RMSProp, Adagrad, etc., are used routinely to train enormous nonconvex models (deep networks). These are variations of first-order methods with:

- adaptive step sizes,
- running averages of squared gradients,
- momentum terms.

While the most common use is for nonconvex problems, many of these methods (e.g. Adagrad-type adaptive steps, momentum acceleration) have their theoretical roots in convex optimisation and mirror-descent style analyses.

So stochastic first-order methods are:

- rigorous for convex problems,
- widely used heuristically for nonconvex problems.

---

## 10.3 ADMM: Alternating Direction Method of Multipliers

ADMM is one of the most important algorithms in modern convex optimisation for structured problems. It is used constantly in signal processing, sparse learning, distributed optimisation, and large-scale statistical estimation.

### 10.3.1 Problem form

ADMM solves problems of the form
$$
\min_{x,z} \; f(x) + g(z)
\quad
\text{subject to} \quad
Ax + Bz = c,
$$
where $f$ and $g$ are convex.

This form appears everywhere:

- $f$ is a data-fit term,
- $g$ is a regulariser or constraint indicator,
- $Ax + Bz = c$ ties them together.

For example, LASSO can be written by introducing a copy variable and enforcing $x=z$.

### 10.3.2 Augmented Lagrangian

ADMM applies the augmented Lagrangian method, which is like dual ascent but with a quadratic penalty on constraint violation. The augmented Lagrangian is
$$
\mathcal{L}_\rho(x,z,y)
=
f(x) + g(z)
+ y^\top (Ax + Bz - c)
+ \frac{\rho}{2} \|Ax + Bz - c\|_2^2,
$$
with dual variable (Lagrange multiplier) $y$ and penalty parameter $\rho>0$.

### 10.3.3 The ADMM updates (two-block case)

Iterate the following:
1. **$x$-update:**
   $$
   x^{k+1}
   :=
   \arg\min_x \mathcal{L}_\rho(x, z^k, y^k)
   $$
   (holding $z,y$ fixed).
2. **$z$-update:**
   $$
   z^{k+1}
   :=
   \arg\min_z \mathcal{L}_\rho(x^{k+1}, z, y^k).
   $$
3. **Dual update:**
   $$
   y^{k+1}
   :=
   y^k
   + \rho (A x^{k+1} + B z^{k+1} - c).
   $$

That is: optimise $x$ given $z$, optimise $z$ given $x$, then update the multiplier.

### 10.3.4 Why ADMM is powerful

- Each subproblem often becomes simple and separable:
  - The $x$-update might be a least-squares or a smooth convex minimisation,
  - The $z$-update might be a proximal operator (soft-thresholding, projection, etc.).
- You never have to solve the full coupled problem in one shot.
- ADMM is embarrassingly parallel / distributable: different blocks can be solved on different machines then averaged via the multiplier step.

### 10.3.5 Convergence

For convex $f$ and $g$, under mild assumptions (closed proper convex functions, some regularity), ADMM converges to a solution of the primal problem, and the dual variable $y^k$ converges to an optimal dual multiplier (Boyd and Vandenberghe, 2004, Ch. 5; also classical ADMM literature).

This is deeply tied to duality (Chapter 8): ADMM is best understood as a method of solving the dual with decomposability, but returning primal iterates along the way.

### 10.3.6 Use in nonconvex problems

In practice, ADMM is often extended to nonconvex problems by simply “pretending it’s fine.” Each subproblem is solved anyway, and the dual variable is updated the same way. The method is no longer guaranteed to find a global minimiser — but it often finds a stationary point that is good enough (e.g. in nonconvex regularised matrix completion, dictionary learning, etc.).

You will see ADMM used in imaging, sparse coding, variational inference, etc., even when parts of the model are not convex.

---

## 10.4 Proximal coordinate and coordinate-prox methods

There’s a natural fusion of the ideas in Sections 10.1 (coordinate descent) and 9.5 (proximal methods): **proximal coordinate descent**.

### 10.4.1 Problem form

Consider composite convex objectives
$$
F(x) = f(x) + R(x),
$$
with $f$ smooth convex and $R$ convex, possibly nonsmooth and separable across coordinates or blocks:
$$
R(x) = \sum_{j=1}^p R_j(x_j).
$$

### 10.4.2 Algorithm sketch

At each iteration, pick coordinate (or block) $j$, and update only $x_j$ by solving the 1D (or low-dim) proximal subproblem:
$$
x_j^{(k+1)}
=
\arg\min_{z}
\left[
\underbrace{
f\big(x_1^{(k+1)}, \dots, x_{j-1}^{(k+1)}, z, x_{j+1}^{(k)}, \dots \big)
}_{\text{local linear/quadratic approximation}}
+ R_j(z)
\right].
$$

Often we linearise $f$ around the current point in that block and add a quadratic term, just like a proximal gradient step but on one coordinate at a time.

### 10.4.3 Why it’s useful

- When $R$ is separable (e.g. $\ell_1$ sparsity penalties), each coordinate subproblem becomes a scalar shrinkage / thresholding step.
- Memory footprint is tiny.
- You get sparsity “for free” as many coordinates get driven to zero and stay there.
- Randomised versions (pick a coordinate at random) are simple and have good expected convergence guarantees in convex problems.

### 10.4.4 Use in nonconvex settings

People run proximal coordinate descent in nonconvex sparse learning (e.g. $\ell_0$-like surrogates, nonconvex penalties for variable selection). The convex convergence guarantees are gone, but empirically the method still often converges to a structured, interpretable solution.

---

## 10.5 Majorization–minimization (MM) and reweighted schemes

Majorization–minimization (MM) is a general pattern:

1. Build a simple convex surrogate that upper-bounds (majorises) your objective at the current iterate,
2. Minimise the surrogate,
3. Repeat.

It is sometimes called “iterative reweighted” or “successive convex approximation.”

### 10.5.1 MM template

Suppose we want to minimise $F(x)$ (convex or not). We construct $G(x \mid x^{(k)})$ such that:

- $G(x^{(k)} \mid x^{(k)}) = F(x^{(k)})$ (touches at current iterate),
- $G(x \mid x^{(k)}) \ge F(x)$ for all $x$ (majorises $F$),
- $G(\cdot \mid x^{(k)})$ is easy to minimise (often convex, often separable).

Then we set
$$
x^{(k+1)} \in \arg\min_x G(x \mid x^{(k)}).
$$

This guarantees $F(x^{(k+1)}) \le F(x^{(k)})$. So the objective is monotonically nonincreasing.

### 10.5.2 Iterative reweighted $\ell_1$ / $\ell_2$

A classical example: to promote sparsity or robustness, you might want to minimise something like
$$
\sum_i w_i(x) \, |x_i|
$$
or a concave penalty on residuals. You replace that concave / nonconvex penalty with a weighted convex penalty that depends on the previous iterate. Then you update the weights and solve again.

In the convex world, MM is just another way to design descent methods.  
In the nonconvex world, MM is a way to attack nonconvex penalties using a sequence of convex subproblems.

This is extremely common in robust regression, compressed sensing with nonconvex sparsity surrogates, and low-rank matrix recovery.

### 10.5.3 Relation to proximal methods

MM can often be interpreted as doing a proximal step on a locally quadratic or linearised upper bound. In that sense, it is philosophically close to proximal gradient (Chapter 9) and to Newton-like local quadratic approximation (Chapter 9), but with the additional twist that we are allowed to handle nonconvex $F$ as long as we *majorise* it with something convex.

---

## 10.6 Summary and perspective

We’ve now seen several algorithmic families that are particularly important at large scale and/or under structural constraints:

1. **Coordinate descent / block coordinate descent** 
    - Updates one coordinate block at a time.  
    - Converges globally for many convex problems.  
    - Scales extremely well in high dimensions.  
    - Used heuristically in nonconvex alternating minimisation.

2. **Stochastic and mini-batch gradient methods**  
    - Use noisy gradient estimates to get cheap iterations.  
    - Converge (in expectation) for convex problems.  
    - Power all of modern large-scale ML, including nonconvex deep learning.

3. **ADMM (Alternating Direction Method of Multipliers)**  
    - Splits a problem into simpler subproblems linked by linear constraints.  
    - Closely tied to duality and KKT (Chapters 7–8).  
    - Converges for convex problems.  
    - Used everywhere, including nonconvex settings, due to its modularity and parallelisability.

4. **Proximal coordinate / coordinate-prox methods**  
    - Merge sparsity-inducing penalties (Chapter 6) with blockwise updates.  
    - Ideal for $\ell_1$-type structure, group lasso, etc.  
    - Often extended to nonconvex penalties for even “more sparse” solutions.

5. **Majorization–minimization (MM)**  
    - Iteratively builds and minimises convex surrogates.  
    - Guarantees monotone descent of the true objective.  
    - Provides a clean bridge from convex optimisation theory into heuristic nonconvex optimisation.
