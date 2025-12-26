# Chapter 15: Advanced Large-Scale and Structured Methods

Modern convex optimization often runs at massive scale: millions (or billions) of variables, datasets too large to fit in memory, and constraints spread across machines or devices. Per-iteration cost and memory usage often of often makes classical solutions impractical for these regimes.

This chapter introduces methods that exploit structure, sparsity, separability, and stochasticity to make convex optimization scalable. These ideas underpin the optimization engines behind most modern machine learning systems.

 

## Motivation: Structure and Scale
In large-scale convex optimization, the challenge is not “does a solution exist?” but rather “can we compute it in time and memory?”.

Bottlenecks include:

- Memory: storing Hessians (or even full gradients) may be impossible.
- Data size: one full pass over all samples can already be expensive.
- Distributed data: samples are spread across devices / workers.
- Sparsity and separability: the problem often decomposes into many small pieces.

A common template is the empirical risk + regularizer form
$$
f(x)
= \frac{1}{N}\sum_{i=1}^N f_i(x) + R(x),
$$
where

- each $f_i(x)$ is a loss term for sample $i$,
- $R(x)$ is a regularizer (possibly nonsmooth, e.g. $\lambda\|x\|_1$).

The methods in this chapter are designed to exploit this structure:

- update only *parts* of $x$ at a time (coordinate/block methods),
- use only *some* data per step (stochastic methods),
- split the problem into simpler subproblems (proximal / ADMM),
- or distribute computation across multiple machines (consensus methods).


## Coordinate Descent
Coordinate descent updates one coordinate (or a small block of coordinates) at a time, holding all others fixed. It is especially effective when updates along a single coordinate are cheap to compute. Given $x^{(k)}$, choose coordinate $i_k$ and define

$$
x^{(k+1)}_i =
\begin{cases}
\displaystyle
\arg\min_{z \in \mathbb{R}}
\; f\big(x_1^{(k+1)},\dots,x_{i_k-1}^{(k+1)},z,x_{i_k+1}^{(k)},\dots,x_n^{(k)}\big),
& i = i_k,\\[4pt]
x_i^{(k)}, & i \ne i_k.
\end{cases}
$$

In practice:

- $i_k$ is chosen either cyclically ($1,2,\dots,n,1,2,\dots$) or randomly.
- Each coordinate update often has a closed form (e.g. soft-thresholding for LASSO).
- You never form or store the full gradient; you only need partial derivatives.

Why it scales:

- Each step is *very* cheap — often $O(1)$ or proportional to the number of nonzeros in the column corresponding to coordinate $i_k$.
- In high dimensions (e.g., millions of features), this can be far more efficient than updating all coordinates at once.

Convergence:  
If $f$ is convex and has Lipschitz-continuous partial derivatives, coordinate descent (cyclic or randomized) converges to the global minimizer. Randomized coordinate descent often has clean expected convergence rates.

ML context:

- LASSO / Elastic Net regression (coordinate updates are soft-thresholding),
- $\ell_1$-penalized logistic regression,
- matrix factorization and dictionary learning (updating one factor vector at a time),
- problems where $R(x)$ is separable: $R(x) = \sum_i R_i(x_i)$.



## Stochastic Gradient and Variance-Reduced Methods
When $N$ (number of samples) is huge, computing the full gradient
$$
\nabla f(x) = \frac{1}{N}\sum_{i=1}^N \nabla f_i(x)
$$
every iteration is too expensive. Stochastic methods replace this full gradient with cheap, unbiased *estimates* based on small random subsets (mini-batches) of data.

### Stochastic Gradient Descent (SGD)

At iteration $k$:

1. Sample a mini-batch $\mathcal{B}_k \subset \{1,\dots,N\}$.
2. Form the stochastic gradient
   $$
   \widehat{\nabla f}(x_k)
   =
   \frac{1}{|\mathcal{B}_k|}
   \sum_{i \in \mathcal{B}_k} \nabla f_i(x_k).
   $$
3. Update
   $$
   x_{k+1} = x_k - \eta_k \,\widehat{\nabla f}(x_k),
   $$
   where $\eta_k > 0$ is the learning rate.

Properties:

- $\mathbb{E}[\widehat{\nabla f}(x_k) \mid x_k] = \nabla f(x_k)$ (unbiased),
- Each iteration is cheap (depends only on $|\mathcal{B}_k|$, not $N$),
- The noise can help escape shallow nonconvex traps (in deep learning).

In convex settings, SGD trades off per-iteration cost against convergence speed: many cheap noisy steps instead of fewer expensive precise ones.

### Step Sizes and Averaging

The step size $\eta_k$ is crucial:

- Too large → iterates diverge or oscillate.
- Too small → extremely slow progress.

Typical schedules for convex problems:

- General convex: $\eta_k = \frac{c}{\sqrt{k}}$,
- Strongly convex: $\eta_k = \frac{c}{k}$.

Two important stabilization techniques:

1. Decay the learning rate over time.
2. Polyak–Ruppert averaging: return the average
   $$
   \bar{x}_k = \frac{1}{k}\sum_{t=1}^k x_t
   $$
   instead of the last iterate. Averaging cancels noise and leads to optimal $O(1/k)$ rates in strongly convex settings.

Mini-batch size can also grow with $k$, gradually reducing variance while keeping early iterations cheap.

### Convergence Rates

For convex $f$:

- With appropriate diminishing $\eta_k$,  
  $\mathbb{E}[f(x_k)] - f^\star = O(k^{-1/2})$.

For strongly convex $f$:

- With $\eta_k = O(1/k)$ and averaging,  
  $\mathbb{E}[\|x_k - x^\star\|^2] = O(1/k)$.

These rates are optimal for unbiased first-order stochastic methods.

### Variance-Reduced Methods (SVRG, SAGA, SARAH)

Plain SGD cannot easily reach very high accuracy because the gradient noise never fully disappears. Variance-reduced methods reduce this noise, especially near the solution, by periodically using the full gradient.

Example: SVRG (Stochastic Variance-Reduced Gradient)

- Pick a reference point $\tilde{x}$ and compute $\nabla f(\tilde{x})$.
- For inner iterations:
  $$
  v_k = \nabla f_{i_k}(x_k) - \nabla f_{i_k}(\tilde{x}) + \nabla f(\tilde{x}),
  \quad
  x_{k+1} = x_k - \eta v_k,
  $$
  where $i_k$ is a random sample index.

Here $v_k$ is still an unbiased estimator of $\nabla f(x_k)$, but its variance decays as $x_k$ approaches $\tilde{x}$. For strongly convex $f$, methods like SVRG and SAGA achieve linear convergence, comparable to full gradient descent but at near-SGD cost.

### Momentum and Adaptive Methods

In practice, large-scale learning often uses SGD with various modifications:

- Momentum / Nesterov: keep a moving average of gradients
  $$
  m_k = \beta m_{k-1} + (1-\beta)\widehat{\nabla f}(x_k),
  \quad
  x_{k+1} = x_k - \eta m_k,
  $$
  which accelerates progress along consistent directions and damps oscillations.

- Adaptive methods (Adagrad, RMSProp, Adam): maintain coordinate-wise scales based on past squared gradients, effectively using a diagonal preconditioner that adapts to curvature and feature scales.

These methods are especially popular in deep learning. For convex problems, their theoretical behavior is subtle, but empirically they often converge faster in wall-clock time.

## Proximal and Composite Optimization
Many large-scale convex problems are composite:
$$
\min_x \; F(x) := g(x) + R(x),
$$
where

- $g$ is smooth convex with Lipschitz gradient (e.g. data-fitting term),
- $R$ is convex but possibly nonsmooth (e.g. $\lambda\|x\|_1$, indicator of a constraint set, nuclear norm).

The proximal gradient method (a.k.a. ISTA) updates as
$$
x_{k+1} = \operatorname{prox}_{\alpha R}\big(x_k - \alpha \nabla g(x_k)\big),
$$
where the proximal operator is
$$
\operatorname{prox}_{\alpha R}(v)
=
\arg\min_x \left(
R(x) + \frac{1}{2\alpha}\|x-v\|_2^2
\right).
$$

Intuition:

- The gradient step moves $x$ in a direction that lowers the smooth term $g$.
- The prox step solves a small “regularized” problem, pulling $x$ toward a structure favored by $R$ (sparsity, low rank, feasibility, etc.).

Examples of prox operators:

- $R(x) = \lambda\|x\|_1$ → soft-thresholding (coordinate-wise shrinkage).
- $R$ = indicator of a convex set $\mathcal{X}$ → projection onto $\mathcal{X}$ (so projected gradient is a special case).
- $R(X) = \|X\|_*$ (nuclear norm) → singular value soft-thresholding.

For large-scale problems:

- Proximal gradient scales like gradient descent: each iteration uses only $\nabla g$ and a prox (often cheap and parallelizable).
- Accelerated variants (FISTA) achieve $O(1/k^2)$ rates for smooth $g$.



## Alternating Direction Method of Multipliers (ADMM)
When objectives naturally split into simpler pieces depending on different variables, ADMM is a powerful tool. It is especially useful when:

- $f$ and $g$ have simple prox operators,
- the problem is distributed or separable across machines.

Consider
$$
\min_{x,z}\; f(x) + g(z)
\quad
\text{s.t. } A x + B z = c.
$$

The augmented Lagrangian is
$$
L_\rho(x,z,y)
=
f(x) + g(z)
+ y^\top(Ax + Bz - c)
+ \frac{\rho}{2}\|A x + B z - c\|_2^2,
$$
with dual variable $y$ and penalty parameter $\rho > 0$.

ADMM performs the iterations:
$$
\begin{aligned}
x^{k+1} &= \arg\min_x L_\rho(x, z^k, y^k),\\[4pt]
z^{k+1} &= \arg\min_z L_\rho(x^{k+1}, z, y^k),\\[4pt]
y^{k+1} &= y^k + \rho \big(A x^{k+1} + B z^{k+1} - c\big).
\end{aligned}
$$

Interpretation:

- The $x$-update solves a subproblem involving $f$ only.
- The $z$-update solves a subproblem involving $g$ only.
- The $y$-update nudges the constraint $A x + B z = c$ toward satisfaction.

For convex $f,g$, ADMM converges to a primal–dual optimal point. It is particularly effective when the $x$- and $z$-subproblems have closed-form prox solutions or can be solved cheaply in parallel.

ML use cases:

- Distributed LASSO / logistic regression,
- matrix completion and robust PCA,
- consensus optimization (each worker has local data but shares a global model),
- some federated learning formulations.

## Majorization–Minimization (MM) and EM Algorithms
The Majorization–Minimization (MM) principle constructs at each iterate $x_k$ a surrogate function $g(\cdot \mid x_k)$ that upper-bounds $f$ and is easier to minimize.

Requirements:
$$
g(x \mid x_k) \ge f(x)\ \text{ for all } x, 
\quad
g(x_k \mid x_k) = f(x_k).
$$

Then define
$$
x_{k+1} = \arg\min_x g(x \mid x_k).
$$

This guarantees monotone decrease:
$$
f(x_{k+1}) \le g(x_{k+1}\mid x_k) \le g(x_k\mid x_k) = f(x_k).
$$

The famous Expectation–Maximization (EM) algorithm is an MM method for latent-variable models, where the surrogate arises from Jensen’s inequality and missing-data structure.

Other examples:

- Iteratively Reweighted Least Squares (IRLS) for logistic regression and robust regression,
- MM surrogates for nonconvex penalties (e.g. smoothly approximating $\ell_0$),
- mixture models and variational inference.

 
## Distributed and Parallel Optimization
When data or variables are split across machines, we need distributed or parallel optimization schemes.

### Synchronous vs Asynchronous

- Synchronous methods: all workers compute local gradients or updates, then synchronize (e.g. parameter server, federated averaging).
- Asynchronous methods: workers update parameters without global synchronization, improving hardware utilization but introducing staleness and variance.

### Consensus Optimization

A standard pattern is consensus form:
$$
\min_{x_1,\dots,x_P,z}
\sum_{i=1}^P f_i(x_i)
\quad \text{s.t. } x_i = z,\; i = 1,\dots,P,
$$
where $f_i$ is the local objective on worker $i$ and $z$ is the global consensus variable.

ADMM applied to this problem:

- Each worker updates its local $x_i$ using only local data,
- The global variable $z$ is updated by averaging or aggregation,
- Dual variables enforce agreement $x_i \approx z$.

This template underlies many federated learning and parameter-server architectures.

### ML Context

- Federated learning (phone/edge devices update local models and send summaries to a server),
- Large-scale convex optimization over sharded datasets,
- Distributed sparse regression, matrix factorization, and graphical model learning.


 

 
## Handling Structure: Sparsity and Low Rank
Large-scale convex problems often have additional structure that we can exploit algorithmically:

| Structure          | Typical Regularizer / Constraint     | Algorithmic Benefit                               |
|--------------------|---------------------------------------|--------------------------------------------------|
| Sparsity           | $\ell_1$, group lasso                | Cheap coordinate updates, soft-thresholding      |
| Low rank           | Nuclear norm $\|X\|_*$               | SVD-based prox; rank truncation                  |
| Block separability | $\sum_i f_i(x_i)$                    | Parallel or distributed block updates            |
| Graph structure    | Total variation on graphs            | Local neighborhood computations                  |
| Probability simplex | simplex constraint or entropy term | Mirror descent, simplex projections              |

Examples:

- In compressed sensing, $\ell_1$ regularization + sparse sensing matrices → very cheap mat–vecs + prox operations.
- In matrix completion, nuclear norm structure + low-rank iterates → approximate SVD instead of full SVD.
- In TV denoising, local difference structure → each prox step involves only neighboring pixels/vertices.

Exploiting structure can yield orders-of-magnitude speedups compared to generic solvers.


## Summary and Practical Guidance
Different large-scale methods are appropriate in different regimes:

| Method                     | Gradient Access      | Scalability | Parallelization          | Convexity Needed | Typical Uses                          |
|----------------------------|----------------------|-------------|--------------------------|------------------|----------------------------------------|
| Coordinate Descent         | Partial gradients    | High        | Easy (blockwise)         | Convex           | LASSO, sparse GLMs, matrix factorization |
| SGD / Mini-batch SGD       | Stochastic gradients | Excellent   | Natural (data parallel)  | Convex / nonconvex | Deep learning, logistic regression     |
| SVRG / SAGA (VR methods)   | Stochastic + periodic full gradient | High | Data parallel           | Convex, often strongly convex | Large-scale convex ML, GLMs          |
| Proximal Gradient (ISTA/FISTA) | Full gradient + prox | Moderate–High | Easy                      | Convex           | Composite objectives with structure   |
| ADMM                       | Local subproblems    | High        | Designed for distributed | Convex           | Consensus, distributed convex solvers |
| MM / EM                    | Surrogates           | Moderate    | Model-specific           | Convex / nonconvex | Latent-variable models, IRLS         |
| Distributed / Federated    | Local gradients      | Very high   | Essential                | Often convex / smooth | Federated learning, multi-agent systems |

 

 