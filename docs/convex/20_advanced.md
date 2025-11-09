# Chapter 15: Advanced Large-Scale and Structured Methods

Modern convex optimization often operates at massive scales — millions of variables, billions of data points, or constraints distributed across devices and networks.  
Classical Newton or interior-point algorithms, while theoretically elegant, become computationally impractical in these regimes.  

This chapter introduces methods that exploit structure, sparsity, separability, and stochasticity to solve large-scale convex problems efficiently.  
These ideas underpin the optimization engines behind most machine learning systems.

 

## 15.1 Motivation: Structure and Scale

In large-scale convex optimization, the difficulty lies not in theory but in computation.

- Memory limits: Storing the full Hessian or even the gradient can be infeasible.  
- Data size: Evaluating the objective over the full dataset is expensive.  
- Distributed data: Information may be spread across machines or devices.  
- Sparsity and separability: Many objectives decompose nicely into smaller components.

Thus, the goal is to design algorithms that make incremental or local progress while exploiting the structure of the problem.

Typical forms include:
$$
f(x) = \frac{1}{N}\sum_{i=1}^N f_i(x) + R(x),
$$
where:
- each $f_i(x)$ represents a data-sample loss term, and  
- $R(x)$ is a regularizer (possibly nonsmooth, such as $\lambda\|x\|_1$).

## 15.2 Coordinate Descent

Coordinate descent updates a single variable (or a small block) at a time while holding others fixed.  

### Algorithm
Given $x^{(k)}$, choose coordinate $i$ and update:
$$
x_i^{(k+1)} = \arg\min_{z} f(x_1^{(k+1)}, \ldots, x_{i-1}^{(k+1)}, z, x_{i+1}^{(k)}, \ldots, x_n^{(k)}).
$$

This can be seen as projecting the gradient onto the coordinate directions. For separable problems, it is computationally much cheaper than full gradient updates.


- Each subproblem is often 1D (or low-dimensional), so it may have a closed form.
- For problems with separable structure — e.g. sums over features, or regularisers like $\|x\|_1 = \sum_i |x_i|$ — the coordinate update is extremely cheap.
- You never form the full gradient or solve a large linear system; you just operate on pieces.

This is especially attractive in high dimensions (millions of features), where a full Newton step would be absurdly expensive.

### Convergence
If $f$ is convex with Lipschitz-continuous partial derivatives, cyclic or randomized coordinate descent converges to the global optimum.

### ML Context
Coordinate descent is widely used in:
- LASSO and Elastic Net regression (where updates are closed-form soft-thresholding),
- logistic regression with $\ell_1$ penalty,
- matrix factorization and dictionary learning.

## 15.3 Stochastic Gradient and Variance-Reduced Methods

When the dataset is large, computing the full gradient
$$
\nabla f(x) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(x)
$$
is prohibitive. Instead, stochastic methods use a random mini-batch $\mathcal{B}$ to estimate it.

### Basic SGD
$$
x_{k+1} = x_k - \eta_k \nabla f_{i_k}(x_k),
$$
where $i_k$ is a random index or mini-batch.

The step size $\eta_k$ typically decays with $k$, such as $\eta_k = c / \sqrt{k}$.

### Convergence
- For convex $f$, SGD achieves expected error $O(1/\sqrt{k})$.  
- For strongly convex $f$, using decreasing $\eta_k$ gives $O(1/k)$ convergence.  
- In practice, adaptive methods (Adam, RMSProp) improve stability.

### Variance Reduction
Methods like SVRG (Stochastic Variance-Reduced Gradient) and SAGA combine stochastic and deterministic steps to reduce variance:
$$
\nabla f_{SVRG}(x) = \nabla f_i(x) - \nabla f_i(\tilde{x}) + \nabla f(\tilde{x}),
$$
where $\tilde{x}$ is a reference (snapshot) point.  
These achieve linear convergence for strongly convex functions, bridging the gap between SGD and full gradient descent.

### ML Context
- Deep neural networks rely almost exclusively on SGD and its adaptive variants.  
- Large-scale convex ML (logistic regression, SVM) also uses SGD or SVRG for efficiency.

## 15.4 Proximal and Composite Optimization

Many modern objectives combine a smooth loss and a nonsmooth regularizer:
$$
\min_x \; g(x) + R(x),
$$
where $g$ is differentiable with Lipschitz gradient and $R$ is convex but possibly nonsmooth.

The proximal gradient method updates as:
$$
x_{k+1} = \mathrm{prox}_{\alpha R}(x_k - \alpha \nabla g(x_k)),
$$
where the proximal operator is:
$$
\mathrm{prox}_{\alpha R}(v) = \arg\min_x \left( R(x) + \frac{1}{2\alpha}\|x-v\|^2 \right).
$$

### Intuition
- The gradient step moves in a descent direction for $g$.  
- The proximal step performs a local “denoising” or shrinkage under $R$ (e.g., soft-thresholding for $\ell_1$ norms).

### ML Context
Proximal methods underpin:
- Sparse regression (LASSO, Elastic Net),
- matrix completion and compressed sensing,
- total-variation image denoising,
- low-rank and structured regularization.


## 15.5 Alternating Direction Method of Multipliers (ADMM)

When an objective separates into parts that depend on different variables, ADMM enables efficient distributed optimization.

Consider:
$$
\min_{x,z}\; f(x) + g(z) \quad \text{s.t. } A x + B z = c.
$$

### Augmented Lagrangian
$$
L_\rho(x,z,y) = f(x) + g(z) + y^T(Ax + Bz - c) + \frac{\rho}{2}\|A x + B z - c\|^2.
$$

### Iterations
ADMM performs alternating updates:
$$
\begin{aligned}
x^{k+1} &= \arg\min_x L_\rho(x, z^k, y^k),\\
z^{k+1} &= \arg\min_z L_\rho(x^{k+1}, z, y^k),\\
y^{k+1} &= y^k + \rho (A x^{k+1} + B z^{k+1} - c).
\end{aligned}
$$

### Interpretation
Each step solves an easier subproblem involving only part of the variables, followed by a dual update to enforce consistency.  
ADMM thus merges ideas from dual ascent and penalty methods.

### Convergence
For convex $f$ and $g$, ADMM converges to the global optimum.  
It is particularly effective when the subproblems are simple (e.g., proximal operators).

### ML Context
ADMM is a key tool for:
- distributed LASSO and logistic regression,
- matrix decomposition and factorization,
- consensus optimization in federated learning,
- distributed deep learning regularization.

## 15.6 Majorization–Minimization (MM) and EM Algorithms

The MM principle iteratively minimizes a surrogate function that upper-bounds the objective.

Given a current point $x_k$, construct a surrogate $g(x|x_k)$ such that:
$$
g(x|x_k) \ge f(x), \quad g(x_k|x_k) = f(x_k).
$$

Then update:
$$
x_{k+1} = \arg\min_x g(x|x_k).
$$

Each iteration ensures $f(x_{k+1}) \le f(x_k)$.

### ML Context
- The Expectation–Maximization (EM) algorithm is an MM method for latent-variable models.  
- IRLS (Iteratively Reweighted Least Squares) for logistic regression and $\ell_p$ regression follows the same idea.  
- MM methods guarantee descent even for complex, nonconvex objectives.

 
## 15.7 Distributed and Parallel Optimization

For large-scale convex problems distributed across multiple nodes, parallel methods are essential.

### Synchronous and Asynchronous Updates
- Synchronous: all workers compute updates and synchronize (used in federated averaging).  
- Asynchronous: updates proceed without waiting, improving throughput but increasing variance.

### Consensus Optimization
In distributed convex optimization, one solves
$$
\min_{x_1,\dots,x_p} \sum_{i=1}^p f_i(x_i)
\quad \text{s.t. } x_i = z,
$$
which can be handled by ADMM or primal–dual methods.  
Each machine optimizes its local copy $x_i$, and the shared variable $z$ enforces consensus.

### ML Context
- Federated learning and parameter-server training frameworks (e.g., TensorFlow Distributed, PyTorch DDP) follow this model.  
- Decentralized convex optimization appears in sensor networks and multi-agent control.

 
## 15.8 Handling Structure: Sparsity and Low Rank

Many convex problems exhibit special structures that algorithms can exploit:

| Structure | Typical Regularizer | Algorithmic Advantage |
|------------|--------------------|-----------------------|
| Sparsity | $\ell_1$ or group lasso | Coordinate updates, proximal shrinkage |
| Low rank | nuclear norm $\|X\|_*$ | SVD-based proximal step |
| Block separability | $\sum_i f_i(x_i)$ | Parallel or distributed updates |
| Graph structure | total variation norm | Local neighborhood computations |
| Simplex or probability constraints | entropy or KL penalty | Mirror descent, projected methods |

Exploiting such structure yields orders-of-magnitude speedups in both memory and computation.

 
## 15.9 Summary and Practical Guidance

| Method | Gradient Access | Scalability | Parallelization | Convexity Required | Typical ML Uses |
|---------|----------------|--------------|-----------------|--------------------|----------------|
| Coordinate Descent | Partial / coordinate | High | Easy | Convex | LASSO, sparse models |
| SGD / SVRG / SAGA | Stochastic | Excellent | Natural | Convex / nonconvex | Deep learning, logistic regression |
| Proximal Gradient | Full gradient + prox | Moderate–High | Easy | Convex | Composite objectives |
| ADMM | Separable subproblems | High | Distributed | Convex | Consensus, large convex solvers |
| MM / EM | Surrogate-based | Moderate | Model-specific | Convex / nonconvex | Probabilistic models, IRLS |
| Distributed / Federated | Local gradients | Very high | Essential | Convex / smooth | Federated learning, large-scale convex optimization |

 
## 15.10 Key Takeaways

- Large-scale convex optimization relies on exploiting structure, stochasticity, and separability.  
- Coordinate and proximal methods handle sparse and composite problems efficiently.  
- Stochastic and variance-reduced methods scale to massive data.  
- ADMM and distributed optimization enable multi-machine or federated settings.  
- MM and EM extend convex ideas to broader nonconvex inference tasks.

 