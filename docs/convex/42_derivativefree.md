# Chapter 20: Derivative-Free and Black-Box Optimization

In many practical optimization problems, gradients are unavailable, unreliable, or prohibitively expensive to compute. Examples include tuning hyperparameters of machine learning models, engineering design through simulation, or optimizing physical experiments. Such problems fall under the class of derivative-free or black-box optimization methods.

Unlike gradient-based methods, which rely on analytical or automatic differentiation, derivative-free algorithms make progress solely from function evaluations. They are indispensable when the objective function is noisy, discontinuous, or non-differentiable.



## 20.1 Motivation and Challenges

Let $f: \mathbb{R}^n \to \mathbb{R}$ be an objective function.  

A derivative-free algorithm seeks to minimize $f(x)$ using only evaluations of $f(x)$, without access to $\nabla f(x)$ or $\nabla^2 f(x)$.

Key challenges:

- No gradient information → difficult to infer descent directions.  
- Expensive evaluations → every call to $f(x)$ might require a simulation or experiment.  
- Noise and stochasticity → evaluations may be corrupted by measurement or sampling error.  
- High-dimensionality → sampling-based methods scale poorly with $n$.

Derivative-free optimization is thus a trade-off between exploration and exploitation, guided by heuristics or surrogate models.



## 20.2 Classification of Derivative-Free Methods

| Category | Representative Algorithms | Main Idea |
|---------------|-------------------------------|----------------|
| Direct Search | Nelder–Mead, Pattern Search, MADS | Explore the space via geometric moves or meshes |
| Model-Based | BOBYQA, Trust-Region DFO | Build local quadratic or surrogate models of $f$ |
| Evolutionary / Population-Based | CMA-ES, Differential Evolution | Evolve a population using stochastic operators |
| Probabilistic / Bayesian | Bayesian Optimization | Use probabilistic surrogate models to guide exploration |

 
## 20.3 Direct Search Methods

Direct search algorithms evaluate the objective function at structured sets of points and use comparisons, not gradients, to decide where to move.

### 20.3.1 Nelder–Mead Simplex Method
Perhaps the most famous derivative-free algorithm, Nelder–Mead maintains a simplex — a polytope of $n+1$ vertices in $\mathbb{R}^n$.

At each iteration:

1. Evaluate $f$ at all simplex vertices.
2. Reflect, expand, contract, or shrink the simplex depending on performance.
3. Continue until simplex collapses near a minimum.

Simple, intuitive, and effective for small-scale smooth problems, though it lacks formal convergence guarantees in general.


### 20.3.2 Pattern Search Methods
These methods (also called coordinate search or compass search) probe the function along coordinate directions or pre-defined patterns.

Typical update rule:
$$
x_{k+1} = x_k + \Delta_k d_i,
$$

where $d_i$ is a direction from a finite set (e.g., coordinate axes).  
If a direction yields improvement, move there; otherwise, shrink $\Delta_k$.


### 20.3.3 Mesh Adaptive Direct Search (MADS)
MADS refines pattern search by maintaining a mesh of candidate points and adaptively changing its resolution. It offers provable convergence to stationary points for certain classes of nonsmooth problems.


## 20.4 Model-Based Methods

Instead of exploring blindly, model-based methods construct an approximation of the objective function from past evaluations.

### 20.4.1 Trust-Region DFO
A local model $m_k(x)$ (often quadratic) is built to approximate $f$ near the current iterate $x_k$:
$$
m_k(x) \approx f(x_k) + g_k^\top (x - x_k) + \tfrac{1}{2}(x - x_k)^\top H_k (x - x_k).
$$
The next iterate solves a trust-region subproblem:
$$
\min_{\|x - x_k\| \le \Delta_k} m_k(x).
$$
The trust region size $\Delta_k$ adapts based on how well $m_k$ predicts true function values.

 
### 20.4.2 BOBYQA (Bound Optimization BY Quadratic Approximation)
BOBYQA builds and maintains a quadratic model using interpolation of previously evaluated points. It is highly efficient for medium-scale problems with simple box constraints and no noise.

 
## 20.5 Evolution Strategies and Population Methods

These methods maintain a population of candidate solutions and update them using statistical principles.

### 20.5.1 Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
CMA-ES is a powerful stochastic search algorithm.  
It iteratively samples new points from a multivariate Gaussian distribution:
$$
x^{(i)}_{k+1} \sim \mathcal{N}(m_k, \sigma_k^2 C_k),
$$
where $m_k$ is the current mean, $\sigma_k$ the global step-size, and $C_k$ the covariance matrix.

After evaluating all samples, the mean is updated toward better-performing points, and the covariance matrix adapts to the landscape geometry.

CMA-ES is invariant to linear transformations and excels in ill-conditioned, noisy, or nonconvex problems.

--- 

### 20.5.2 Differential Evolution (DE)
DE evolves a population $\{x_i\}$ via vector differences:
$$
v_i = x_{r1} + F(x_{r2} - x_{r3}),
$$
$$
u_i = \text{crossover}(x_i, v_i),
$$
$$
x_i' = \begin{cases} 
u_i, & \text{if } f(u_i) < f(x_i),\\
x_i, & \text{otherwise.}
\end{cases}
$$
where $r1, r2, r3$ are random distinct indices and $F$ controls mutation strength.

DE combines simplicity and robustness, performing well across continuous and discrete spaces.

 
## 20.6 Bayesian Optimization

When function evaluations are *expensive* (e.g., training a neural network or running a CFD simulation), Bayesian Optimization (BO) is preferred.

### 20.6.1 Core Idea
Model the objective as a random function $f(x) \sim \mathcal{GP}(m(x), k(x,x'))$ (Gaussian Process prior).  
After each evaluation, update the posterior mean and variance to quantify uncertainty.

Use an acquisition function $a(x)$ to select the next evaluation point:
$$
x_{k+1} = \arg\max_x a(x),
$$
balancing *exploration* (high uncertainty) and *exploitation* (low expected value).

Common acquisition functions:

- Expected Improvement (EI)
- Probability of Improvement (PI)
- Upper Confidence Bound (UCB)


### 20.6.2 Surrogate Models Beyond Gaussian Processes
When dimensionality is high or data is noisy, other surrogate models may replace GPs:
- Tree-structured Parzen Estimators (TPE)
- Random forests (SMAC)
- Neural network surrogates (Bayesian neural networks)

These variants enable Bayesian optimization in complex or discrete search spaces.


## 20.7 Hybrid and Adaptive Approaches

Modern applications often combine derivative-free and gradient-based techniques:

- Use Bayesian optimization for coarse global search, then local refinement with gradient descent.
- Alternate between CMA-ES and SGD to exploit both exploration and fast convergence.
- Apply direct search methods to tune hyperparameters of differentiable optimizers.

Such hybridization reflects a pragmatic view: no single optimizer is best — adaptability matters most.



## 20.8 Practical Considerations

| Aspect | Guideline |
|-------------|---------------|
| Function evaluations expensive | Use Bayesian or model-based methods |
| Noisy evaluations | Use averaging, smoothing, or robust estimators |
| High dimension ($n > 50$) | Prefer CMA-ES or evolutionary strategies |
| Box constraints | Methods like BOBYQA, DE, or PSO |
| Parallel computation available | Population-based methods excel |

---

Derivative-free optimization expands our toolkit beyond calculus, allowing us to optimize *anything we can evaluate*. It emphasizes adaptation, surrogate modeling, and population intelligence rather than analytical structure.

In the next chapter, we explore metaheuristic and evolutionary algorithms, which generalize these ideas further by mimicking natural and collective behaviors — turning randomness into a powerful search strategy.
