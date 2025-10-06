

## Algorithms for Convex Optimisation
Convex optimization algorithms exploit the geometry of convex sets and functions.  
Because every local minimum is global, algorithms can **converge reliably** without worrying about bad local minima.

We divide algorithms into three main families:

1. **First-order methods** – use gradients (scalable, but slower convergence).  
2. **Second-order methods** – use Hessians (faster convergence, more expensive).  
3. **Interior-point methods** – general-purpose, highly accurate solvers.


###  Gradient Descent (First-Order Method)

#### Algorithm
For step size $\alpha > 0$:
$$
x^{k+1} = x^k - \alpha \nabla f(x^k)
$$

#### Convergence
- If $f$ is convex and $\nabla f$ is Lipschitz continuous with constant $L$:
  - With fixed step $\alpha \le \tfrac{1}{L}$, we have:
    $
    f(x^k) - f(x^*) = \mathcal{O}\left(\tfrac{1}{k}\right)
    $
- If $f$ is **$\mu$-strongly convex**:
    $$
    f(x^k) - f(x^*) \le \left(1 - \tfrac{\mu}{L}\right)^k \big(f(x^0) - f(x^*)\big)
    $$
    (linear convergence).

#### Pros & Cons
- ✅ Simple, scalable to very high dimensions.  
- ❌ Slow convergence compared to higher-order methods.


### Accelerated Gradient Methods

- **Nesterov’s Accelerated Gradient (NAG):** Improves convergence rate from $\mathcal{O}(1/k)$ to $\mathcal{O}(1/k^2)$.  
- Widely used in **machine learning** (e.g., training deep neural networks).  

### Newton’s Method (Second-Order Method)

#### Algorithm
Update rule:
$$
x^{k+1} = x^k - [\nabla^2 f(x^k)]^{-1} \nabla f(x^k)
$$

#### Convergence
- Quadratic near optimum:  
  $$
  \|x^{k+1} - x^*\| \approx \mathcal{O}(\|x^k - x^*\|^2)
  $$
- Very fast, but requires Hessian and solving linear systems.

#### Damped Newton
To maintain global convergence:
$$
x^{k+1} = x^k - \alpha_k [\nabla^2 f(x^k)]^{-1} \nabla f(x^k), \quad \alpha_k \in (0,1]
$$


### Quasi-Newton Methods

Approximate the Hessian to reduce cost.

- **BFGS** and **L-BFGS** (limited memory version).  
- Used in large-scale optimization (e.g., machine learning, statistics).  
- Convergence: superlinear.  


### Subgradient Methods

For **nondifferentiable convex functions** (e.g., $f(x) = \|x\|_1$).

Update rule:
$$
x^{k+1} = x^k - \alpha_k g^k, \quad g^k \in \partial f(x^k)
$$

- $\partial f(x)$: subdifferential (set of all subgradients).  
- Convergence: $\mathcal{O}(1/\sqrt{k})$ with diminishing step sizes.  
- Useful in **large-scale, nonsmooth optimization**.

### Proximal Methods

For composite problems:  
$$
\min_x f(x) + g(x)
$$
where $f$ is smooth convex, $g$ convex but possibly nonsmooth.

**Proximal operator:**
$$
\text{prox}_g(v) = \arg\min_x \left( g(x) + \tfrac{1}{2}\|x-v\|_2^2 \right)
$$

- **Proximal gradient descent:**  
  $$
  x^{k+1} = \text{prox}_{\alpha g}(x^k - \alpha \nabla f(x^k))
  $$
- Widely used in sparse optimization (e.g., Lasso).


###  Interior-Point Methods
Transform constrained problem into a sequence of unconstrained problems using **barrier functions**.

For constraint $g_i(x) \le 0$, replace with barrier:
$$
\phi(x) = -\sum_{i=1}^m \log(-g_i(x))
$$

Solve:
$$
\min_x \; f(x) + \tfrac{1}{t} \phi(x)
$$
for increasing $t$.

#### Properties
- Polynomial-time complexity for convex problems.  
- Extremely accurate solutions.  
- Basis of general-purpose solvers (e.g., CVX, MOSEK, Gurobi).  


### Coordinate Descent

At each iteration, optimize w.r.t. one coordinate (or block of coordinates):

$$
x_i^{k+1} = \arg\min_{z} f(x_1^k, \dots, x_{i-1}^k, z, x_{i+1}^k, \dots, x_n^k)
$$

- Works well for high-dimensional problems.  
- Used in Lasso, logistic regression, and large-scale ML problems.


### Primal-Dual and Splitting Methods

- **ADMM (Alternating Direction Method of Multipliers):**  
  Splits problem into subproblems, solves in parallel.  
  Popular in distributed optimization and ML.  

- **Primal-dual interior-point methods:**  
  Solve both primal and dual simultaneously.  

---

### Summary of Convergence Rates

| Method                  | Smooth Convex | Strongly Convex |
|-------------------------|---------------|-----------------|
| Gradient Descent        | $\mathcal{O}(1/k)$ | Linear |
| Accelerated Gradient    | $\mathcal{O}(1/k^2)$ | Linear |
| Subgradient             | $\mathcal{O}(1/\sqrt{k})$ | – |
| Newton’s Method         | Quadratic (local) | Quadratic |
| Interior-Point          | Polynomial-time | Polynomial-time |



### Choosing an Algorithm

- **Small problems, high accuracy:** Newton, Interior-point.  
- **Large-scale smooth problems:** Gradient descent, Nesterov acceleration, L-BFGS.  
- **Large-scale nonsmooth problems:** Subgradient, Proximal, ADMM.  
- **Sparse / structured constraints:** Coordinate descent, Proximal methods.  

## Log-concavity and Log-convexity

A function $f: \mathbb{R}^n \to \mathbb{R}_{++}$ is:

- **Log-concave** if $\log f(x)$ is concave.
- **Log-convex** if $\log f(x)$ is convex.

### Relevance

- Log-concave functions appear in probability (many common distributions have log-concave densities, such as Gaussian, exponential, and uniform). This ensures tractability of maximum likelihood estimation.
- Log-convexity is useful in geometric programming, where monomials and posynomials are log-convex.

### Examples

- Gaussian density is log-concave.
- Exponential function is log-convex.


## Geometric Programming

Geometric programming (GP) is a class of problems of the form:
$$
\begin{aligned}
& \min_x \quad & f_0(x) \\
& \text{s.t.} \quad & f_i(x) \leq 1, \quad i=1, \dots, m,
\end{aligned}
$$
where each $f_i$ is a posynomial.

- **Monomial**: $f(x) = c x_1^{a_1} x_2^{a_2} \dots x_n^{a_n}$, with $c > 0$, exponents real.
- **Posynomial**: Sum of monomials.

By applying the log transformation $y_i = \log x_i$, the problem becomes convex.