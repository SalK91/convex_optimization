So far, all optimization methods we have discussed, including Gradient Descent (GD) and Mirror Descent (MD), assume exact evaluation of gradients or subgradients. In practice, however, computing exact gradients is often impossible or computationally expensive. Stochastic Gradient Descent (SGD) and related methods address this by using noisy or approximate gradients, enabling scalable optimization in large-scale or complex settings


## 1. Motivation

Classical optimization relies on computing the exact gradient $\nabla f(x)$ at each iteration. However, in many real-world scenarios, this is impractical:

1. **Exact gradient unavailable**  
   - Some functions are non-differentiable or analytically intractable.  
   - Example: complex composite objectives or non-smooth loss functions.

2. **Computational cost is prohibitive**  
   - Large datasets make full gradient computation expensive.  
   - Neural networks require backpropagation over all samples per step, which can be infeasible.

3. **Stochastic optimization naturally arises**  
   - **Data scale:** Objective involves a sum or expectation over massive datasets.  
   - **Intrinsic randomness:** Objective defined as an expectation over a stochastic process, e.g., in reinforcement learning, online optimization, or probabilistic modeling.


## 2. Stochastic Gradient Descent (SGD)

Instead of computing the full gradient, SGD uses a **stochastic estimate** $g_t$ such that:

$$
\mathbb{E}[g_t \mid x_t] = \nabla f(x_t)
$$

The update rule becomes:

$$
x_{t+1} = x_t - \eta g_t
$$

where:

- $g_t$ is a **stochastic (noisy) gradient**,  
- $\eta$ is the step size or learning rate.

> **Intuition:** We follow a “noisy compass” that points roughly in the right direction instead of computing the exact gradient at every step.

 
### 2.1 Sources of Gradient Noise

- **Finite datasets:** Only a subset (mini-batch) of data is used.  
- **Random sampling:** Randomly select samples to approximate the full gradient.  
- **Inherent stochasticity:** In reinforcement learning or simulation-based optimization, gradients are intrinsically noisy.

**Example: Empirical Risk Minimization (ERM)**

For a dataset $\{z_1, \dots, z_n\}$ and loss function $\ell(x; z_i)$:

$$
f(x) = \frac{1}{n} \sum_{i=1}^n \ell(x; z_i)
$$

- Exact gradient: $\nabla f(x) = \frac{1}{n} \sum_i \nabla \ell(x; z_i)$  
- SGD gradient: $g_t = \nabla \ell(x; z_{i_t})$, where $i_t$ is randomly sampled

> ✅ Reduces per-step computation from $O(n)$ to $O(1)$ or $O(\text{batch size})$.

 
### 2.2 SGD Update Rule

1. Sample a stochastic gradient $g_t$ (single sample or mini-batch).  
2. Update:

$$
x_{t+1} = x_t - \eta g_t
$$

- Converges **in expectation** under standard assumptions (convexity, bounded variance).  
- Simple, yet highly effective for large-scale optimization.

 

### 2.3 Comparison: GD vs SGD

| Feature | Gradient Descent | Stochastic Gradient Descent |
|---------|-----------------|----------------------------|
| Gradient | Full / exact | Noisy / approximate |
| Step cost | High (entire dataset) | Low (single sample / mini-batch) |
| Update direction | Accurate | Approximate, stochastic |
| Trajectory | Smooth | Noisy, zig-zag |
| Convergence | Deterministic | In expectation, slower per iteration, but scalable |

> Visual intuition: SGD zig-zags along the gradient surface but gradually converges toward a minimum.

 
## 3. Practical Considerations

### 3.1 Step Size / Learning Rate

- Constant or decaying $\eta_t$  
- Too large → divergence or oscillation  
- Too small → slow convergence

### 3.2 Mini-batching

- Trades off **variance** vs **computation**:  
  - Smaller batches → noisier but cheaper updates  
  - Larger batches → smoother updates but more computation

### 3.3 Momentum and Variants

- **Momentum, RMSProp, Adam**: reduce stochastic noise, improve convergence.

### 3.4 Noise as a Feature

- Noise can help **escape shallow local minima**.  
- SGD is more robust in non-convex landscapes, e.g., deep neural networks.

 
## 4. Convergence Guarantees

For convex $f$ with bounded gradient variance:

$$
\mathbb{E}[f(\bar{x}_T)] - f(x^*) \le O\left(\frac{1}{\sqrt{T}}\right)
$$

- $\bar{x}_T = \frac{1}{T} \sum_{t=1}^T x_t$  
- $T$ = number of iterations  

> Slower than exact GD ($O(1/T)$ for smooth convex problems), but much cheaper per iteration.

 
## 5. Example: Stochastic Optimization in Regression

Consider minimizing expected squared error:

$$
\min_x \mathbb{E}_\xi \big[ (y - x^\top \xi)^2 \big]
$$

Given a dataset $\{(y_1, \xi_1), \dots, (y_n, \xi_n)\}$, the **empirical risk** is:

$$
\hat{F}(x) = \frac{1}{n} \sum_{i=1}^n (y_i - x^\top \xi_i)^2
$$

### 5.1 Gradient Descent (GD)

$$
x_{t+1} = x_t - \eta \nabla \hat{F}(x_t) 
= x_t - \eta \frac{1}{n} \sum_{i=1}^n \nabla f_i(x_t)
$$

- Full gradient requires a pass over all $n$ samples → expensive for large datasets.

### 5.2 Stochastic Gradient Descent (SGD)

Randomly sample an index $i_t \in \{1, \dots, n\}$ and compute:

$$
g_t = \nabla f_{i_t}(x_t)
$$

Update rule:

$$
x_{t+1} = x_t - \eta_t g_t
$$

- Key property:

$$
\mathbb{E}[g_t] = \nabla \hat{F}(x_t)
$$

Thus, $g_t$ is an **unbiased estimate** of the true gradient.

 
## 6. Randomized Coordinate Descent (RCD)

Another approach to reduce computational cost is **Randomized Coordinate Descent (RCD)**. Instead of computing the full gradient, RCD updates **only a randomly selected coordinate** (or block of coordinates) at each iteration:

$$
x_{t+1}^{(i)} = x_t^{(i)} - \eta \frac{\partial f(x_t)}{\partial x^{(i)}}, \quad i \sim \text{Uniform}(\{1, \dots, d\})
$$

- **Key idea:** Update only a subset of coordinates to reduce per-iteration cost from $O(d)$ to $O(1)$ (or $O(\text{block size})$).  
- **Connection to SGD:** Both are stochastic approximations of full gradient descent:
  - SGD introduces randomness via data sampling.  
  - RCD introduces randomness via coordinate selection.  

### 6.1 Benefits of RCD

- Efficient for **high-dimensional problems**.  
- Exploits **sparsity** in variables.  
- Supports **parallel and distributed updates** through block-coordinate schemes.

### 6.2 Variants

- **Cyclic coordinate descent:** systematically updates each coordinate in order.  
- **Importance sampling:** selects coordinates with probability proportional to the magnitude of partial derivatives to accelerate convergence.

### 6.3 Comparison to GD and SGD

| Method | Gradient Computation | Iteration Cost | Convergence Behavior |
|--------|-------------------|----------------|--------------------|
| GD | Full gradient | High ($O(d)$) | Deterministic, fast for small-scale problems |
| SGD | Stochastic gradient | Low ($O(\text{mini-batch})$) | Noisy updates, scalable for large datasets |
| RCD | Partial gradient (coordinate) | Very low ($O(1)$ or block) | Stochastic, efficient for high-dimensional sparse problems |

> RCD and SGD illustrate a **unified principle**: use a cheaper, stochastic approximation of the true gradient to achieve scalable optimization.

 

## 7. Mini-batching and Variance Reduction

### 7.1 Mini-batch SGD

Compute the gradient over a small subset (mini-batch) of size $b$:

$$
g_t = \frac{1}{b} \sum_{i \in \mathcal{B}_t} \nabla f_i(x_t)
$$

- **Benefits:**  
  - Reduces variance of the gradient estimate.  
  - Parallelizable on GPUs/TPUs.  
  - Smoother update trajectory than single-sample SGD.

- **Trade-offs:**  
  - Smaller batch → cheaper but noisier updates.  
  - Larger batch → more computation, lower variance.

 
### 7.2 Variance-Reduced Gradient Methods (SVRG)

**SVRG (Stochastic Variance Reduced Gradient)** improves convergence by correcting stochastic gradients:

$$
g_t = \nabla f_{i_t}(x_t) - \nabla f_{i_t}(\tilde{x}) + \nabla f(\tilde{x})
$$

where:

- $i_t$ is a random index,  
- $\tilde{x}$ is a reference point with precomputed full gradient $\nabla f(\tilde{x}) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(\tilde{x})$.

- **Benefits:**  
  - Faster convergence than standard SGD for strongly convex problems.  
  - Maintains low per-iteration cost similar to SGD.  
  - Reduces stochastic oscillations by anchoring gradients around a reference.

- **Intuition:** Standard SGD “wanders” due to noise; SVRG periodically corrects it using a reference full gradient.
