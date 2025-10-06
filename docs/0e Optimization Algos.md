# Optimization Algorithms

Optimization is at the **heart of machine learning**: training a model is equivalent to finding the parameters $\theta$ that **minimize a loss function** $L(\theta)$ or **maximize a likelihood**. In other words, we solve problems of the form:

$$
\min_{\theta \in \mathcal{X}} L(\theta),
$$

where $\mathcal{X}$ is the feasible set of parameters.

The landscape of optimization problems varies widely:

- Some functions are **smooth and differentiable**, allowing gradient-based methods.  
- Some functions are **nonsmooth or piecewise**, requiring subgradient or proximal methods.  
- Constraints or bounds on parameters require **projection or constrained optimization**.  
- The scale of the problem (large datasets or high-dimensional parameters) affects algorithm choice.  
- Stochasticity (e.g., noisy gradients from mini-batches) motivates **stochastic optimization** methods.

Choosing the right algorithm involves understanding:

1. **Function properties**: convexity, smoothness, Lipschitz constants.  
2. **Step size and momentum requirements**: trade-off between speed and stability.  
3. **Convergence guarantees**: linear vs. sublinear, deterministic vs. stochastic.  

