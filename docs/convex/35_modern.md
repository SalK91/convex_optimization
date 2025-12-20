# Chapter 18: Modern Optimizers in Machine Learning

The past decade has seen an explosion of nonconvex optimization problems, driven largely by deep learning. Training neural networks, large language models, and reinforcement learning agents all depend on stochastic optimization—balancing accuracy, generalization, and efficiency on massive, noisy datasets.

This chapter connects the principles of convex optimization to the modern optimizers that power today’s machine learning systems. While these algorithms often lack formal global guarantees, they are remarkably effective in practice.

 
## Stochastic Optimization Overview

In machine learning, we often minimize an empirical risk:
$$
\min_{x \in \mathbb{R}^n} \; f(x) = \frac{1}{N} \sum_{i=1}^N \ell(x; z_i),
$$
where $\ell(x; z_i)$ is the loss on data sample $z_i$.

Computing the full gradient $\nabla f(x)$ is infeasible when $N$ is large. Instead, stochastic methods estimate it using a mini-batch of samples:

$$
g_k = \frac{1}{|B_k|} \sum_{i \in B_k} \nabla \ell(x_k; z_i).
$$

This yields the Stochastic Gradient Descent (SGD) update:

$$
x_{k+1} = x_k - \alpha_k g_k.
$$

SGD is the foundation for nearly all deep learning optimizers.

## Momentum and Acceleration

SGD’s noisy gradients can cause slow convergence and oscillations. Momentum smooths the update by accumulating a moving average of past gradients:

$$
v_{k+1} = \beta v_k + (1-\beta) g_k, \quad x_{k+1} = x_k - \alpha v_{k+1},
$$
where $\beta \in [0,1)$ controls inertia.

Nesterov momentum adds a correction term anticipating the future position:

$$
v_{k+1} = \beta v_k + g(x_k - \alpha \beta v_k), \quad x_{k+1} = x_k - \alpha v_{k+1}.
$$

Momentum-based methods help traverse ravines and saddle regions efficiently.


## Adaptive Learning Rate Methods

Different parameters often require different step sizes.  
Adaptive methods adjust learning rates automatically using the history of squared gradients.

### AdaGrad
Keeps a cumulative sum of squared gradients:

$$
G_k = \sum_{t=1}^k g_t \odot g_t,
$$
and updates parameters as:

$$
x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_k + \epsilon}} \odot g_k.
$$
Good for sparse data, but the learning rate can shrink too quickly.


### RMSProp
A refinement of AdaGrad using exponential averaging:

$$
E[g^2]_k = \beta E[g^2]_{k-1} + (1-\beta) g_k^2,
$$

$$
x_{k+1} = x_k - \frac{\alpha}{\sqrt{E[g^2]_k + \epsilon}} g_k.
$$

RMSProp prevents the learning rate from vanishing and works well for nonstationary objectives.


### Adam: Adaptive Moment Estimation
Adam combines momentum and adaptive scaling:

$$
m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k, \quad v_k = \beta_2 v_{k-1} + (1-\beta_2) g_k^2,
$$

$$
\hat{m}_k = \frac{m_k}{1-\beta_1^k}, \quad \hat{v}_k = \frac{v_k}{1-\beta_2^k},
$$

$$
x_{k+1} = x_k - \alpha \frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon}.
$$

Adam adapts quickly to changing gradient scales, converging faster than vanilla SGD.


## Variants and Modern Extensions

| Optimizer | Key Idea | Notes |
|----------------|---------------|------------|
| AdamW | Decoupled weight decay from gradient update | Better regularization |
| RAdam | Rectified Adam—adaptive variance correction | Improves stability early in training |
| Lookahead | Combines fast and slow weights | Enhances robustness and convergence |
| AdaBelief | Uses prediction error instead of raw gradient variance | More adaptive learning rates |
| Lion | Uses sign-based updates and momentum | Efficient for large-scale training |

These variants represent the frontier of stochastic optimization in deep learning frameworks.

 

## Implicit Regularization and Generalization

Modern optimizers not only minimize loss—they also affect generalization. SGD and its variants exhibit implicit bias toward flat minima, which often correspond to models with better generalization properties.

Empirical findings suggest:

- Large-batch training finds sharper minima (risk of overfitting).  
- Noisy, small-batch SGD promotes flat, generalizable minima.  
- Adaptive optimizers may converge faster but generalize slightly worse.

This trade-off drives ongoing research into optimizer design.


## Practical Considerations

| Aspect | Guideline |
|-------------|---------------|
| Learning Rate | Most critical hyperparameter; use warm-up and decay schedules |
| Batch Size | Balances gradient noise and hardware efficiency |
| Initialization | Affects early dynamics, especially for Adam variants |
| Gradient Clipping | Prevents instability in exploding gradients |
| Mixed Precision | Use with adaptive optimizers for speed and memory savings |


## Comparative Behavior

| Method | Adaptivity | Speed | Memory | Typical Use |
|-------------|----------------|------------|--------------|-----------------|
| SGD + Momentum | Moderate | Slow-medium | Low | General-purpose, good generalization |
| RMSProp | Adaptive per-parameter | Medium-fast | Medium | Recurrent networks, nonstationary data |
| Adam / AdamW | Fully adaptive | Fast | High | Deep networks, large-scale training |
| RAdam / AdaBelief / Lion | Advanced adaptivity | Fast | Medium | Cutting-edge training tasks |



## Optimization in Modern Deep Networks

In deep learning, optimization interacts with architecture, loss, and regularization:

- Batch normalization modifies effective learning rates.  
- Skip connections ease gradient flow.  
- Large-scale distributed training relies on adaptive optimizers for stability.  

Optimization is no longer an isolated procedure but part of the model’s design philosophy.


Modern stochastic optimizers extend classical first-order methods into high-dimensional, noisy, nonconvex regimes. They are the engines behind deep learning—adapting dynamically, balancing efficiency and generalization.
