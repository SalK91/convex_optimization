
## First-Order Gradient-Based Methods

**Used when:** Only gradient information is available; scalable to high-dimensional problems.

### Gradient Descent (GD)
- **Problem:** Minimize smooth or convex functions.  
- **Update:** 
$$
x^{k+1} = x^k - \alpha \nabla f(x^k)
$$  
- **Convergence:** Convex → $O(1/k)$; Strongly convex → linear.  
- **Use case:** Small convex problems, theoretical baseline.  
- **Pitfalls:** Step size too small → slow; too large → divergence.

### Stochastic Gradient Descent (SGD)
- **Problem:** Minimize empirical risk over large datasets.  
- **Update:** 
$$
x^{k+1} = x^k - \alpha \nabla f_B(x^k)
$$ 
(mini-batch gradient)  
- **Pros:** Scales to huge datasets; cheap per iteration.  
- **Cons:** Noisy updates → requires learning rate schedules.  
- **ML use:** Deep learning, large-scale logistic regression.  

**Best Practices:** Learning rate warmup, linear scaling with batch size, momentum to stabilize updates, cyclic learning rates for exploration.

### Momentum & Nesterov Accelerated Gradient
- **Problem:** Reduce oscillations and accelerate convergence in ill-conditioned problems.  
- **Momentum:** 
$$
v_{t+1} = \beta v_t + \nabla f_B(x_t), \quad x_{t+1} = x_t - \alpha v_{t+1}
$$  
- **Nesterov:** Gradient computed at lookahead point → theoretically optimal for convex problems.  
- **ML use:** CNNs, ResNets, EfficientNet.  
- **Pitfalls:** High momentum → oscillations; careful learning rate tuning required.

### Adaptive Methods (AdaGrad, RMSProp, Adam, AdamW)
- **Problem:** Adjust learning rate per parameter for fast/stable convergence.  
- **Behavior:**  
  - AdaGrad → aggressive decay, good for sparse features.  
  - RMSProp → fixes AdaGrad’s rapid decay.  
  - Adam → RMSProp + momentum.  
  - AdamW → decouples weight decay for better generalization.  
- **ML use:** Transformers, NLP, sparse models.  
- **Pitfalls:** Adam may converge to sharp minima → worse generalization than SGD in CNNs.  
- **Best Practices:** Warmup, cosine LR decay, weight decay with AdamW.



## Second-Order & Curvature-Aware Methods

**Used when:** Hessian or curvature information improves convergence; mostly for small/medium models.

### Newton’s Method
- **Problem:** Solve $ \min f(x) $ with smooth Hessian.  
- **Update:** 
$$
x^{k+1} = x^k - [\nabla^2 f(x^k)]^{-1} \nabla f(x^k)
$$  
- **Pros:** Quadratic convergence.  
- **Cons:** Hessian expensive in high dimensions.  
- **ML use:** GLMs, small convex models.

### Quasi-Newton (BFGS, L-BFGS)
- **Problem:** Approximate Hessian using low-rank updates.  
- **Pros:** Efficient for medium-scale problems.  
- **Cons:** BFGS memory-heavy; L-BFGS preferred.  
- **ML use:** Logistic regression, Cox models.

### Conjugate Gradient
- **Problem:** Solve large linear/quadratic problems efficiently.  
- **ML use:** Hessian-free optimization; combined with Pearlmutter trick for Hessian-vector products.

### Natural Gradient & K-FAC
- **Problem:** Precondition gradients using Fisher Information → invariant to parameterization.  
- **ML use:** Large CNNs, transformers; improves convergence in distributed training.


## Constrained & Specialized Optimization

### Interior-Point
- **Problem:** Constrained optimization via barrier functions.  
- **ML use:** Structured convex problems, LP/QP.

### ADMM / Augmented Lagrangian
- **Problem:** Split constraints into easier subproblems with dual updates.  
- **ML use:** Distributed optimization, structured sparsity.  

### Frank–Wolfe
- **Problem:** Projection-free constrained optimization; linear subproblem instead of projection.  
- **ML use:** Simplex, nuclear norm problems.

### Coordinate Descent
- **Problem:** Update one variable at a time.  
- **ML use:** Lasso, GLMs, sparse regression.

### Proximal Methods
- **Problem:** Efficiently handle nonsmooth penalties.  
- **Algorithms:** ISTA ($O(1/k)$), FISTA ($O(1/k^2)$)  
- **ML use:** Sparse coding, Lasso, elastic net.

### Derivative-Free / Black-Box
- **Problem:** Optimize when gradients unavailable or unreliable.  
- **Algorithms:** Nelder–Mead, CMA-ES, Bayesian Optimization  
- **ML use:** Hyperparameter tuning, neural architecture search, small networks.


## Optimization Problem Styles

### Maximum Likelihood Estimation (MLE)
- **Problem:** Maximize likelihood or minimize negative log-likelihood:  
$$
\hat{\theta} = \arg\max_\theta \sum_i \log p(x_i|\theta)
$$  
- **Algorithms:** Newton/Fisher scoring, L-BFGS, SGD, EM, Proximal/Coordinate.  
- **ML use:** Logistic regression, GLMs, Gaussian mixture models, HMMs.  
- **Notes:** EM guarantees monotonic likelihood increase; Fisher scoring uses expected curvature → stable.

### Empirical Risk Minimization (ERM)
- **Problem:** Minimize average loss with optional regularization:  
$$
\min_\theta \frac{1}{n} \sum_i \mathcal{L}(f_\theta(x_i),y_i) + R(\theta)
$$  
- **Algorithms:** GD, SGD, Momentum, Adam, L-BFGS, Proximal.  
- **ML use:** Regression, classification, deep learning.

### Regularized / Penalized Optimization
- **Problem:** Add penalties to encourage sparsity or smoothness:  
$$
\min_\theta f(\theta)+\lambda g(\theta)
$$  
- **Algorithms:** Proximal gradient, ADMM, Coordinate Descent, ISTA/FISTA.  
- **ML use:** Lasso, Elastic Net, sparse dictionary learning.

### Constrained Optimization
- **Problem:** Minimize with equality/inequality constraints.  
- **Algorithms:** Interior-point, ADMM, Frank–Wolfe, penalty/barrier methods.  
- **ML use:** Fairness constraints, structured prediction.

### Bayesian / MAP Optimization
- **Problem:** Maximize posterior:  
$$
\min_\theta -\log p(X|\theta) - \log p(\theta)
$$  
- **Algorithms:** Gradient-based, Laplace approximation, Variational Inference, MCMC.  
- **ML use:** Bayesian neural networks, probabilistic models.

### Minimax / Adversarial Optimization
- **Problem:** 
$$
\min_\theta \max_\phi f(\theta,\phi)
$$  
- **Algorithms:** Gradient descent/ascent, extragradient, mirror descent.  
- **ML use:** GANs, adversarial training, robust optimization.

### Reinforcement Learning / Policy Optimization
- **Problem:** Maximize expected cumulative reward:  
$$
\max_\theta \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_t r(s_t,a_t)\Big]
$$  
- **Algorithms:** Policy gradient, Actor-Critic, Natural Gradient.  
- **ML use:** RL agents, sequential decision-making.

### Multi-Objective Optimization
- **Problem:** Optimize multiple competing objectives → Pareto front.  
- **Algorithms:** Scalarization, weighted sum, evolutionary algorithms.  
- **ML use:** Multi-task learning, accuracy vs fairness trade-offs.

### Metric / Embedding Learning
- **Problem:** Learn embeddings preserving similarity/distance:  
$$
\min_\theta \sum_{i,j}\ell(d(f_\theta(x_i),f_\theta(x_j)),y_{ij})
$$  
- **Algorithms:** SGD/Adam with careful sampling.  
- **ML use:** Contrastive learning, triplet loss, Siamese networks.

### Combinatorial / Discrete Optimization
- **Problem:** Optimize discrete/integer variables.  
- **Algorithms:** Branch-and-bound, integer programming, RL-based relaxation, Gumbel-softmax.  
- **ML use:** Feature selection, neural architecture search, graph matching.

### Derivative-Free / Black-Box
- **Problem:** Gradients unavailable or noisy.  
- **Algorithms:** Bayesian Optimization, CMA-ES, Nelder–Mead.  
- **ML use:** Hyperparameter tuning, neural architecture search, small networks.

---

## Learning Rate & Practical Tips
- Step decay, cosine annealing, OneCycle, warmup.  
- Gradient clipping (global norm 1–5), batch/layer normalization, FP16 mixed precision.  
- Decouple weight decay from Adam (AdamW).

---

## Summary
| Algorithm | Problem Type | ML / AI Use Case |
|-----------|-------------|----------------|
| GD | Smooth / convex | Small convex models, baseline |
| SGD | Large-scale ERM | Deep learning, logistic regression |
| SGD + Momentum | Ill-conditioned / deep nets | CNNs (ResNet, EfficientNet) |
| Nesterov Accelerated GD | Convex / ill-conditioned | CNNs, small convex models |
| AdaGrad | Sparse features | NLP, sparse embeddings |
| RMSProp | Stabilized adaptive LR | RNNs, sequence models |
| Adam | Adaptive large-scale | Transformers, small nets |
| AdamW | Adaptive + weight decay | Transformers, NLP |
| Newton / Fisher Scoring | Smooth convex | GLMs, small MLE |
| BFGS / L-BFGS | Medium convex | Logistic regression, Cox models |
| Conjugate Gradient | Linear / quadratic | Hessian-free optimization, linear regression |
| Natural Gradient / K-FAC | Deep nets | CNNs, transformers |
| Proximal / ISTA / FISTA | Nonsmooth / sparse | Lasso, sparse coding, elastic net |
| Coordinate Descent | Separable / sparse | Lasso, GLMs |
| Interior-Point | Constrained convex | LP/QP problems |
| ADMM | Distributed convex | Sparse or structured optimization |
| Frank–Wolfe | Projection-free constraints | Simplex, nuclear norm problems |
| EM Algorithm | Latent variable MLE | GMM, HMM, LDA |
| Policy Gradient / Actor-Critic | Sequential / RL | RL agents |
| Bayesian Optimization | Black-box / derivative-free | Hyperparameter tuning, NAS |
| CMA-ES / Nelder-Mead | Black-box | Small networks, continuous black-box |
| Minimax / Gradient Ascent-Descent | Adversarial | GANs, robust optimization |
| Multi-Objective / Evolutionary | Multiple objectives | Multi-task learning, fairness |
| Metric Learning / Triplet Loss | Similarity embedding | Contrastive learning, Siamese nets |

