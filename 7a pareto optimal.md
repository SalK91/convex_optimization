# Pareto Optimality 

## Classical Optimality

In standard convex optimisation, we consider a **single objective function** $f(x)$ and aim to find a globally optimal solution:
$$
x^* \in \arg \min_{x \in \mathcal{X}} f(x),
$$
where $\mathcal{X}$ is the feasible set.  

Here, optimality is absolute: there exists a single best point (or set of equivalent best points) with respect to one measure of performance.


## Multi-objective Optimisation

Many practical problems in machine learning and optimisation involve **multiple competing objectives**. For instance:

- In **supervised learning**, one wishes to minimise prediction error while also controlling model complexity.  
- In **fairness-aware learning**, we want high accuracy while limiting demographic disparity.  
- In **finance**, an investor balances expected return against risk.  

Formally, a multi-objective optimisation problem is written as:
$$
\min_{x \in \mathcal{X}} F(x) = (f_1(x), f_2(x), \dots, f_k(x)),
$$
where $f_1, f_2, \dots, f_k$ are the competing objectives.


## Pareto Optimality

### Strong Pareto Optimality
A solution $x^* \in \mathcal{X}$ is **Pareto optimal** if there is no $x \in \mathcal{X}$ such that:
$$
f_i(x) \leq f_i(x^*) \quad \text{for all } i,
$$
with strict inequality for at least one objective $j$.  

Intuitively, no feasible point strictly improves one objective without worsening another.

### Weak Pareto Optimality
A solution $x^*$ is **weakly Pareto optimal** if there is no $x \in \mathcal{X}$ such that:
$$
f_i(x) < f_i(x^*) \quad \text{for all } i.
$$

In other words, no solution improves *every* objective simultaneously.

### Geometric Intuition
If we plot feasible solutions in the objective space $(f_1(x), f_2(x))$, the Pareto frontier is the **lower-left boundary** for minimisation problems.  
- Points on the frontier are non-dominated (Pareto optimal).  
- Points inside the feasible region but above the boundary are dominated.  



## Scalarisation

Since multi-objective optimisation problems usually admit a set of Pareto optimal solutions rather than a single best point, practitioners use **scalarisation**. This reduces multiple objectives to a single scalar objective that can be optimised with standard methods.

### Weighted Sum Scalarisation
The most common approach is the weighted sum:
$$
\min_{x \in \mathcal{X}} \; \sum_{i=1}^k w_i f_i(x),
\quad w_i \geq 0, \quad \sum_{i=1}^k w_i = 1.
$$

- Each choice of weights $w$ corresponds to a different point on the Pareto frontier.  
- Larger $w_i$ prioritises objective $f_i$ relative to others.  

**Convexity caveat:**  
If the feasible set and objectives are convex, weighted sum scalarisation can recover the **convex part** of the Pareto frontier. Non-convex regions of the frontier may not be attainable using weighted sums alone.

### $\varepsilon$-Constraint Method
Another approach is to optimise one objective while converting others into constraints:
$$
\min_x f_1(x) \quad \text{s.t. } f_i(x) \leq \varepsilon_i, \quad i = 2, \dots, k.
$$
Here $\varepsilon_i$ are tolerance levels. By adjusting them, we can explore different trade-offs.  

This connects directly to **regularisation in machine learning**:  
- In ridge regression, we minimise data fit subject to a complexity budget $\|x\|_2^2 \leq \tau$.  
- The equivalent penalised form $\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2$ is obtained via Lagrangian duality, where $\lambda$ is the multiplier associated with $\tau$.  

### Duality and Scalarisation
Scalarisation is deeply connected to **duality** in convex optimisation:  
- The weights $w_i$ or multipliers $\lambda$ can be interpreted as Lagrange multipliers balancing objectives.  
- Adjusting these parameters changes the point on the Pareto frontier that is selected.  
- This explains why hyperparameters like $\lambda$ in regularisation are so influential: they represent trade-offs in a hidden multi-objective problem.



## Example 1: Regularised Least Squares

Consider the regression problem with data matrix $A \in \mathbb{R}^{m \times n}$ and target $b \in \mathbb{R}^m$.  

We want to minimise:
1. Prediction error: $f_1(x) = \|Ax - b\|_2^2$  
2. Model complexity: $f_2(x) = \|x\|_2^2$  

This is a two-objective optimisation problem.  

- Using the **weighted sum**:
$$
\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2,
$$
where $\lambda \geq 0$ determines the trade-off.  

- Alternatively, using the **$\varepsilon$-constraint**:
$$
\min_x \|Ax - b\|_2^2 \quad \text{s.t. } \|x\|_2^2 \leq \tau.
$$

Both formulations yield Pareto optimal solutions, with $\lambda$ and $\tau$ providing different parametrisations of the frontier.

## Example 2: Portfolio Optimisation (Risk–Return)

In finance, suppose an investor chooses portfolio weights $w \in \mathbb{R}^n$.

- Expected return: $f_1(w) = -\mu^\top w$ (we minimise negative return).  
- Risk: $f_2(w) = w^\top \Sigma w$ (variance of returns, a convex function).  

The problem is:
$$
\min_w \big( -\mu^\top w, \; w^\top \Sigma w \big).
$$

- Using weighted sum scalarisation:
$$
\min_w \; -\alpha \mu^\top w + (1-\alpha) w^\top \Sigma w,
\quad 0 \leq \alpha \leq 1.
$$
- Different $\alpha$ values give different points on the **efficient frontier**.  

This convex formulation underpins modern portfolio theory.



## Example 3: Probabilistic Modelling (ELBO)

In variational inference, the Evidence Lower Bound (ELBO) is:
$$
\text{ELBO} = \mathbb{E}_{q(z)}[\log p(x|z)] - \text{KL}(q(z) \| p(z)).
$$

This can be seen as a scalarisation of two competing objectives:  
1. Data fit (reconstruction term).  
2. Simplicity or prior adherence (KL divergence).  

By weighting the KL divergence with a parameter $\beta$, we obtain the $\beta$-VAE:
$$
\max_q \; \mathbb{E}_{q(z)}[\log p(x|z)] - \beta \, \text{KL}(q(z) \| p(z)).
$$

Here, $\beta$ plays the role of a scalarisation weight, selecting different Pareto optimal trade-offs between reconstruction accuracy and disentanglement.



## Broader Connections in AI and ML

- **Fairness vs accuracy:** Balancing accuracy with fairness metrics is a multi-objective problem often approached via scalarisation.  
- **Generalisation vs training error:** Regularisation is a scalarisation of fit versus complexity.  
- **Compression vs performance:** The information bottleneck principle is a Pareto trade-off between accuracy and representation complexity.  
- **Inference vs divergence:** Variational inference (ELBO) is naturally a scalarised multi-objective problem.  



## Summary

- **Classical optimisation** yields a single best solution.  
- **Multi-objective optimisation** gives a set of non-dominated (Pareto optimal) solutions.  
- **Scalarisation** provides practical methods to compute Pareto optimal solutions.  
- **Weighted sums** recover convex parts of the frontier, while **$\varepsilon$-constraints** provide flexibility.  
- Scalarisation connects directly to **duality**, where weights act as Lagrange multipliers.  
- Examples in ML (ridge regression, ELBO) and finance (portfolio optimisation) demonstrate its wide relevance.  

Scalarisation is not only a mathematical device but the foundation for understanding regularisation, fairness, generalisation, and many practical trade-offs in machine learning.

