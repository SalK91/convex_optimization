# Chapter 10: Multi-Objective Convex Optimization 

Up to now we have focused on problems with a single objective: minimize one convex function over a convex set. However, real-world learning, engineering, and decision-making tasks almost always involve competing criteria:

- loss vs. fairness,
- return vs. risk,
- energy use vs. performance.

Multi-objective optimization provides the mathematical framework for balancing such competing goals. In convex settings, these trade-offs have elegant geometric and analytic structure, captured by Pareto optimality and by scalarization techniques that convert multiple objectives into a single convex problem.


## Classical Optimality (One Objective)

In standard convex optimization, we solve:

$$
x^* \in \arg\min_{x \in \mathcal{X}} f(x),
$$

where $f$ is convex and $\mathcal{X}$ is convex.  
In this setting, it is natural to speak of the minimizer — or set of minimizers — because the task is governed by a single quantitative measure.

However, when multiple objectives $(f_1,\dots,f_k)$ must be minimized simultaneously, a single “best” point usually does not exist.  Improving one objective can worsen another. Multi-objective optimization replaces the idea of a unique minimizer with the idea of efficient trade-offs.



## Multi-Objective Convex Optimization

A multi-objective optimization problem takes the form

$$
\min_{x \in \mathcal{X}} F(x) = (f_1(x), \dots, f_k(x)),
$$

where each $f_i$ is convex.  
This framework appears in many ML and statistical tasks:

| Domain | Objective 1 | Objective 2 | Trade-off |
|-------|-------------|-------------|-----------|
| Regression | Fit error | Regularization | Accuracy vs. complexity |
| Fair ML | Loss | Fairness metric | Utility vs. fairness |
| Portfolio | Return | Risk | Profit vs. stability |
| Autoencoders | Reconstruction | KL divergence | Fidelity vs. disentanglement |

Because objectives typically conflict, one cannot minimize all simultaneously. The natural notion of optimality becomes *Pareto efficiency*.


## Pareto Optimality

### Strong Pareto Optimality

A point $x^*$ is Pareto optimal if there is no other $x$ such that

$$
f_i(x) \le f_i(x^*)\quad \forall i,
$$

with strict inequality for at least one objective. Thus, no trade-off-free improvement is possible: to improve one metric, you must worsen another.

### Weak Pareto Optimality

A point $x^*$ is weakly Pareto optimal if no feasible point satisfies

$$
f_i(x) < f_i(x^*)\quad \forall i.
$$

Weak optimality rules out simultaneous strict improvement in all objectives.

### Geometric View

For two objectives $(f_1, f_2)$, the feasible set in objective space is a region in $\mathbb{R}^2$. Its lower-left boundary, the set of points not dominated by others, is the Pareto frontier.

- Points *on* the frontier are the best achievable trade-offs.
- Points *above* or *inside* the region are dominated and thus suboptimal.

The Pareto frontier explicitly exposes the structure of trade-offs in a problem.


## Scalarization: Turning Many Objectives into One

Multi-objective problems rarely have a unique minimizer. Scalarization constructs a single-objective surrogate problem whose solutions lie on the Pareto frontier.

### Weighted-Sum Scalarization

$$
\min_{x \in \mathcal{X}} \sum_{i=1}^k w_i f_i(x),
\qquad w_i \ge 0,\quad \sum_i w_i = 1.
$$

- The weights encode relative importance.  
- Varying $w$ traces (part of) the Pareto frontier.  
- When $f_i$ and $\mathcal{X}$ are convex, this method recovers the convex portion of the frontier.

### ε-Constraint Method

$$
\min_{x} \ f_1(x)
\quad \text{s.t. } f_i(x) \le \varepsilon_i,\ \ i = 2,\dots,k.
$$

- Here the tolerances $\varepsilon_i$ act as performance budgets.  
- Each choice of $\varepsilon$ yields a different Pareto-efficient point.

This formulation directly highlights the trade-off between one primary objective and several secondary constraints.

### Duality Connection

Scalarization has a tight relationship with duality (Chapter 9):

- Weights $w_i$ in a weighted sum act like dual variables.
- Regularization parameters (e.g., the $\lambda$ in L2 or L1 regularization) correspond to dual multipliers.
- Moving along $\lambda$ traces the Pareto frontier between data fit and model complexity.

This connection explains why tuning regularization is equivalent to choosing a point on a trade-off curve.


## Examples and Applications

### Example 1: Regularized Least Squares

Consider

$$
f_1(x) = \|Ax - b\|_2^2,\qquad 
f_2(x) = \|x\|_2^2.
$$

Two scalarizations:

1. Weighted:
   $$
   \min_x \ \|Ax - b\|_2^2 + \lambda \|x\|_2^2.
   $$

2. ε-constraint:
   $$
   \min_x \ \|Ax - b\|_2^2 \quad \text{s.t. } \|x\|_2^2 \le \tau.
   $$

$\lambda$ and $\tau$ trace the same Pareto curve — the classical bias–variance trade-off.


### Example 2: Portfolio Optimization (Risk–Return)

Let $w$ be portfolio weights, $\mu$ expected returns, and $\Sigma$ the covariance matrix. Objectives:

$$
f_1(w) = -\mu^\top w, \qquad
f_2(w) = w^\top \Sigma w.
$$

Weighted scalarization:

$$
\min_w \ -\alpha \mu^\top w + (1-\alpha) w^\top \Sigma w,
\quad 0 \le \alpha \le 1.
$$

Varying $\alpha$ recovers the efficient frontier of Modern Portfolio Theory.


### Example 3: Fairness–Accuracy in ML

$$
\min_\theta \ \mathbb{E}[\ell(y, f_\theta(x))]
\quad \text{s.t. } D(f_\theta(x),y) \le \varepsilon,
$$

where $D$ is a fairness metric.  
Scalarized form:

$$
\min_\theta\  \mathbb{E}[\ell(y, f_\theta(x))] + \lambda D(f_\theta(x), y).
$$

Tuning $\lambda$ walks across the fairness–accuracy Pareto frontier.


### Example 4: Variational Autoencoders and β-VAE

The ELBO is:

$$
\mathbb{E}_{q(z)}[\log p(x|z)] - \mathrm{KL}(q(z)\|p(z)).
$$

Objectives:

- Reconstruction fidelity,
- Latent simplicity.

β-VAE scalarization:

$$
\max_q \ \mathbb{E}[\log p(x|z)] - \beta \,\mathrm{KL}(q(z)\|p(z)).
$$

$\beta$ controls the trade-off between reconstruction and disentanglement — a Pareto frontier in latent space.


Overall, multi-objective convex optimization extends the geometry and structure of convex analysis to settings with trade-offs and competing priorities. The Pareto frontier reveals the set of achievable compromises, while scalarization methods let us navigate this frontier using tools from single-objective convex optimization, duality, and regularization theory.