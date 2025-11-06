# Chapter 10: Pareto Optimality and Multi-Objective Convex Optimization

Optimization often focuses on a single objective function — minimizing one measure of performance. However, real-world problems rarely involve a single criterion. In practice, we must balance multiple conflicting goals: accuracy vs. complexity, fairness vs. utility, return vs. risk, etc.  

This chapter introduces Pareto optimality, which generalizes classical convex optimization to the multi-objective setting, and explores how scalarisation connects multi-objective problems to duality and regularisation.
 
## 10.1 Classical Optimality

In standard convex optimization, we minimize one function:
\[
x^* \in \arg\min_{x \in \mathcal{X}} f(x),
\]
where $\mathcal{X}$ is a convex feasible set.

Here, optimality is absolute — there exists a single best point (or equivalence class) minimizing one measure of performance.  


But what if we have multiple objectives that cannot all be minimized simultaneously?


## 10.2 Multi-Objective Convex Optimization

In many learning and design problems, several objectives compete:

| Application | Objective 1 | Objective 2 | Trade-off |
|--------------|--------------|--------------|------------|
| Regression | Fit error | Regularization | Accuracy vs complexity |
| Fair ML | Prediction loss | Fairness metric | Accuracy vs fairness |
| Portfolio design | Return | Risk | Profit vs stability |
| Information theory | Accuracy | Compression | Fit vs simplicity |

Formally, we write:
\[
\min_{x \in \mathcal{X}} F(x) = (f_1(x), f_2(x), \dots, f_k(x)),
\]
where each $f_i(x)$ is convex.  
The solution concept changes: there is no single global minimum. Instead, there is a *set* of trade-off solutions.



## 10.3 Pareto Optimality

### (a) Strong Pareto Optimality

A point $x^* \in \mathcal{X}$ is Pareto optimal if no other $x \in \mathcal{X}$ satisfies:
\[
f_i(x) \le f_i(x^*) \quad \forall i,
\]
with strict inequality for at least one $j$.

Intuitively: no feasible solution can improve one objective without worsening another.

### (b) Weak Pareto Optimality

A point $x^*$ is weakly Pareto optimal if no $x$ satisfies:
\[
f_i(x) < f_i(x^*) \quad \forall i.
\]
That is, no feasible solution strictly improves all objectives simultaneously.

### (c) Geometric Intuition

In two dimensions $(f_1, f_2)$, the Pareto frontier forms the *lower-left boundary* of the feasible region (for minimization):

- Points *on* the frontier are Pareto optimal — non-dominated.  
- Points *above* or *inside* are dominated (inferior in all respects).

> The frontier visualizes the fundamental trade-offs in the problem.

 
## 10.4 Scalarisation: Reducing Many Objectives to One

Since multi-objective problems rarely have a unique minimizer, we often scalarise them — combine all objectives into a single composite scalar that we can minimize using standard methods.

### (a) Weighted Sum Scalarisation

\[
\min_{x \in \mathcal{X}} \; \sum_{i=1}^k w_i f_i(x),
\quad w_i \ge 0,\quad \sum_i w_i = 1.
\]

- The weights $w_i$ encode relative importance of objectives.
- Each choice of $w$ yields a different point on the Pareto frontier.
- Larger $w_i$ emphasizes objective $f_i$.

Convexity caveat:  
If the objectives and feasible set are convex, the weighted-sum method recovers the convex portion of the Pareto frontier. Nonconvex parts cannot be reached with simple weights.

 

### (b) $\varepsilon$-Constraint Scalarisation

Alternatively, minimize one objective while turning others into constraints:
\[
\min_x f_1(x) \quad \text{s.t. } f_i(x) \le \varepsilon_i,\; i=2,\dots,k.
\]

- The tolerances $\varepsilon_i$ act as *performance budgets*.  
- Varying them explores different Pareto-optimal trade-offs.  

Example connection:  
Ridge regression minimizes fit error subject to a bound on model complexity:
\[
\min_x \|Ax - b\|_2^2 \quad \text{s.t. } \|x\|_2^2 \le \tau.
\]
The Lagrangian form
\[
\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2
\]
is a weighted-sum scalarisation — $\lambda$ acts as a trade-off parameter.

 
### (c) Duality and Scalarisation

Scalarisation is closely related to duality (Chapter 9):

- The weights $w_i$ or Lagrange multipliers $\lambda_i$ act as dual variables.  
- Changing them selects different Pareto-optimal points on the frontier.  
- Regularisation parameters in ML (like $\lambda$) are dual to constraint levels — they *move along* the Pareto frontier.

 

## 10.5 Examples

### Example 1 – Regularised Least Squares

Objectives:
\[
f_1(x)=\|Ax-b\|_2^2, \quad f_2(x)=\|x\|_2^2.
\]

Two equivalent formulations:

1. Weighted sum:
   \[
   \min_x \|Ax-b\|_2^2 + \lambda \|x\|_2^2.
   \]
2. $\varepsilon$-constraint:
   \[
   \min_x \|Ax-b\|_2^2 \quad \text{s.t. } \|x\|_2^2 \le \tau.
   \]

Both yield Pareto optimal solutions; $\lambda$ and $\tau$ parameterize the same trade-off curve.

 

### Example 2 – Portfolio Optimization (Risk–Return)

Decision variable: portfolio weights \( w \in \mathbb{R}^n \).  
Objectives:
\[
f_1(w) = -\mu^\top w \quad (\text{negative return}), \qquad
f_2(w) = w^\top \Sigma w \quad (\text{risk}).
\]

Weighted sum formulation:
\[
\min_w \; -\alpha \mu^\top w + (1-\alpha) w^\top \Sigma w,
\quad 0 \le \alpha \le 1.
\]

- Varying $\alpha$ traces the efficient frontier in risk–return space.
- This is the foundation of Modern Portfolio Theory (Markowitz).

 

### Example 3 – Probabilistic Modelling (ELBO and $\beta$-VAE)

The Evidence Lower Bound (ELBO) in variational inference:
\[
\text{ELBO} = \mathbb{E}_{q(z)}[\log p(x|z)] - \mathrm{KL}(q(z)\|p(z)).
\]

Two competing objectives:
- Data fit: maximize expected log-likelihood.  
- Simplicity: minimize KL divergence.

Scalarised form (\(\beta\)-VAE):
\[
\max_q \; \mathbb{E}_{q(z)}[\log p(x|z)] - \beta\, \mathrm{KL}(q(z)\|p(z)).
\]

Parameter $\beta$ controls the trade-off — different $\beta$ yield different Pareto-optimal points between reconstruction accuracy and disentanglement.
