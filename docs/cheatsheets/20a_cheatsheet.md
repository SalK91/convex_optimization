# Comprehensive Optimization Algorithm Cheat Sheet

This reference summarizes optimization algorithms across convex optimization, large-scale machine learning, and derivative-free global search.  
It balances **theoretical precision** with **practical intuition**—from gradient-based solvers to black-box evolutionary methods.

---

## 🧭 How to Read This Table

Each method lists:
- **Problem Type** — the class of objectives it applies to.
- **Assumptions** — smoothness, convexity, or structural conditions.
- **Core Update Rule** — canonical iteration.
- **Scalability** — computational feasibility.
- **Per-Iteration Cost** — approximate computational complexity.
- **Applications** — typical ML or engineering use cases.

---

## 🚀 First-Order Methods

| Method | Problem Type | Assumptions | Core Update Rule | Scalability | Per-Iteration Cost | Applications |
|--------|---------------|-------------|------------------|--------------|--------------------|--------------|
| Gradient Descent (GD) | Unconstrained smooth (convex/nonconvex) | Differentiable; $L$-smooth | $x_{k+1} = x_k - \eta \nabla f(x_k)$ | Medium | $O(nd)$ | Logistic regression, least squares |
| Nesterov’s Accelerated GD | Smooth convex (fast rate) | Convex, $L$-smooth | $y_k = x_k + \frac{k-1}{k+2}(x_k - x_{k-1})$; $x_{k+1} = y_k - \eta \nabla f(y_k)$ | Medium | $O(nd)$ | Accelerated convex models |
| (Polyak) Heavy-Ball Momentum | Unconstrained smooth | Differentiable, $\beta \in (0,1)$ | $x_{k+1} = x_k - \eta \nabla f(x_k) + \beta(x_k - x_{k-1})$ | Large | $O(nd)$ | Deep networks, convex smooth losses |
| Conjugate Gradient (CG) | Quadratic or linear systems $Ax=b$ | $A$ symmetric positive definite | $p_{k+1}=r_{k+1}+\beta_k p_k$, $x_{k+1}=x_k+\alpha_k p_k$ | Large | $O(nd)$ | Large-scale least squares, implicit Newton steps |
| Mirror Descent | Non-Euclidean geometry | Convex; mirror map $\psi$ strongly convex | $x_{k+1} = \nabla \psi^*(\nabla \psi(x_k) - \eta \nabla f(x_k))$ | Medium | $O(nd)$ | Probability simplex, online learning |

> *Conjugate Gradient (CG)* bridges first- and second-order methods: it achieves exact convergence in at most $d$ steps for quadratic problems without storing the Hessian, making it ideal for large-scale convex systems.

---

## ⚙️ Second-Order Methods

| Method | Problem Type | Assumptions | Core Update Rule | Scalability | Per-Iteration Cost | Applications |
|--------|---------------|-------------|------------------|--------------|--------------------|--------------|
| Newton’s Method | Smooth convex | Twice differentiable; $\nabla^2 f(x)$ PD | $x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1}\nabla f(x_k)$ | Small–Medium | $O(d^3)$ | Logistic regression (IRLS), convex solvers |
| BFGS / L-BFGS | Smooth convex | Differentiable, approximate Hessian | Solve $B_k p_k=-\nabla f(x_k)$; update $B_k$ via secant rule | Medium | $O(d^2)$ | GLMs, medium ML models |
| Trust-Region | Smooth convex/nonconvex | Twice differentiable | $\min_p \tfrac{1}{2}p^\top \nabla^2 f(x_k)p + \nabla f(x_k)^\top p$ s.t. $\|p\|\le\Delta_k$ | Medium | $O(d^2)$ | TRPO, physics-based ML |

---

## 🧮 Proximal, Projected & Splitting Methods

| Method | Problem Type | Assumptions | Core Update Rule | Scalability | Cost | Applications |
|--------|---------------|-------------|------------------|--------------|------|--------------|
| Proximal Gradient (ISTA) | Composite $f=g+h$ | $g$ smooth, $h$ convex | $x_{k+1}=\operatorname{prox}_{\alpha h}(x_k-\alpha\nabla g(x_k))$ | Medium | $O(nd)$ | LASSO, sparse recovery |
| FISTA | Same as ISTA | Convex, $L$-smooth $g$ | Like ISTA with momentum | Medium | $O(nd)$ | Compressed sensing |
| Projected Gradient (PG) | Convex constrained | $f$ smooth; easy projection | $x_{k+1}=\Pi_C(x_k-\eta\nabla f(x_k))$ | Medium | $O(nd)$ + projection | Box/simplex constraints |
| ADMM | Separable convex + linear constraints | $f,g$ convex | Alternating minimization + dual update | Medium | $O(nd)$ per block | Distributed ML, consensus |
| Majorization–Minimization (MM) | Convex/nonconvex | $g(x|x_k)\ge f(x)$ | $x_{k+1}=\arg\min g(x|x_k)$ | Medium | model dependent | EM, IRLS, robust regression |

---

## 🧩 Coordinate & Block Methods

| Method | Problem Type | Assumptions | Core Update Rule | Scalability | Cost | Applications |
|--------|---------------|-------------|------------------|--------------|------|--------------|
| Coordinate Descent (CD) | Separable convex | Convex, differentiable | Update one coordinate: $x_{i}^{k+1}=x_i^k-\eta\partial_i f(x^k)$ | Large | $O(d)$ | LASSO, SVM duals |
| Block Coordinate Descent (BCD) | Block separable | Convex per block | Minimize over $x^{(j)}$ while fixing others | Large | $O(nd_j)$ | Matrix factorization, alternating minimization |

> *Coordinate descent exploits separability; often faster than full gradient when updates are cheap or sparse.*

---

## 🎲 Stochastic & Mini-Batch Methods

| Method | Problem Type | Assumptions | Core Update Rule | Scalability | Cost | Applications |
|--------|---------------|-------------|------------------|--------------|------|--------------|
| Stochastic Gradient Descent (SGD) | Large-scale / streaming | Unbiased stochastic gradients | $x_{k+1}=x_k-\eta_t\nabla f_{i_k}(x_k)$ | Very Large | $O(bd)$ | Deep learning, online learning |
| Variance-Reduced (SVRG/SAGA/SARAH) | Finite-sum convex | Smooth, strongly convex | $v_k=\nabla f_{i_k}(x_k)-\nabla f_{i_k}(\tilde{x})+\nabla f(\tilde{x})$ | Large | $O(bd)$ | Logistic regression, GLMs |
| Adaptive SGD (Adam/RMSProp/Adagrad) | Nonconvex stochastic | Bounded variance | $m_k=\beta_1m_{k-1}+(1-\beta_1)g_k$, $v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2$ | Very Large | $O(bd)$ | Neural networks |
| Proximal Stochastic (Prox-SGD / Prox-SAGA) | Nonsmooth stochastic | $f=g+h$ with prox of $h$ known | $x_{k+1}=\operatorname{prox}_{\eta h}(x_k-\eta\widehat{\nabla g}(x_k))$ | Large | $O(bd)$ | Sparse online learning |

---

## 🧱 Interior-Point & Augmented Methods

| Method | Problem Type | Assumptions | Core Update Rule | Scalability | Cost | Applications |
|--------|---------------|-------------|------------------|--------------|------|--------------|
| Interior-Point | Convex with inequalities | Slater’s condition, self-concordant barrier | Solve $\min f_0(x)-\tfrac{1}{t}\sum_i\log(-g_i(x))$ | Small–Medium | $O(d^3)$ | LP, QP, SDP |
| Augmented Lagrangian (ALM) | Constrained convex | $f,g$ convex; equality constraints | $L_\rho(x,\lambda)=f(x)+\lambda^T g(x)+\tfrac{\rho}{2}\|g(x)\|^2$ | Medium | $O(nd)$ | Penalty methods, PDEs |

---

## 🌐 Derivative-Free & Black-Box Optimization

| Method | Problem Type | Assumptions | Core Idea | Scalability | Cost | Applications |
|--------|---------------|-------------|------------|--------------|------|--------------|
| Nelder–Mead Simplex | Low-dimensional, smooth or noisy | No gradients; continuous $f$ | Maintain simplex of $d+1$ points; reflect–expand–contract–shrink operations | Small | $O(d^2)$ | Parameter tuning, physics models |
| Simulated Annealing | Nonconvex, global | Stochastic exploration via temperature | Random perturbations accepted w.p. $\exp(-\Delta f/T)$; $T\downarrow$ | Medium | High (many samples) | Hyperparameter tuning, design optimization |
| Multi-start Local Search | Nonconvex | None; relies on restart diversity | Run local solver from multiple random inits, pick best result | Medium | $k\times$ local solver | Avoids local minima; cheap global heuristic |
| Evolutionary Algorithms (EA) | Black-box, global | Population-based; fitness function only | Mutate, select, recombine candidates | Large | $O(Pd)$ per gen | Global optimization, control, AutoML |
| Genetic Algorithms (GA) | Combinatorial / continuous | Chromosomal encoding of solutions | Apply selection, crossover, mutation; evolve over generations | Medium–Large | $O(Pd)$ | Feature selection, neural architecture search |
| Evolution Strategies (ES) | Continuous, black-box | Gaussian mutation around mean | $\theta_{k+1} = \theta_k + \eta \sum_i w_i \epsilon_i f(\theta_k+\sigma \epsilon_i)$ | Large | $O(Pd)$ | Reinforcement learning, black-box control |
| Derivative-Free Optimization (DFO) | Black-box, noisy $f$ | Only function values available | Gradient estimated via random perturbations: $g\approx\frac{f(x+hu)-f(x)}{h}u$ | Medium | $O(d)$–$O(d^2)$ | Robotics, policy search, design |
| Black-Box Optimization Framework | General | No analytical gradients; often stochastic | Unified term covering EA, GA, ES, and DFO | Medium–Large | varies | Hyperparameter search, AutoML, reinforcement learning |
| Numerical Encodings | Used in GA/EA | Represents variables in binary, integer, or floating-point form | Choice of encoding impacts mutation/crossover behavior | N/A | negligible | Optimization of mixed or discrete variables |

> *Black-box and evolutionary methods trade theoretical guarantees for robustness and global search power. They are essential when gradients are unavailable or noninformative.*

---

## 📈 Convergence & Complexity Snapshot

| Method Type | Convergence (Convex) | Notes |
|--------------|----------------------|-------|
| Subgradient | $O(1/\sqrt{k})$ | Nonsmooth convex |
| Gradient Descent | $O(1/k)$ | Smooth convex |
| Accelerated Gradient | $O(1/k^2)$ | Optimal first-order |
| Newton / Quasi-Newton | Quadratic / Superlinear | Local only |
| Strongly Convex | $(1-\mu/L)^k$ | Linear rate |
| Variance-Reduced | Linear (strongly convex) | Finite-sum optimization |
| ADMM / Proximal | $O(1/k)$ | Composite convex |
| Interior-Point | Polynomial time | High-accuracy convex |
| Derivative-Free / Heuristics | No formal bound | Empirical convergence only |

---

## 🧠 Practitioner Summary

| Situation | Recommended Methods |
|------------|--------------------|
| Gradients available, smooth convex | Gradient Descent, Nesterov |
| Curvature matters, moderate scale | Newton, BFGS, Conjugate Gradient |
| Nonsmooth regularizer | Proximal Gradient, ADMM |
| Simple constraints | Projected Gradient |
| Large-scale / streaming | SGD, Adam, RMSProp |
| Finite-sum convex | SVRG, SAGA |
| Online / adaptive | Mirror Descent, FTRL |
| No gradients (black-box) | DFO, Nelder–Mead, ES, GA |
| Global nonconvex search | Simulated Annealing, Multi-starts, Evolutionary Algorithms |
| Distributed / separable | ADMM, ALM |
| High-precision convex programs | Interior-Point, Trust-Region |

---

### 🧩 Notes on Global & Black-Box Optimization

- **Conjugate Gradient**: memory-efficient quasi-second-order method for large convex quadratics.  
- **Nelder–Mead**: simplex reflection algorithm; widely used in physics and hyperparameter tuning.  
- **Simulated Annealing**: probabilistic global search inspired by thermodynamics.  
- **Multi-Starts**: pragmatic global exploration by repeated local optimization.  
- **Evolutionary / Genetic / ES**: population-based global heuristics; robust to noise and discontinuity.  
- **Derivative-Free Optimization (DFO)**: umbrella for random, surrogate-based, or adaptive black-box methods.  
- **Numerical Encoding**: crucial in discrete search—how real or binary variables are represented determines performance.

---

> **Summary Insight:**  
> - Convex + differentiable → use gradient-based or Newton-type methods.  
> - Convex + nonsmooth → use proximal, ADMM, or coordinate descent.  
> - Large-scale or stochastic → use SGD or adaptive variants.  
> - No gradients or nonconvex → use derivative-free or evolutionary methods.  
> - The structure of the objective, not its size alone, determines the optimal solver family.

