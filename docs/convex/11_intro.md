# Chapter 1: Introduction and Overview

Optimization is the mathematical language of decision-making and learning. In machine learning, statistics, and control, nearly every task — from fitting a regression line to training a neural network — can be viewed as minimizing (or maximizing) an objective function under constraints.

Yet most real-world objectives are nonlinear and non-convex, riddled with local minima. Convex optimization isolates the *tractable core* of this vast landscape: a class of problems where geometry guarantees both theoretical and computational simplicity.

 
## 1.1 Why Convex Optimization Matters

A function or problem is *convex* when it has a single, “bowl-shaped” valley — any local minimum is automatically global.  
This simple geometric property provides powerful consequences:

- Mathematical guarantees of optimality – no spurious minima or saddle points.  
- Efficient algorithms – gradient and interior-point methods converge reliably.  
- Robustness to perturbations – small changes in data yield small changes in solutions.  
- Strong duality and verifiable bounds – via convex dual problems.

> Convexity ⇒ Robustness.  
> Convex problems are stable: small perturbations to inputs cause proportionally small shifts in the solution.  
> Many difficult non-convex problems are attacked by constructing convex relaxations, whose solutions yield bounds or high-quality approximations.

 

## 1.2 Canonical Convex Problem Form

A general convex optimization problem is written as

$$
\begin{array}{ll}
\text{minimize}_{x} & f(x) \\
\text{subject to} & g_i(x) \le 0, \quad i = 1,\dots,m, \\
                  & h_j(x) = 0, \quad j = 1,\dots,p,
\end{array}
$$

where  

- each $f$ and $g_i$ is convex,  
- each $h_j$ is affine, i.e., $h_j(x)=a_j^\top x+b_j$.

> Note: Equalities must be affine for the feasible region to remain convex.  
> Nonlinear equality constraints can destroy convexity.

A problem is convex only when both the objective and the feasible set are convex.
