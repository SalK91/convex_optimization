# Chapter 1: Introduction and Overview

Optimization is the mathematical language of decision-making and learning. In machine learning and data science we constantly fit, estimate, or infer by minimizing a loss. Yet most real-world objectives are nonlinear and messy; solving them globally can be hard.

Convex optimization is the *tractable core* of this landscape.  
A convex problem has a single “bowl-shaped” valley, no false minima, so any local optimum is automatically global.  
That geometric simplicity gives us:

- mathematical guarantees of optimality,  
- efficient algorithms that scale,  
- robustness to noise and initialization.
 
> Convexity leads to robustness. Small perturbations to problem data typically result in small changes to the solution in convex optimization, a desirable feature in practice. Moreover, many approximation techniques (like convex relaxations of NP-hard problems) rely on solving a convex problem to get bounds or heuristic solutions for the original task. 


## 1.2 Canonical Convex Problem Form

A generic convex optimization problem can be written as:

$$
\begin{array}{ll}
\text{minimize}_{x} & f(x) \\
\text{subject to} & g_i(x) \le 0,\quad i = 1,\dots,m, \\
                  & h_j(x) = 0,\quad j = 1,\dots,p,
\end{array}
$$

where each $f$ and $g_i$ is convex, and each $h_j$ is affine (affine functions are both convex and concave).


## 1.3 How the Web-Book Is Structured

This Web-book builds step-by-step from mathematics to algorithms:

1. Linear Algebra Foundations (Ch 2) — geometry of vectors, subspaces, and positive-semidefinite matrices.  
2. Multivariable Calculus (Ch 3) — gradients, Hessians, and first-/second-order optimality.  
3. Convex Sets (Ch 4) — feasible regions and geometric intuition.  
4. Convex Functions (Ch 5) — what makes an objective convex.  
5. Subgradients (Ch 6) — handling nondifferentiable convex functions (e.g., $\lvert x \rvert$, $\max$).  
6. KKT Conditions (Ch 7) — first-order optimality for constrained problems.  
7. Duality (Ch 8) — lower bounds, certificates of optimality, and geometric interpretation.  
8. Algorithms (Ch 9–10) — gradient, proximal, Newton, stochastic, and ADMM methods.  
9. Modeling and Practice (Ch 11) — convex modeling patterns, solver selection, ML applications.  
10. Appendices — common inequalities, projections, and support-function geometry.



## MIssing elements

1. Subgradient descent