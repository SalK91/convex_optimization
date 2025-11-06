# Chapter 1: Introduction and Overview

Optimization is the mathematical language of decision-making and learning. In machine learning and data science we constantly fit, estimate, or infer by minimizing a loss. Yet most real-world objectives are nonlinear and messy; solving them globally can be hard.

Convex optimization is the *tractable core* of this landscape. A convex problem has a single “bowl-shaped” valley, no false minima, so any local optimum is automatically global. That geometric simplicity gives us:

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
 

 Chapter 01: Mathematical Concepts
Chapter 1.1: Differentiability
Chapter 1.2: Taylor Approximation
Chapter 1.3: Convexity
Chapter 1.4: Conditions for optimality
Chapter 1.5: Quadratic forms I
Chapter 1.6: Quadratic forms II
Chapter 1.7: Matrix calculus
Chapter 02: Optimization problems
Chapter 2.1: Unconstrained optimization problems
Chapter 2.2: Constrained optimization problems
Chapter 2.3: Other optimization problems
Chapter 03: Univariate Optimization
Chapter 3.1: Golden ratio
Chapter 3.2: Brent
Chapter 04: First order methods
Chapter 4.01: Gradient descent
Chapter 4.02: Step size and optimality
Chapter 4.03: Deep dive: Gradient descent
Chapter 4.04: Weaknesses of GD – Curvature
Chapter 4.05: GD – Multimodality and Saddle points
Chapter 4.06: GD with Momentum
Chapter 4.07: GD in quadratic forms
Chapter 4.09: SGD
Chapter 4.10: SGD Further Details
Chapter 4.11: ADAM and friends
Chapter 05: Second order methods
Chapter 5.01: Newton-Raphson
Chapter 5.03: Gauss-Newton
Chapter 06: Constrained Optimization
Chapter 6.01: Introduction
Chapter 6.02: Linear Programming
Chapter 6.04: Duality in optimization
Chapter 6.05: Nonlinear programs and Lagrangian
Chapter 07: Derivative Free Optimization
Chapter 7.01: Coordinate Descent
Chapter 7.02: Nelder-Mead
Chapter 7.03: Simulated Annealing
Chapter 7.04: Multi-Starts
Chapter 08: Evolutionary Algorithms
Chapter 8.01: Introduction
Chapter 8.02: ES / Numerical Encodings
Chapter 8.03: GA / Bit Strings
Chapter 8.04: CMA-ES Algorithm
Chapter 8.05: CMA-ES Algorithm Wrap Up
Chapter 10: Bayesian Optimization
Chapter 10.01: Black Box Optimization
Chapter 10.02: Basic BO Loop and Surrogate Modelling
Chapter 10.03: Posterior Uncertainty and Acquisition Functions I
Chapter 10.04: Posterior Uncertainty and Acquisition Functions II
Chapter 10.05: Important Surrogate Models
