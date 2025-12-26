# Chapter 1:  Introduction and Overview

Optimization is at the heart of most machine-learning methods. Whether training a linear model or a deep neural network, learning usually means adjusting parameters to minimize a loss that measures how well the model fits the data. Convex optimization is a particularly important and well-understood part of optimization. When both the objective and the constraints are convex, the problem has helpful properties:

1. No bad local minima: any local minimum is also the global minimum.  
2. Predictable behavior: algorithms like gradient descent have clear and well-studied convergence.  
3. Solutions are easy to verify: convex problems come with simple mathematical conditions that tell us when we have reached the optimum.


## Motivation: Optimization in Machine Learning

Many supervised learning problems can be written in a common form:

$$
\min_{x \in \mathcal{X}} 
\; \frac{1}{N}\sum_{i=1}^{N} \ell(a_i^\top x, b_i) 
+ \lambda R(x),
$$

where

- $\ell(\cdot,\cdot)$ is a loss function that measures how well the model predicts $b_i$ from $a_i$,  
- $R(x)$ is a regularizer that encourages certain structure (such as sparsity or small weights),  
- $\mathcal{X}$ is a set of allowed parameter values, often simple and convex.

Many widely used losses and regularizers are convex. Examples include least squares, logistic loss, hinge loss, Huber loss, the $\ell_1$ norm, and the $\ell_2$ norm. Convexity is what makes these problems tractable and allows them to be solved efficiently at scale using well-behaved optimization algorithms.

 
## Why Convex Optimization Remains Central in ML

Although many modern models are nonconvex, convex optimization continues to play a major role:

1. Convex surrogate losses: Losses such as logistic, hinge, and Huber are convex substitutes for harder objectives like the $0\text{–}1$ loss. They make optimization practical while still leading to models that generalize well.

2. Convex subproblems inside larger algorithms:  Many nonconvex methods solve convex problems as part of their inner loop. Examples include least-squares steps in matrix factorization, proximal updates in regularized learning, and simple convex problems that appear in line-search procedures.

These roles make convex optimization a key component of modern ML toolkits, even when the main model is nonconvex.
 
## Web-Book Roadmap and How to Use It

* Chapter 2: Linear Algebra Foundations. Basic vector/matrix operations and linear algebra needed for optimization.

* Chapter 3: Multivariable Calculus. Differentiation and derivatives of functions of many variables (gradients, Hessians).

* Chapter 4: Convex Sets and Geometry. Definitions and examples of convex sets, cones, affine spaces, and geometric properties.

* Chapter 5: Convex Functions. Convexity for functions, epigraphs, and key examples (norms, quadratic functions, log-sum-exp, etc.).

* Chapter 6: Nonsmooth Optimization – Subgradients. Generalized derivatives for convex functions that are not differentiable, and subgradient methods.

* Chapter 7: First-Order Optimality Conditions. Gradient-based optimality for smooth problems, supporting theory for necessary and sufficient conditions.

* Chapter 8: Optimization Principles – From Gradient Descent to KKT. Unconstrained and constrained gradient methods, culminating in the Karush–Kuhn–Tucker (KKT) conditions.

* Chapter 9: Lagrange Duality Theory. Duality principles, weak/strong duality, and interpretations of Lagrange multipliers.

* Chapter 10: Pareto Optimality and Multi-Objective Optimization. Trade-offs in optimizing multiple goals and Pareto efficiency.

* Chapter 11: Regularized Approximation. Balancing fit vs. complexity with regularization (ℓ₁, ℓ₂, elastic net, etc.).

* Chapter 12: Algorithms for Convex Optimization. General convex optimization solvers and algorithmic frameworks (interior-point, gradient methods, etc.).

* Chapter 13: Equality-Constrained Problems. Specialized methods (e.g. KKT with only equalities, reduced-space methods).

* Chapter 14: Inequality-Constrained Problems. Algorithms handling general inequality constraints, barrier methods.

* Chapter 15: Advanced Large-Scale and Structured Methods. Techniques for very large or structured problems (decomposition, coordinate descent, etc.).

* Chapter 16: Modeling Patterns and Algorithm Selection. Practical guidance on modeling choices and selecting the right algorithm in practice.

* Chapter 17: Canonical Problems in Convex Optimization. Well-known problem templates (linear, quadratic, SOCP, SDP) and how to recognize them.

* Chapter 18: Modern Optimizers in Machine Learning Frameworks. How convex optimization appears in ML libraries and frameworks.

* Chapter 19: Beyond Convexity – Nonconvex and Global Optimization. Overview of nonconvex problems and global methods (to contrast with convex theory).

* Chapter 20: Derivative-Free and Black-Box Optimization. Techniques when gradients are not available.

* Chapter 21: Metaheuristic and Evolutionary Optimization. Heuristic algorithms (genetic algorithms, simulated annealing) for hard problems.

* Chapter 22: Advanced Topics in Combinatorial Optimization. Combinatorial optimization problems and convex relaxations.

This roadmap helps the reader see how the material progresses from foundations (Ch.2–5) to theory (Ch.6–11) to algorithms (Ch.12–15) and on to specialized and modern topics (Ch.16–22).


## Acknowledgments
The content and structure of this web book are strongly informed by the Stanford University course EE364A: Convex Optimization I, taught by Stephen Boyd. 