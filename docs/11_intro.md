# Chapter 1: Introduction and Overview

Convex optimization is the subfield of mathematical optimization that deals with problems where both the objective function and the constraints are **convex**. A generic convex optimization problem can be written in standard form as: 

$$
\begin{array}{ll}
\text{minimize}_{x} & f(x) \\
\text{subject to}   & g_i(x) \le 0, \quad i=1,\dots,m, \\
& h_j(x) = 0, \quad j=1,\dots,p~,
\end{array}
$$

where $f(x)$ is a convex objective function, each $g_i(x)$ is a convex inequality constraint, and each $h_j(x)$ is an affine equality constraint (affine functions are both convex and concave). Such problems are convex in that any local minimum is guaranteed to be a global minimum (Boyd and Vandenberghe, 2004). This is the crucial property that makes convex optimization tractable: there are efficient algorithms to find the global optimum of convex problems, even in high dimensions, with no risk of getting stuck in local optima. In contrast, nonconvex optimization problems can have many local minima and are generally hard to solve in a globally optimal way (Boyd and Vandenberghe, 2004).



## Why Convexity Matters

The key concept that underpins the efficiency of convex optimization is the convexity of sets and functions. Geometrically, a set $C \subset \mathbb{R}^n$ is convex if for any two points in $C$, the straight line segment joining them lies entirely in $C$. Similarly, a function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if its epigraph (the set of points lying on or above its graph) is a convex set; equivalently, $f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y)$ for all $0\le\theta\le 1$ and all $x,y$ in the domain. Convexity implies a single "bowl-shaped" landscape with no local dips aside from the global minimum. This structure allows powerful theoretical guarantees. For example, any local minimum of a convex function (subject to a convex feasible region) must be a global minimum. Additionally, convex problems enjoy strong duality properties, which means we can often find the optimal value by solving a related *dual problem*. These properties will be explored in later sections.

Convexity also leads to robustness. Small perturbations to problem data typically result in small changes to the solution in convex optimization, a desirable feature in practice. Moreover, many approximation techniques (like convex relaxations of NP-hard problems) rely on solving a convex problem to get bounds or heuristic solutions for the original task. In summary, convex optimization provides a toolkit that form the foundation for more complex or application-specific techniques.

## Outline of this Website

This Web-book is intended for early graduate students and assumes only basic linear algebra and multivariable calculus. We will build from fundamental mathematical concepts up to the core theory of convex optimization. Each chapter introduces prerequisite material with a focus on clarity and pedagogy, preparing the reader for advanced optimization topics. The chapters are organized as follows:

- **Chapter 2: Linear Algebra Foundations.** Reviews the vector space concepts used throughout optimization. We cover vectors, matrices, norms, inner products, positive definiteness, and related topics from linear algebra that are essential in convex analysis (e.g., understanding quadratic forms and orthogonality).

- **Chapter 3: Multivariable Calculus for Optimization.** Summarizes key results from calculus in $\mathbb{R}^n$. We discuss gradients, Hessians, Taylor expansions, and the conditions for optima of unconstrained problems. These tools are necessary for understanding how to characterize and find extrema in optimization.
- **Chapter 4: Convex Sets and Geometric Fundamentals.** Introduces convex sets, their properties, and geometric insights. We define convex combinations, affine sets, polyhedra, and other fundamental geometric objects. We also discuss operations that preserve convexity of sets and the importance of convex hulls.
- **Chapter 5: Convex Functions.** Focuses on properties of convex functions. We formally define convex (and concave) functions, give many examples, and derive conditions for convexity (including second-derivative tests). We cover important inequalities like Jensen’s inequality and operations that preserve convexity of functions.
- **Chapter 6: Nonsmooth Convex Optimization – Subgradients.** Extends the concept of gradients to convex functions that are not differentiable. We introduce subgradients and subdifferentials as fundamental tools to handle nondifferentiable convex functions. This chapter explains how optimality conditions can still be expressed via subgradients and lays the groundwork for algorithms like the subgradient method.
- **Chapter 7: Optimization Principles – From Gradient Descent to KKT.** Bridges unconstrained and constrained optimization. We begin with the basic gradient descent method for unconstrained convex problems and then develop the Karush–Kuhn–Tucker (KKT) conditions for constrained problems. The KKT conditions generalize the method of Lagrange multipliers to handle inequality constraints (they are the first-order optimality conditions in nonlinear programming). We explain the meaning of each KKT condition and how they characterize optimal solutions.
- **Chapter 8: Lagrange Duality Theory.** Delves into the powerful theory of duality in convex optimization. We define the Lagrangian, derive the dual function and dual problem, and discuss weak and strong duality. This chapter shows how every convex optimization problem has a corresponding dual problem which provides lower bounds on the optimum and often leads to insightful optimality conditions (duality ties back to KKT). We also introduce Slater’s condition as a sufficient condition for strong duality in convex problems.

- **Chapter 9: Algorithms for Convex Optimization.** Surveys practical solvers: gradient methods, Newton and quasi-Newton methods, proximal algorithms, coordinate descent, and interior-point methods — emphasizing implementation and convergence in ML settings.

- **Chapter 10: Advanced Large-Scale and Structured Methods.** Focuses on methods for large datasets and structured problems, including stochastic gradient methods, ADMM, distributed optimization, and sparsity-exploiting algorithms.

- **Chapter 11: Modelling Patterns and Algorithm Selection in Practice.** Discusses how to model real-world ML problems as convex programs, choose appropriate solvers, handle non-convex extensions, and integrate optimization into end-to-end ML workflows.


- **Appendix A: Common Inequalities and Identities.** A handy reference list of fundamental mathematical inequalities and identities frequently used in proofs and exercises in convex optimization. For example, Cauchy–Schwarz, Jensen’s inequality, AM-GM inequality, and other algebraic facts.

- **Appendix B: Support Functions and Dual Geometry (Advanced).** An advanced topic for further study, introducing support functions of convex sets and related concepts of dual geometry (polar and dual cones). These tools provide deeper geometric insight into convex analysis and optimization duality but are not required for the core chapters. They are included for completeness and for readers interested in convex geometry.


**References**  

- Boyd, S. and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.  
- Nesterov, Y. (2018). *Lectures on Convex Optimization*. Springer.  


## MIssing elements

1. Subgradient descent