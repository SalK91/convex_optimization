# Tractable Problems

A problem is tractable if it can be **solved efficiently with guarantees** (typically polynomial time).  
Convexity is the key to tractability: **convex problems are tractable** under mild assumptions. Non-convex problems are generally intractable or require heuristics.


## Relationship Between Convexity and Tractability

- Convex objective + convex feasible set → **global optima reachable efficiently**.  
- Non-convex objectives or feasible sets → may have multiple local minima → **not tractable in general**.  
- Tractability also depends on:
  - **Dimension of the problem**  
  - **Smoothness / Lipschitz properties**  
  - **Availability of projections / proximal operators**  


## Tools to Assess Tractability

- **Gradient-based methods:** require convexity for guaranteed convergence.  
- **Proximal algorithms:** require convex objectives or convex regularizers.  
- **Duality:** strong duality holds for convex problems → efficient solution via dual.  
- **Numerical stability:** condition number of Hessian/Gram matrix affects convergence.  


## Examples of Tractable Problems

1. Least Squares Regression: convex → tractable via gradient descent or closed-form solution.  
2. LASSO / Sparse Regression: convex → tractable via proximal methods.  
3. Quadratic Programs (PSD Hessian): convex → tractable.  
4. Non-convex bilinear problem: not tractable → requires heuristics or relaxation.


## Practical Checklist for Tractability

1. Confirm the problem is convex.  
2. Check smoothness, Lipschitz continuity, and gradient availability.  
3. Verify constraints are simple enough for **projections or proximal steps**.  
4. Consider dual formulations for efficient computation.  
5. Ensure numerical conditioning is acceptable for iterative solvers.

