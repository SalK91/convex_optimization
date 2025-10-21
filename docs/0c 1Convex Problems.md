# Convex Problems
Convex problems form the backbone of optimization theory because they have well-defined geometric and algebraic properties.  
A problem is convex if **the objective function is convex** and **the feasible set is convex**, i.e., any weighted average of feasible points remains feasible.  
Convexity is essential to guarantee **global optimality** of solutions.


## Convexity of Objective Functions

- **Linear / Affine:** $f(x) = c^\top x + d$ → convex.  
- **Quadratic:** $f(x) = \frac{1}{2} x^\top Q x + c^\top x$, convex if $Q \succeq 0$.  
- **Sum of Convex Functions:** convex.  
- **Maximum / Supremum of Convex Functions:** convex.  
- **Composition Rules:** $f(g(x))$ convex if $f$ convex and non-decreasing, $g$ convex.  

**Tools for verification:**

- Hessian: $\nabla^2 f(x) \succeq 0$ for twice-differentiable $f$.  
- Epigraph: $\text{epi}(f) = \{(x,t): f(x) \le t\}$ convex → $f$ convex.  
- Subgradient checks for nonsmooth functions.


## Convexity of Constraints

- **Inequalities:** $f_i(x) \le 0$, convex $f_i$ → convex feasible region.  
- **Equalities:** $h_j(x) = 0$, must be affine.  
- **Conic/Set Membership:** $x \in C$ with convex $C$.  


## Common Non-Convex Structures

- Bilinear or indefinite quadratic terms.  
- Nonlinear equality constraints.  
- Products, ratios, fractional terms.  
- Discrete/integer variables.  


## Examples

- Least Squares: convex.  
- LASSO: convex.  
- Quadratic Program with PSD Hessian: convex.  
- Bilinear $x_1 x_2$ problem: non-convex.


## Checklist for Convexity

1. Is the objective convex?  
2. Are inequalities convex?  
3. Are equalities affine?  
4. Avoid non-convex structures.  
5. Use Hessian, epigraph, or subgradient verification.  
6. Optional: dual/Fenchel check.
