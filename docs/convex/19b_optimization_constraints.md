# Chapter 14: Optimization Algorithms for Inequality-Constrained Problems

In practice, optimization problems often include inequalities that restrict feasible solutions to a convex region. Examples include nonnegativity of variables, margin constraints in support vector machines, fairness or safety limits, and physical conservation laws. This chapter introduces algorithms for solving such problems efficiently, focusing on the logarithmic barrier and interior-point methods that underpin modern convex solvers.
 
## 14.1 Problem Setup

We consider the general convex optimization problem with both equality and inequality constraints:

$$
\begin{aligned}
\text{minimize}   &\quad f_0(x) \\
\text{subject to} &\quad f_i(x) \le 0, \quad i=1,\dots,m,\\
&\quad A x = b.
\end{aligned}
$$

Assumptions:

- Each $f_i$ is convex and twice differentiable.  
- $A \in \mathbb{R}^{p\times n}$ has full row rank ($\mathrm{rank}(A)=p$).  
- There exists a strictly feasible point $\bar{x}$ such that $f_i(\bar{x})<0$ and $A\bar{x}=b$ (Slater’s condition).  

Under these assumptions, strong duality holds and the KKT conditions are necessary and sufficient for optimality.

### Examples

| Problem | $f_0(x)$ | $f_i(x)$ | Notes / ML context |
|----------|-----------|-----------|--------------------|
| Linear Program (LP) | $c^T x$ | $a_i^T x - b_i$ | Feature selection, resource allocation |
| Quadratic Program (QP) | $\tfrac{1}{2}x^T P x + q^T x$ | Linear $a_i^T x - b_i$ | SVM training, ridge regression |
| QCQP | Quadratic | Quadratic | Portfolio optimization, control |
| Geometric Program (log domain) | Convex in $\log x$ | Linear in $\log x$ | Network flow, resource allocation |
| Entropy minimization | $\sum_i x_i \log x_i$ | $F x \le g$ | Probability calibration, information bottleneck |

 
## 14.2 Indicator-Function Reformulation

Define the indicator of the nonpositive orthant:

$$
I_-(u)=
\begin{cases}
0, & u \le 0,\\
+\infty, & u > 0.
\end{cases}
$$

Then the constrained problem is equivalent to

$$
\min_x \; f_0(x) + \sum_{i=1}^m I_-(f_i(x))
\quad \text{s.t. } A x = b.
$$

This form is conceptually clear but nondifferentiable since $I_-$ is discontinuous. To apply Newton-type algorithms, we replace $I_-$ with a smooth approximation: the logarithmic barrier.

## 14.3 Logarithmic-Barrier Approximation

We approximate each $I_-(f_i(x))$ by a differentiable barrier function $\Phi(u) = -\tfrac{1}{t} \log(-u)$ for $u < 0$. The smoothed subproblem becomes

$$
\min_x \; f_0(x) - \frac{1}{t} \sum_{i=1}^m \log(-f_i(x))
\quad \text{s.t. } A x = b.
$$

- For small $t$: the barrier is strong and keeps points deep inside the feasible region.  
- As $t \to \infty$: the barrier weakens and the solution approaches the true optimum.

Hence, the original inequality-constrained problem is replaced by a sequence of smooth equality-constrained subproblems.

## 14.4 Properties of the Barrier Function

Define

$$
\phi(x) = -\sum_{i=1}^m \log(-f_i(x)), \qquad
\mathrm{dom}\,\phi = \{x : f_i(x) < 0\}.
$$

Then $\phi$ is convex and twice differentiable:

$$
\nabla \phi(x) = \sum_i \frac{1}{-f_i(x)} \nabla f_i(x),
$$

$$
\nabla^2 \phi(x) =
\sum_i \frac{1}{f_i(x)^2} \nabla f_i(x)\nabla f_i(x)^T
+ \sum_i \frac{1}{-f_i(x)} \nabla^2 f_i(x).
$$

Near the boundary $f_i(x)=0$, the gradient norm grows without bound — producing a repulsive force that prevents violation of constraints.


## 14.5 Central Path and Approximate KKT Conditions

For each $t > 0$, let $x^*(t)$ minimize the barrier problem

$$
\min_x\; t f_0(x) + \phi(x)
\quad \text{s.t. } A x = b.
$$

The curve $\{x^*(t) : t > 0\}$ is the central path. As $t \to \infty$, $x^*(t)$ approaches the true optimal solution $x^*$. Along this path there exist dual variables $(\lambda^*(t), v^*(t))$ satisfying

$$
\begin{aligned}
\nabla f_0(x^*(t)) + \sum_i \lambda_i^*(t) \nabla f_i(x^*(t)) + A^T v^*(t) &= 0,\\
A x^*(t) &= b,\\
-\lambda_i^*(t) f_i(x^*(t)) &= \tfrac{1}{t}, \quad \lambda_i^*(t) \ge 0.
\end{aligned}
$$

The complementarity condition is relaxed: $\lambda_i f_i(x) = -1/t$ instead of $0$. As $t \to \infty$, these approximate KKT conditions converge to the exact ones.

## 14.6 Geometric and Physical Intuition
The centering subproblem

$$
\min_x\; t f_0(x) - \sum_i \log(-f_i(x))
$$

can be viewed as a particle system in a potential field:

- The objective $f_0(x)$ pulls toward lower cost (external force).  
- Each constraint $f_i(x)\le0$ creates a repulsive potential that diverges near the boundary.  

At equilibrium, these forces balance:

$$
\nabla f_0(x^*(t)) + \sum_i \frac{1}{t(-f_i(x^*(t)))} \nabla f_i(x^*(t)) = 0.
$$

Thus, the solution remains strictly feasible — this is the essence of the interior-point philosophy.



## 14.7 Barrier-Method Algorithm

The barrier method converts the original inequality-constrained problem into a sequence of smooth equality-constrained subproblems.  
Each subproblem is solved exactly (to high precision) while a *barrier parameter* $t$ is gradually increased, allowing the iterates to approach the boundary and the true constrained optimum.
### Algorithm Outline

Given:

- a strictly feasible starting point $x$ (so $f_i(x) < 0$ for all $i$),
- an initial barrier parameter $t > 0$,
- a barrier scaling factor $\mu > 1$ (usually between 10 and 20),
- and a desired accuracy $\varepsilon > 0$ (for stopping),

the algorithm proceeds as follows:

1. Centering step: Solve  
   $$
   \min_x \; f_0(x) - \frac{1}{t} \sum_{i=1}^m \log(-f_i(x))
   \quad \text{s.t. } A x = b
   $$
   using Newton’s method for equality-constrained optimization. The result $x^*(t)$ is the *centering point* for the current $t$.

2. Update iterate:  Set $x := x^*(t)$.

3. Stopping criterion:  Stop if 
   $$
   \frac{m}{t} < \varepsilon.
   $$
   Here $m$ is the number of inequality constraints, and $\varepsilon$ is the desired tolerance on suboptimality. This rule is derived from the duality gap bound:
   $$
   f_0(x^*(t)) - p^* \le \frac{m}{t},
   $$
   meaning that if $m/t$ is smaller than $\varepsilon$, the current solution is guaranteed to be within $\varepsilon$ of the true optimum.

4. Increase barrier parameter: Set $t := \mu t$ and return to Step 1.
   Each centering subproblem maintains strict feasibility, and increasing $t$ gradually weakens the barrier, allowing the iterates to approach the true constraint boundary. A typical choice is $\mu \in [10, 20]$.


### Understanding $\varepsilon$ — the Accuracy Parameter
The parameter $\varepsilon$ controls how close to the optimal solution we wish to stop.

- Mathematically, $\varepsilon$ specifies an upper bound on the duality gap:
  $$
  f_0(x) - p^* \le \varepsilon.
  $$

- Conceptually, $\varepsilon$ represents the trade-off between accuracy and computational cost:
  - Smaller $\varepsilon$ → more iterations (larger $t$ required).
  - Larger $\varepsilon$ → faster termination, but lower accuracy.

In practice:

- For numerical optimization or ML training, $\varepsilon$ is often set between $10^{-3}$ and $10^{-8}$ depending on problem size and desired precision.  
- Convex solvers (like CVX, MOSEK, or ECOS) typically use $\varepsilon \approx 10^{-6}$ as a default high-accuracy target.


### Intuitive Interpretation

- Think of $\varepsilon$ as the “distance” between the current point and the true optimum in terms of objective value.  
- The ratio $m/t$ acts like a thermometer for this distance — as $t$ grows, the temperature (error) cools down.
- Once $m/t < \varepsilon$, we know the algorithm has cooled sufficiently: the point lies extremely close to the optimal constrained solution.

### Summary of Key Parameters

| Symbol | Meaning | Typical Value / Range | Intuitive Role |
|---------|----------|------------------------|----------------|
| $m$ | Number of inequality constraints | problem dependent | Total number of barrier terms |
| $t$ | Barrier parameter | starts small (1–10), grows by $\mu$ | Controls strength of barrier |
| $\mu$ | Barrier growth factor | 10–20 | Controls how fast we approach constraint boundary |
| $\varepsilon$ | Desired accuracy (tolerance) | $10^{-3}$ to $10^{-8}$ | Stopping threshold based on duality gap |

### Intuitive Summary

- Each centering step finds the best *interior* point for a given barrier strength $1/t$.  
- Increasing $t$ reduces the barrier effect, letting $x$ approach the boundary.  
- The stopping rule $m/t < \varepsilon$ ensures that the objective value of $x$ differs from the true optimum by less than $\varepsilon$.  
- Smaller $\varepsilon$ means tighter optimality, but more work (larger $t$ and more iterations).


 
## 14.8 Computational and Practical Notes

- Each centering problem is solved by equality-constrained Newton steps (KKT system).  
- Barrier methods inherit superlinear convergence near the optimum.  
- Initialization must be strictly feasible; feasibility restoration can be costly.  
- Large $t$ makes the barrier steep, so line search and step damping are essential.

In machine learning:
- SVM and logistic regression margin constraints fit naturally in this form.  
- Interior-point solvers for QPs are used in sparse regression and convex relaxations.  
- Barrier penalties act as smooth approximations to hard constraints in physics-informed and fairness-aware models.

 
## 14.9 Comparison: Equality vs Inequality-Constrained Methods

| Aspect | Equality Constraints | Inequality Constraints |
|--------|---------------------|------------------------|
| Feasible set | Affine manifold | Convex region with boundary |
| Algorithms | Newton, projected Newton, KKT | Barrier, interior-point, primal–dual |
| Feasibility handling | Exact | Maintained via barrier term |
| Complementarity | $A x = b$ | $\lambda_i f_i(x) = 0$ (or $= -1/t$) |
| Feasible start | Optional | Required (strict) |
| ML relevance | Normalization, fairness, balance | Nonnegativity, margins, sparsity, safety constraints |

