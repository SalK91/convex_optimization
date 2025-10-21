# Norms and Metric Geometry

Norms and metrics provide the mathematical framework to measure distances and sizes of vectors in optimization. They are central to analyzing convergence, defining constraints, and designing algorithms. This chapter develops the theory of norms, induced metrics, unit-ball geometry, and dual norms, emphasizing their role in convex optimization and machine learning.

## Norms and Induced Metrics

A norm on a vector space $V$ is a function $\|\cdot\|: V \to \mathbb{R}$ satisfying:

- Positive definiteness: $\|x\| \ge 0$ and $\|x\| = 0 \iff x = 0$  
- Homogeneity: $\|\alpha x\| = |\alpha| \|x\|$ for all $\alpha \in \mathbb{R}$  
- Triangle inequality: $\|x + y\| \le \|x\| + \|y\|$

Examples of common norms:

- $\ell_2$ norm: $\|x\|_2 = \sqrt{\sum_i x_i^2}$ (Euclidean)  
- $\ell_1$ norm: $\|x\|_1 = \sum_i |x_i|$  
- $\ell_\infty$ norm: $\|x\|_\infty = \max_i |x_i|$  
- General $\ell_p$ norm: $\|x\|_p = \left( \sum_i |x_i|^p \right)^{1/p}$

A norm induces a metric (distance function) $d(x, y) = \|x - y\|$. Metrics satisfy non-negativity, symmetry, and the triangle inequality. In optimization, metrics define step sizes, stopping criteria, and convergence guarantees.

Example: In gradient descent with step size $\alpha$, the next iterate is

$$
x_{k+1} = x_k - \alpha \nabla f(x_k),
$$

and convergence is analyzed using $\|x_{k+1} - x^*\|$ in the chosen norm.

## Unit-Ball Geometry and Intuition

The unit ball of a norm $\|\cdot\|$ is

$$
B = \{ x \in V \mid \|x\| \le 1 \}.
$$


The unit-ball geometry of a norm becomes significant whenever the norm is used in an optimization problem, either as a regularizer or as a constraint. The unit ball represents the set of all points whose norm is less than or equal to one, and its shape provides deep geometric insight into how the optimizer can move and what kinds of solutions it prefers.  

To understand its significance, consider the interaction between the level sets of the objective function and the shape of the unit ball. Level sets are contours along which the objective function has the same value. In unconstrained optimization without norms, the optimizer moves along the steepest descent of these level sets, and the solution is determined entirely by the objective’s curvature and gradients. However, when a norm is added either as a constraint (e.g., \(\|x\| \le 1\)) or as a regularization term (e.g., \(\lambda \|x\|\)) the unit ball effectively modifies the landscape: it defines directions that are more expensive or restricted, and it interacts with the level sets to shape the solution path.  


- \(\ell_2\) norm:
  The unit ball is smooth and round. All directions are treated equally. Level sets intersect the ball symmetrically, allowing gradient steps to proceed smoothly in any direction. \(\ell_2\) regularization encourages small but evenly distributed values across all coordinates, producing smooth solutions.  

- \(\ell_1\) norm: The unit ball has sharp corners along the coordinate axes. Level sets often intersect the corners, meaning some coordinates are driven exactly to zero. This produces sparse solutions, as the edges of the \(\ell_1\) ball act like “funnels” guiding the optimizer toward axes.  

- \(\ell_\infty\) norm:  The unit ball is a cube. It constrains the maximum magnitude of any coordinate, allowing free movement along directions that keep all components within bounds but blocking steps that exceed the faces. This is useful for preventing extreme values in any dimension.  



## Dual Norms

In constrained optimisation and duality theory, expressions like $\langle y, x \rangle$ naturally appear—often representing how a force (such as a gradient or dual variable) interacts with a feasible step direction. To measure how much influence such a vector $y$ can exert when movement is restricted by a norm constraint, we define the dual norm:

\[
\|y\|_* = \sup_{\|x\| \le 1} \langle y, x \rangle.
\]

This definition asks:  
> If movement is only allowed within the unit ball of the original norm, what is the maximum directional effect that $y$ can generate?

### Intuition — movement vs. influence

In constrained optimization, the primal norm defines where you are allowed to move, and the dual norm measures how effectively the gradient can move you within that region.

- The primal norm determines the shape of the feasible directions, forming a mobility region (for example, an $\ell_2$ ball is round and smooth, an $\ell_1$ ball is sharp and cornered).  
- The dual norm tells how much progress a gradient can make when pushing against that region.  
- If the gradient aligns with a flat face or a corner of the feasible region, movement becomes limited (as in $\ell_1$ geometry, leading to sparse solutions).  
- If the feasible region is smooth (as in $\ell_2$ geometry), the gradient can always push effectively, producing smooth updates.

### Example — maximum decrease under a norm constraint

Consider the constrained problem:

\[
\min_x f(x) \quad \text{subject to} \quad \|x\| \le 1.
\]

For a small step $s$, the change in $f$ is approximately

\[
f(x+s) \approx f(x) + \langle \nabla f(x), s \rangle.
\]

To decrease $f$, we want to **minimize** $\langle \nabla f(x), s \rangle$ over feasible steps $\|s\| \le 1$, or equivalently:

\[
\max_{\|s\| \le 1} \langle -\nabla f(x), s \rangle.
\]

By definition of the dual norm, this maximum is exactly

\[
\max_{\|s\| \le 1} \langle -\nabla f(x), s \rangle = \|\nabla f(x)\|_*.
\]

Intuition:

- The primal norm defines the feasible region of allowed steps.  
- The dual norm** measures the largest possible influence the gradient can exert within that region.  
- If the unit ball is round (smooth), the gradient can push efficiently in any direction; if it has corners (as in $\ell_1$), the gradient’s effective action is concentrated along certain axes.  

Thus, the dual norm is the true measure of how powerful a gradient or dual variable is under a constraint, not just its raw magnitude. It appears naturally in optimality conditions, Lagrangian duality, and subgradient methods, providing a precise bound on the effect of forces inside the feasible set.


## Metric Properties in Optimization

Metrics derived from norms allow analysis of convergence rates. For an $L$-smooth function $f$, the update $x_{k+1} = x_k - \alpha \nabla f(x_k)$ satisfies

$$
\|x_{k+1} - x^*\| \le \|x_k - x^*\| - \alpha (1 - \frac{L \alpha}{2}) \|\nabla f(x_k)\|^2.
$$

This shows the choice of norm directly affects step size rules, stopping criteria, and algorithmic stability.

Unit-ball shapes also influence **proximal operators**. For a regularizer $R(x) = \lambda \|x\|_1$, the proximal step shrinks components along the axes, exploiting the corners of the $\ell_1$ unit ball to enforce sparsity.



## Norms on Function Spaces: $L_p$ Norms

Function spaces generalise vector norms. For functions defined on an interval $[a,b]$, the $L_p$ norm is
$$
\|f - g\|_{L^p} = \left( \int_a^b |f(x) - g(x)|^p \, dx \right)^{1/p},
$$
for $1 \le p < \infty$, and
$$
\|f - g\|_{L^\infty} = \operatorname*{ess\,sup}_{x \in [a,b]} |f(x) - g(x)|.
$$

Interpretations:

- $L_1$ measures total absolute discrepancy, the shaded area between graphs:
  $$
  \|f - g\|_{L^1} = \int_a^b |f(x) - g(x)| \, dx.
  $$
  Small, widespread differences accumulate; a narrow spike of small width contributes little to $L_1$.
- $L_2$ is the RMS or energy of the error:
  $$
  \|f - g\|_{L^2} = \left( \int_a^b |f(x) - g(x)|^2 \, dx \right)^{1/2}.
  $$
  Squaring amplifies large errors; a spike of height $A$ and width $\varepsilon$ contributes about $A^2 \varepsilon$ to the squared norm.
- $L_\infty$ is the worst-case deviation:
  $$
  \|f - g\|_{L^\infty} = \sup_{x \in [a,b]} |f(x) - g(x)|.
  $$
  A single pointwise deviation, no matter how narrow, can dominate the norm. Use $L_\infty$ in robust optimisation and adversarial settings.

