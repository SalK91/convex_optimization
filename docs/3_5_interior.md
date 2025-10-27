When optimizing with inequality constraints (e.g. linear or convex constraints $g_i(x) \le 0$), a different class of algorithms — interior point methods — has proven highly effective. Unlike gradient or simplex methods that move along the boundary of the feasible region, interior point methods approach the solution from the interior of the feasible set, guided by barrier functions. They exploit smoothness in the interior by incorporating constraints into the objective, enabling the use of Newton-like steps even for constrained problems.

Barrier Functions and the Central Path

Consider a convex optimization problem in standard form:

$$
\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \le 0, \quad i = 1, \dots, m.
\end{aligned}
$$


with $f$ and $g_i$ convex. An interior point approach introduces a barrier function $b(x)$ that blows up near the boundary $g_i(x)=0$. A classic choice is the logarithmic barrier: if $g_i(x)\le 0$ are the constraints, define

$$
b(x) = -\sum_{i=1}^m \log(-g_i(x))
$$


which is finite when $g_i(x)<0$ (interior) and tends to $+\infty$ as any $g_i(x)\to 0$ (approaching the boundary). We then solve a series of unconstrained problems:

$$
\min_x \; F_t(x) := t f(x) + b(x)
$$


where $t>0$ is a parameter that controls the trade-off between the original objective and the barrier. For each fixed $t$, the minimizer $x^(t)$ of $t f(x)+b(x)$ is an interior point that balances minimizing $f$ against staying away from the constraint boundaries. As $t$ increases, we place more weight on $f$ and less on the barrier. In the limit $t \to \infty$ (or equivalently using $\mu = 1/t \to 0^+$ as barrier parameter), $x(t)$ approaches the true constrained optimum $x^*$. The curve of solutions ${x(t): t>0}$ is called the central path. Points on this path are characterized by a certain perturbed optimality condition (the Karush-Kuhn-Tucker conditions with $g_i(x) < 0$ and Lagrange multipliers $\propto 1/t$).

 

For example, in a simple linear program, the central path is the set of strictly feasible solutions that minimize $c^T x + \frac{1}{t}\sum_i \log x_i$ for increasing $t$. Initially (small $t$) the solution is heavily influenced by the barrier and stays well within the feasible region. As $t$ grows, the solution moves closer to the boundary and to the true optimum, which often lies on the boundary (one or more $g_i(x)=0$ at optimum). The central path converges to the optimal point as the barrier vanishes.

 

Interior point methods follow this central path. They start with a moderate $t$ and a feasible interior $x^(t)$. Then they gradually increase $t$ (decrease barrier) and use the previous solution as a warm start to compute $x^(t+\Delta t)$. Effectively, they trace the trajectory $x^*(t)$ towards the optimum, rather than jumping directly to the boundary.

 

Why use the barrier at all? Because while the constraints are active, directly attacking them can be complicated (especially if many constraints; think of simplex methods hopping between tight constraints). The barrier method keeps the iterate strictly feasible ($g_i(x)<0$ for all $i$) at all times, avoiding the combinatorial complexity of dealing with active sets. Instead, constraint satisfaction is handled smoothly by the barrier term, and we can deploy powerful smooth optimization (like Newton’s method) on $F_t(x) = t f(x)+b(x)$. Each $F_t$ is unconstrained and differentiable on the interior, amenable to Newton’s fast convergence.

Newton Steps and Feasibility vs. Optimality Trade-off

To solve $\min_x F_t(x)$ for a given $t$, interior point methods typically use Newton’s method because $F_t(x)$ is twice-differentiable (assuming $f$ and $g_i$ are). In fact, one derives the gradient and Hessian of the log-barrier $b(x)$ and then computes Newton steps for the equation $\nabla F_t(x)=0$ (the perturbed optimality condition). These Newton steps, however, must remain in the interior (we can’t step outside or we’d evaluate $\log(-g_i(x))$ at an invalid point). Thus damped Newton steps are employed: we choose a step length $\lambda$ to ensure $x+\lambda d$ stays feasible ($g_i(x+\lambda d) < 0$) for all $i$. This usually means a backtracking line search that stops when $x+\lambda d$ is safely inside the domain (for instance, requiring $g_i(x+\lambda d) \le (1-\tau)g_i(x)$ for some $\tau\in(0,1)$ to maintain a margin). By choosing conservative step sizes, one guarantees the iterates never hit the boundary
stat.cmu.edu
.

 

Each $F_t$ is solved to a certain accuracy (often just a few Newton steps are needed if started near the optimum for that $t$). Then $t$ is increased (the barrier is made smaller). There is a trade-off in choosing how fast to increase $t$: a short-step method increases $t$ gradually, staying very close to the central path, which ensures each Newton solve is very efficient (few iterations) but requires many little increases in $t$. A long-step or path-following method takes more aggressive increases in $t$, deviating more from the central path and possibly requiring more Newton iterations to recenter. In practice, modern solvers use predictor-corrector variants (like Mehrotra’s algorithm) which intelligently choose $t$ and include an extra correction Newton step to stay on track. These methods are very efficient.

 

Feasibility vs. optimality: The barrier parameter $1/t$ controls this balance. When $1/t$ is large, the barrier term dominates and the algorithm prioritizes feasibility (staying well inside the region, far from the boundaries). The solution $x^*(t)$ at small $t$ is in the “center” of the feasible region (hence central path). When $1/t$ is tiny (large $t$), the barrier is weak and the algorithm prioritizes optimality of $f(x)$, tolerating being near constraint boundaries. The iterates gradually shift from emphasizing feasibility to emphasizing optimality. The beauty of interior point methods is that they maintain a gentle balance: you never violate feasibility, and you make smooth progress toward optimality.

 

Because interior point methods effectively solve a series of Newton systems, their iteration count is largely independent of problem size and more a function of condition measures (like self-concordance parameters of the barrier). They run in polynomial time theoretically and often extremely fast in practice. For instance, Karmarkar’s interior point algorithm was a breakthrough that showed linear programming can be solved in worst-case polynomial time, and in practice interior point solvers can solve huge LPs faster than the simplex method for many cases. For general convex problems, interior point methods are the method of choice when high accuracy is needed and the problem isn’t so large as to preclude forming Hessians (they’re used in many commercial solvers for e.g. quadratic programming, semidefinite programming, etc.).

 

Example (Linear Program): $\min{c^T x : Ax=b,, x>0}$. The log-barrier method solves $\min_x t,c^T x - \sum_i \log x_i$ subject to $Ax=b$. The central path equation yields $t c_i - \frac{1}{x_i} = \text{(something from dual vars)}$ for each $i$. When $t$ is small, $1/x_i$ terms dominate, so $x$ stays away from 0; as $t\to \infty$, the solution forces $c_i \approx 0$ for basic variables at optimum and some $x_i \to 0$ for non-basics. The interior path thus goes from a “centered” positive solution and ends at the optimal corner of the polytope. Figure 15.2 in some lecture notes (see reference) shows contour lines of the barrier for increasing $t$: they gradually approximate the sharp corners of the feasible polyhedron, and the central path (dotted line) goes through the middle of the feasible region and eventually to the corner solution. This illustrates why it’s called an interior-point method: it **starts at an interior point and moves along a path within the interior to the solution】.

Summary

Interior point methods convert constrained problems into a sequence of unconstrained ones via barrier functions. By following the central path and using damped Newton steps, they achieve a powerful combination of feasibility maintenance and fast convergence. The geometry of interior methods is captured by the central path staying in the strict interior and asymptotically approaching the boundary optimum. Modern interior point algorithms are primal-dual (they track dual variable estimates as well), which often improves practical performance and provides stopping criteria. They are widely used in solving linear programs, quadratic programs, and general conic programs at large scale, often outperforming simplex or cutting-plane methods for large problems. The development of interior point techniques was a major milestone in convex optimization, marrying the smooth Newton technique with inequality constraints in a mathematically elegant and efficient way. As a result, we can now solve huge convex problems to very high accuracy by exploiting this approach, traversing the interior of the feasible region with confidence and speed.