We now bring together geometry and calculus to characterize when a solution is optimal and to understand duality – a powerful perspective that every convex optimization problem comes paired with another (dual) problem providing bounds on the optimum.

 

**Unconstrained optimality:** As noted, if $f: \mathbb{R}^n \to \mathbb{R}$ is convex and differentiable, the necessary and sufficient condition for $x^*$ to be a (global) minimizer is $\nabla f(x^*) = 0$. Geometrically, this means the tangent hyperplane is flat; algebraically, there’s no descent direction. If $f$ is not differentiable, the condition generalizes to $0 \in \partial f(x^*)$ – the origin must be in the subgradient set, meaning there are subgradients pointing every way, so no single direction of descent. These conditions are first-order optimality conditions.

 

**Constrained optimality (KKT conditions):** Consider a problem $\min f(x)$ s.t. $h_i(x)=0$ (for $i=1,\dots,m$ equality constraints) and $g_j(x) \le 0$ (for $j=1,\dots,p$ inequality constraints), with $f$ convex, $h_i$ affine, $g_j$ convex. The Karush–Kuhn–Tucker (KKT) conditions give a set of equations and inequalities that characterize optimal $x^*$ and associated dual variables $\lambda_i$ (for equalities) and $\mu_j \ge 0$ (for inequalities). They are:

1. **Primal feasibility:** $h_i(x^*)=0$ for all $i$, and $g_j(x^*) \le 0$ for all $j$. (i.e. $x^*$ satisfies the original constraints)

2. **Dual feasibility:** $\mu_j \ge 0$ for all $j$. (Lagrange multipliers for inequalities are nonnegative)

3. **Stationarity:** $\nabla f(x^*) + \sum_i \lambda_i \nabla h_i(x^*) + \sum_j \mu_j \nabla g_j(x^*) = 0$. This is essentially $\nabla_x \mathcal{L}(x^*,\lambda,\mu) = 0$, where $\mathcal{L}(x,\lambda,\mu) = f(x)+\sum_i \lambda_i h_i(x) + \sum_j \mu_j g_j(x)$ is the Lagrangian. It generalizes “gradient = 0” to account for constraint forces.

4. **Complementary slackness:** $\mu_j, g_j(x^*) = 0$ for all $j$. This means for each inequality, either the constraint is tight ($g_j(x^*)=0$) in which case $\mu_j$ can be positive, or the constraint is loose ($g_j(x^*)<0$) in which case optimal $\mu_j$ must be 0. Essentially $\mu_j$ “activates” only on active constraints.

If strong duality holds (which it does under convexity and Slater’s condition – existence of a strictly feasible point), these conditions are necessary and sufficient for optimality. They are the heart of convex optimization theory. Geometrically, the stationarity condition says $\nabla f(x^*)$ is a linear combination of the constraint normals ($\nabla h_i, \nabla g_j$) with certain weights. This is exactly the intuitive picture: at optimum $x^*$, if constraints that are active form surfaces, the gradient of $f$ must lie in their span – otherwise, there’d be a direction along the surface to decrease $f$. The multipliers $\lambda,\mu$ are those weights.

 

Consider a simple case: $\min f(x)$ s.t. $a^T x = b$. At optimum $x^*$, we expect $\nabla f(x^)$ to be orthogonal to the feasible plane (or else we could move along the plane and decrease $f$). The constraint normal is $a$. So we expect $\nabla f(x^*) = \lambda a$ for some $\lambda$. That is the stationarity KKT condition in this case, and $\lambda$ is the Lagrange multiplier. It measures how much increasing $b$ would increase the optimal value (sensitivity).

 

For inequality, say one constraint $g_1(x)\le0$ active. Then at optimum, $\nabla f(x^) = \mu_1 \nabla g_1(x^)$. If the constraint is not active, $\mu_1 = 0$ (since gradient of $f$ has no reason to align with that constraint as it’s not “binding”).

 

**Dual problem and interpretation:** The Lagrangian $\mathcal{L}(x,\lambda,\mu)$ introduces these multipliers. The Lagrange dual function is $q(\lambda,\mu) = \inf_x \mathcal{L}(x,\lambda,\mu)$. For convex problems, one can often swap inf and sup and get $\max_{\lambda,\mu \ge 0} q(\lambda,\mu)$ as the dual problem, which is concave maximization (or convex minimization after sign flip). The dual problem variables $(\lambda,\mu)$ can be thought of as setting prices for constraint violations. The dual function $q(\lambda,\mu)$ gives for each $(\lambda,\mu)$ a lower bound on the primal optimum $p^*$ (weak duality: $q(\lambda,\mu) \le p^*$). Strong duality means $\max_{\lambda,\mu\ge0} q(\lambda,\mu) = p^*$. At optimum, the maximizing $(\lambda^,\mu^*)$ are optimal dual variables. They often have economic interpretation: e.g. in resource allocation, $\mu_j$ is the shadow price of resource $j$ (how much objective would worsen if resource $j$ capacity is slightly tightened). Complementary slackness then says if a resource isn’t fully used, its price is zero (it’s not scarce so extra of it has no value).

 

Geometrically, the dual variables define a hyperplane (through the origin in $\mathbb{R}^n$ dual space) $h(\lambda,\mu)(x) = \sum_i \lambda_i h_i(x) + \sum_j \mu_j g_j(x)$ that supports the epigraph of $f$ at the optimum. In fact, the stationary condition is exactly that $\nabla f(x^) + \sum_i \lambda_i \nabla h_i(x^) + \sum_j \mu_j \nabla g_j(x^) = 0$, which implies $\nabla (f + \sum_i \lambda_i h_i + \sum_j \mu_j g_j)(x^*)=0$. Since for feasible $x$, $h_i(x)=0$ and $g_j(x)\le0$, this means at $x^*$, $f(x) \ge f(x^*) + \nabla f(x^*)^T (x-x^*)$ and similarly linear terms for constraints give $0 \ge \lambda^T h(x^*) + \mu^T g(x^) +$ linear terms. Combining these weighted by $\lambda,\mu$ yields an inequality $f(x) \ge f(x^*) + (\text{some linear form})$ for all feasible $x$ that matches with $f(x^*)$ at $x^*$. This linear form is precisely the dual hyperplane that touches $f$ at $x^*$.

 

More concretely, for each active inequality $g_j$, the dual $\mu_j$ can be seen as the slope of the supporting line to $f$ along that constraint boundary. If we relax a constraint by a small $\epsilon$ (making feasible region slightly bigger if $\mu_j>0$), the optimal value should not increase; in fact one can show $d p^*/d(\text{constraint}j)|{\epsilon=0} = -\mu_j$. This sensitivity relation (from the envelope theorem or from dual solution) is why duals are important: they tell us marginal values of constraints.

 

Complementary slackness intuition: If a constraint is not active, it doesn’t affect the optimum, so its price $\mu_j$ should be zero. Conversely, if $\mu_j>0$, it indicates a “push back” from that constraint, meaning the solution is exactly on that constraint. Complementary slackness encodes this mutual dependency, reducing the KKT equations significantly (active set can be identified if one guesses which $\mu_j>0$).

 

Strong duality and no duality gap: For convex problems satisfying Slater’s condition (there exists a strictly interior feasible point for inequality constraints), we have zero duality gap ($p^* = d^*$). This is very useful: solving the dual (often easier, smaller dimensional or more separable) yields the primal solution indirectly. Sometimes the dual can be solved analytically when primal can’t. Also, stopping criteria for iterative solvers often involve duality gap (if we have a primal and dual feasible solution with nearly equal objective, we’re close to optimal).

 

For example, in support vector machines (SVMs), one solves the dual problem (which is easier as a QP in dual variables bounded between 0 and C, with one equality constraint) and then recovers the primal (weights vector) from a weighted combination of support vectors (those with nonzero dual $\alpha_i$). The support vectors are exactly the data points corresponding to active constraints (they lie on the margin, making the inequality $y_i(w^T x_i + b) - 1 \le 0$ tight). Their dual $\alpha_i$ are the support vector weights.

 

Economic and geometric summary: The primal problem asks “what is the smallest loss for a feasible plan?” The dual asks “what is the largest reward for a certifying combination of constraints?”. At optimum, both meet. The dual variables are like a witness proving optimality: they provide a lower bound that matches the primal upper bound. For convex optimization, constructing dual variables that satisfy KKT is often how you prove something is optimal. E.g. for linear programming, complementary slackness can identify optimal basis, etc.

 

Finally, Slater’s condition: if there is a strictly feasible $x$ (s.t. $h_i(x)=0$, $g_j(x)<0$ for all $j$), then strong duality holds. If not, there can be a gap (though for linear problems there is always either a solution or an extreme ray proving infeasibility – dual helps with that too by Farkas’ lemma). Slater’s condition also allows us to derive KKT by Lagrange saddle-point arguments.

 

In conclusion, optimality conditions and duality give us an alternative “dual view” of convex optimization: instead of directly searching in primal variable $x$, we can think in terms of dual variables that measure constraint trade-offs. This viewpoint often simplifies problems (converting constrained problems to unconstrained duals), provides sensitivity analysis, and deepens understanding: every constraint induces a “force” at optimum, and every unconstrained direction at optimum must have zero force (gradient zero). The harmony of these forces is succinctly captured by the KKT conditions, which can be seen as the equations of equilibrium for the optimization problem.