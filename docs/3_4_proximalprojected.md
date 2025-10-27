Real-world convex problems often involve constraints and nonsmooth terms. Projected gradient methods and proximal gradient methods extend our first-order algorithms to handle these situations by incorporating projection or proximal steps into the update.

### Projected Gradient Descent (Constraints)

Suppose we want to minimize $f(x)$ subject to $x \in \mathcal{X}$, where $\mathcal{X}$ is a convex feasible set (e.g. a polytope or ball). A gradient descent step $y_{k} = x_k - \alpha \nabla f(x_k)$ might produce a point $y_k$ outside $\mathcal{X}$. The idea of projected gradient descent is simple: after the gradient step, project $y_k$ back onto $\mathcal{X}$:
 
 $$
x_{k+1} = \Pi_X \big( x_k - \alpha \nabla f(x_k) \big)
$$


Here $\Pi_{\mathcal{X}}(y) = \arg\min_{x\in \mathcal{X}}|x - y|^2$ is the orthogonal projection onto $\mathcal{X}$. This two-step iteration ensures that $x_{k+1}\in\mathcal{X}$ for all $k$. Geometrically, we take the usual gradient descent step into $y_k$ which may lie off the feasible set, then find the closest feasible point. The correction is orthogonal to the boundary of $\mathcal{X}$ at the projection point (no component of the step along the boundary is wasted, since $y_k - x_{k+1}$ is perpendicular to $\mathcal{X}$). Thus, projected gradient still decreases $f(x)$ to first order while respecting the constraints.

For example, if $\mathcal{X}={x:|x|_2 \le 1}$ (the unit Euclidean ball), the projection is $\Pi{\mathcal{X}}(y) = \frac{y}{\max{1,|y|_2}}$: any $y$ outside the ball is radially scaled back to the boundary, and if $y$ is inside the ball it stays unchanged. Projected gradient descent is widely used in constrained convex optimization (like projection onto probability simplex, box constraints $x\in [l,u]$, etc.) because of its simplicity: one just needs a routine to project onto $\mathcal{X}$, which is often straightforward.

Convergence: If $f$ is convex and $L$-smooth, projected gradient descent inherits similar convergence guarantees as unconstrained gradient descent (albeit the analysis uses more geometry from $\mathcal{X}$). With a suitable step size, $f(x_k)\to f(x^*)$ and for strongly convex problems $x_k\to x^*$ linearly. The projection step does not spoil convergence; it only ensures feasibility. Intuitively, the error analysis includes an extra term for distance from $\mathcal{X}$, but the projection minimizes that, keeping the iterates as close as possible to the unconstrained path.

Projection is actually a special case of a more general operator important in convex optimization: the proximal operator.

### Proximal Operators and Proximal Gradient

Consider an optimization problem with a convex but nonsmooth term, for example:

$$
\min_{x \in \mathbb{R}^n} \; f(x) + g(x)
$$


where $f(x)$ is convex and differentiable (smooth loss or fit term) and $g(x)$ is convex but possibly nondifferentiable (regularizer or indicator of constraints). Here $g(x)$ could be things like $L_1$ norm ($\ell_1$ penalty), $\ell_\infty$ norm, indicator functions enforcing $x$ in some set, etc. The proximal gradient method addresses such problems by splitting the handling of $f$ and $g$.

We know how to deal with $f$ using a gradient step. To handle $g$, we use its proximal operator. The proximal operator of $g$ (with parameter $\lambda>0$) is defined as:

$$
\operatorname{prox}_{\lambda g}(y)
= \arg\min_x \left\{
g(x) + \frac{1}{2\lambda} \|x - y\|^2
\right\}
$$


This is the solution of a regularized minimization of $g(x)$ where we stay as close as possible to a given point $y$. In words, $\operatorname{prox}{\lambda g}(y)$ is a point that compromises between minimizing $g(x)$ and staying near $y$ (the quadratic term $\frac{1}{2\lambda}|x-y|^2$ keeps $x$ from straying too far from $y$). The parameter $\lambda$ scales how strongly we insist on proximity to $y$: as $\lambda \to 0$, $\operatorname{prox}{\lambda g}(y)\approx y$ (we barely move); as $\lambda$ grows, we’re willing to move farther to reduce $g(x)$. The proximal operator is well-defined for any closed convex $g$ (it has a unique minimizer due to strong convexity of the quadratic term), and it generalizes the notion of projection:

If $g(x)$ is the indicator function of a convex set $\mathcal{X}$ (meaning $g(x)=0$ if $x\in\mathcal{X}$ and $+\infty$ otherwise), then $\operatorname{prox}{\lambda g}(y)$ is exactly $\Pi{\mathcal{X}}(y)$. That’s because the minimization $\min_x {I_{\mathcal{X}}(x) + \frac{1}{2\lambda}|x-y|^2}$ forces $x$ to lie in $\mathcal{X}$ (outside $\mathcal{X}$ the objective is infinite) and then reduces to $\min_{x\in \mathcal{X}}|x-y|^2$, the definition of projection. Thus, projection is a special case of a proximal operator. Prox operators can be thought of as “softened” projections that not only enforce constraints but can also induce certain structures (like sparsity).

Given $\operatorname{prox}_{\lambda g}$, the proximal gradient method for $f+g$ works as follows:

1. Take a usual gradient descent step on the smooth part $f$: $y_{k} = x_k - \alpha \nabla f(x_k)$.

2. Apply the proximal operator of $g$: $x_{k+1} = \operatorname{prox}_{\alpha g}(y_k)$.

In formula form,

$$
x_{k+1} = \operatorname{prox}_{\alpha g}\!\left( x_k - \alpha \nabla f(x_k) \right)
$$


This update handles two things: the gradient step tries to reduce the objective $f$, and the prox step pulls the solution toward satisfying the “desirable structure” encoded by $g$. If $g$ is an indicator of $\mathcal{X}$, this becomes projected gradient descent (since $\operatorname{prox}{\alpha I{\mathcal{X}}}(y)=\Pi_{\mathcal{X}}(y)$). If $g$ is something like $\lambda |x|_1$, the prox step becomes a soft-thresholding of $y$ (encouraging sparsity). If $g$ is the absolute value function (total variation, etc.), prox becomes a shrinkage operator, and so on. The proximal gradient method is also known as Forward-Backward Splitting: a forward (gradient) step on $f$, followed by a backward (proximal) step on $g$.

Convergence: If $f$ is convex and $L$-smooth and $g$ is convex (proper, closed), then proximal gradient descent with a fixed step $\alpha \le 1/L$ converges to the optimum at essentially the same rate as plain gradient descent: $O(1/k)$ in the general convex case, and linear convergence if $f+g$ is strongly convex (under a suitable condition like $g$ being simple or strongly convex). The intuition is that the nonsmooth part $g$ doesn’t hurt the rate as long as its proximal operator is efficiently computable; the “heavy lifting” of ensuring a decrease is done by the smooth part’s Lipschitz condition and the prox step never increases the objective. However, each iteration’s cost includes computing $\operatorname{prox}_{\alpha g}$, which in some cases might be as hard as solving a subproblem. Fortunately, for many common $g$, the prox step is easy: e.g., $\ell_1$ norm (soft-thresholding), $\ell_2$ norm (shrink towards zero), indicator of linear constraints (simple clipping or normalization), etc.

Proximal point algorithm (implicit gradient): A conceptual algorithm worth mentioning is the proximal point method, which is like taking only implicit steps on the entire objective. It iterates $x_{k+1} = \operatorname{prox}{\lambda f}(x_k)$, i.e. solves $x{k+1} = \arg\min_x {f(x) + \frac{1}{2\lambda}|x-x_k|^2}$ exactly at each step. This is in general a difficult subproblem (you basically solve the original problem in each step!), so it’s not an algorithm you’d implement for a generic $f$. But theoretically, the proximal point method has strong convergence guarantees under very mild assumptions (it converges for any $\lambda>0$ as long as a minimizer exists). It provides an implicit stabilization of the iteration (always moving to a point that is an actual minimizer of a nearby objective), which is why it converges even for some nonconvex problems and monotone operators. In convex optimization, the proximal point algorithm is more of a theoretical tool—many modern algorithms (like ADMM and SVRG) can be interpreted as approximate or accelerated versions of it.

Proximal vs gradient: one can view the standard gradient descent as the limit of the proximal point method when $\lambda$ is small. Gradient descent does $x_{k+1}\approx x_k - \lambda \nabla f(x_k)$ which is the first-order condition of the prox subproblem if we only take an infinitesimal step. Proximal steps, in contrast, solve the subproblem exactly, which is why they can be more stable. Proximal gradient hits a sweet spot: we solve the easy part ($g$) exactly via prox, and handle the hard part ($f$) via gradient.

**Example – Lasso (L1 regularization):** Take $f(x) = \frac{1}{2}|Ax - b|^2$ (a least-squares loss) and $g(x) = \lambda |x|1$ (an $\ell_1$ penalty encouraging sparsity). The prox operator $\operatorname{prox}{\alpha \lambda |\cdot|1}(y)$ is the soft-thresholding: $[\operatorname{prox}{\alpha \lambda |\cdot|_1}(y)]_i = \text{sign}(y_i)\max{|y_i| - \alpha\lambda,,0}$. So proximal gradient descent (a.k.a. Iterative Shrinkage-Thresholding Algorithm (ISTA)) does:

$$
\begin{aligned}
y_k &= x_k - \alpha A^\top (A x_k - b), \\
x_{k+1,i} &= \operatorname{sign}(y_{k,i}) \, \max\{ |y_{k,i}| - \alpha \lambda, \, 0 \}.
\end{aligned}
$$


for each component $i$. This method will converge to the Lasso solution. Furthermore, one can add Nesterov acceleration to this (getting the FISTA algorithm), achieving an $O(1/k^2)$ rate for the objective similar to accelerated gradient.

In summary, proximal gradient methods allow us to tackle optimization problems with constraints or nonsmooth terms by splitting the problem into a smooth part (handled by a gradient step) and a simple nonsmooth part (handled by a prox step). The geometry underlying this is the idea of projection as a prox operator: we extend the notion of moving “back to the feasible region” (projection) to moving toward a region that yields lower $g(x)$ (proximal step). This framework vastly expands the scope of problems solvable by first-order methods, including Lasso, logistic regression with regularization, matrix norm minimizations, etc., all while maintaining convergence guarantees comparable to gradient descent.