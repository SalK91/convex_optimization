First-order methods (gradient descent and its variants) use only gradient information and can be slow on ill-conditioned problems. Newton’s method is a second-order algorithm that uses the Hessian (matrix of second derivatives) to curvature-correct the steps. Newton’s method can converge in far fewer iterations — often quadratically fast near the optimum — by taking into account how the gradient changes, not just its value.

### Newton’s Method: Using Second-Order Information

Newton’s method is derived from the second-order Taylor approximation of $f$ around the current point $x_k$. We approximate:

$$
f(x_k + d) \approx f(x_k)
+ \nabla f(x_k)^\top d
+ \tfrac{1}{2} \, d^\top \nabla^2 f(x_k) \, d
$$


where $\nabla^2 f(x_k)$ (the Hessian matrix $H_k$) captures the local curvature. This quadratic model of $f$ is minimized (setting derivative to zero) by solving $H_k,d = -\nabla f(x_k)$. Thus the Newton step is

$$
d_{\text{newton}} = -[\nabla^2 f(x_k)]^{-1} \, \nabla f(x_k)
$$


and the update becomes $x_{k+1} = x_k + d_{\text{newton}} = x_k - H_k^{-1}\nabla f(x_k)$. In other words, we scale the gradient by the inverse Hessian, which adjusts the step length in each direction according to curvature. Directions in which the function curves gently (small Hessian eigenvalues) get a larger step, and directions of steep curvature (large eigenvalues) get a smaller step. This preconditioning by $H_k^{-1}$ leads to much more isotropic progress toward the optimum.

**Geometric interpretation:** Newton’s method is effectively performing gradient descent in a re-scaled space where the metric is defined by the Hessian. At $x_k$, consider the local quadratic approximation $q_k(d) = f(x_k) + \nabla f(x_k)^\top d + \frac{1}{2}d^\top H_k d$. This is a bowl-shaped function (assuming $H_k$ is positive-definite, which is true if $f$ is locally convex). The minimizer of $q_k$ is $d_{\text{newton}}$ as above. Newton’s method jumps directly to the bottom of this local quadratic model. If the model were exact (as it would be for a quadratic $f$), Newton’s step would land at the exact minimizer in one iteration. For non-quadratic $f$, the step is only approximate, but as $x_k$ approaches $x^*$, the quadratic model becomes very accurate and Newton steps become nearly exact.

One way to view Newton’s update is as iterative refinement: the update $x_{k+1} = x_k - H_k^{-1}\nabla f(x_k)$ solves the linear system $\nabla^2 f(x_k), (x_{k+1}-x_k) = -\nabla f(x_k)$, which is the Newton condition for a stationary point of the second-order model. Thus Newton’s method finds where the gradient would be zero if the current local curvature remained constant. This typically yields a huge improvement in $f$. In effect, Newton’s method stretches/scales space so that in the new coordinates the function looks like a unit ball shape (equal curvature in all directions), then it makes a standard step. After each step, a new linearization occurs at the new point.

### Convergence Properties and Affine Invariance

When $f$ is strongly convex with a Lipschitz-continuous Hessian, Newton’s method exhibits quadratic convergence in a neighborhood of the optimum. This means once $x_k$ is sufficiently close to $x^*$, the error shrinks squared at each step: $|x_{k+1}-x^*| = O(|x_k - x^*|^2)$. Equivalently, the number of correct digits in the solution roughly doubles every iteration. In terms of $f(x_k) - f(x^*)$, if gradient descent was $O(c^k)$ and accelerated gradient $O(1/k^2)$, Newton’s local rate is on the order of $O(c^{2^k})$ – extremely fast. For example, if you are 1% away from optimum at iteration $k$, Newton’s method might be 0.01% away at iteration $k+1$. This fast convergence is what makes Newton’s method so powerful for well-behaved problems. (Formally, one can show near $x^*$: $f(x_{k+1})-f(x^*) \le C [f(x_k)-f(x^*)]^2$ under appropriate conditions, hence the log of the error drops exponentially.)

However, global convergence is not guaranteed without modifications: if started far from the optimum or if $H_k$ is not positive-definite, the Newton step might not even be a descent direction (e.g., on nonconvex or badly scaled functions, Newton’s step can overshoot or go to a saddle). To address this, in practice one uses a damped Newton method: incorporate a step size $\lambda_k\in(0,1]$ and update $x_{k+1} = x_k - \lambda_k H_k^{-1}\nabla f(x_k)$. Typically $\lambda_k$ is chosen by a line search to ensure $f(x_{k+1})<f(x_k)$. Early on, $\lambda_k$ might be small (cautious steps) while $x_k$ is far from optimum, but eventually $\lambda_k$ can be taken as 1 (full Newton steps) in the vicinity of the solution, recovering the rapid quadratic convergence. This strategy ensures global convergence: from any starting point in a convex problem, damped Newton will converge to $x^*$.

A remarkable property of Newton’s method is affine invariance. This means the trajectory of Newton’s method is independent of linear coordinate transformations of the problem. If we apply an invertible affine mapping $y = A^{-1}x$ to the variables and solve in $y$-space, Newton’s steps in $y$ map exactly to Newton’s steps in $x$-space under $A$. In contrast, gradient descent is not affine invariant (scaling coordinates stretches the gradient in those directions, affecting the path and convergence speed). Affine invariance highlights that Newton’s method automatically handles conditioning and scaling: by using $H_k^{-1}$ it “preconditions” the problem optimally for the local quadratic structure. Another way to say this: Newton’s method is invariant to quadratic change of coordinates, because the Hessian provides the curvature metric. This is why Newton’s method is extremely effective on ill-conditioned problems; it essentially neutralizes the condition number by working in the Hessian’s eigenbasis where the function looks round.

Cost per iteration: The main drawback of Newton’s method is the cost of computing and inverting the $n \times n$ Hessian. This is $O(n^3)$ in general for matrix inversion or solving $H_k d = -\nabla f$ (though exploiting structure or approximations can reduce this). For very high-dimensional problems (like $n$ in the millions), Newton’s method becomes impractical. It’s mainly used when $n$ is moderate (up to a few thousands perhaps), or $H_k$ has special structure (sparse or low-rank updates). Each Newton iteration is expensive, but ideally you need far fewer iterations than first-order methods. There is a trade-off between doing more cheap steps (gradient descent) versus fewer expensive steps (Newton).

### Quasi-Newton Methods (BFGS, L-BFGS)
Quasi-Newton methods aim to retain the fast convergence of Newton’s method without having to compute the exact Hessian. They do this by building up an approximate Hessian inverse from successive gradient evaluations. The most famous is the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm, which iteratively updates a matrix $H_k$ intended to approximate $[\nabla^2 f(x_k)]^{-1}$ (the inverse Hessian). At each iteration, after computing $\nabla f(x_k)$, the difference in gradients $\Delta g = \nabla f(x_k) - \nabla f(x_{k-1})$ and the step $\Delta x = x_k - x_{k-1}$ are used to adjust the previous estimate $H_{k-1}$ into a new matrix $H_k$ that satisfies the secant condition: $H_k, \Delta g = \Delta x$. This condition ensures $H_k$ captures the curvature along the most recent step. The BFGS update formula (a specific symmetric rank-two update) guarantees that $H_k$ remains positive-definite and tends to become a better approximation over time
stat.cmu.edu
. In simplified terms, BFGS “learns” the curvature of $f$ as the iterations progress, by observing how the gradient changes with steps.

BFGS updates have the form:
​
$$
H_k =
\left(I - \frac{\Delta x \, \Delta g^\top}{\Delta g^\top \Delta x}\right)
H_{k-1}
\left(I - \frac{\Delta g \, \Delta x^\top}{\Delta g^\top \Delta x}\right)
+ \frac{\Delta x \, \Delta x^\top}{\Delta g^\top \Delta x}
$$


which is efficient to compute (rank-2 update on the matrix). One can show (under certain assumptions) that these updates lead $H_k$ to converge to the true inverse Hessian as $k$ grows. Practically, after enough iterations, the direction $p_k = -H_k \nabla f(x_k)$ behaves like the Newton direction. BFGS with an appropriate line search is known to achieve superlinear convergence (faster than any geometric rate, though not quite quadratic) once in the neighborhood of the optimum, for strongly convex functions. It’s a very effective compromise: each iteration is only $O(n^2)$ to update the $H_k$ and compute $p_k$ (much cheaper than $O(n^3)$ for solving Newton equations), but the iteration count remains low.

For very large $n$, storing the full $H_k$ becomes memory-intensive ($n^2$ elements). L-BFGS (Limited-memory BFGS) addresses this by never storing the full matrix; instead it maintains a history of only the last $m$ updates $(\Delta x, \Delta g)$ and implicitly defines $H_k$ via this limited history. The user specifies a small memory parameter (say $m=5$ or $10$), so L-BFGS uses only the last $m$ gradient evaluations to build a compressed approximate Hessian. Each iteration then costs $O(nm)$, which is only linear in $n$. L-BFGS is very popular for large-scale convex optimization because it often provides a good acceleration over plain gradient descent with minimal overhead in memory/computation.

**Quasi-Newton vs Newton:** Quasi-Newton methods, especially BFGS, often approach the performance of Newton’s method without needing an analytic Hessian. They are not affine invariant (scaling the inputs can affect the updates), but they are far more robust than simple gradient descent on difficult problems. Since they rely only on gradient evaluations, they can be applied in situations where Hessians are unavailable or too expensive. In machine learning, BFGS/L-BFGS were historically popular for training logistic regression, CRFs, and other convex models before first-order stochastic methods became dominant for extremely large data. They are still used for moderate-scale problems or as subsolvers in higher-level algorithms.

**BFGS in action:** One way to appreciate BFGS is that it preconditions gradient descent on the fly. Early iterations of BFGS behave like a quick learning phase: the algorithm figures out an effective metric to apply to the gradients. After a while, the $H_k$ matrix it builds “whitens” the Hessian of $f$ – making the level sets more spherical – so that subsequent updates take nearly optimal routes to $x^*$. Indeed, BFGS determines the descent direction by preconditioning the gradient with curvature information accumulated from past steps. It’s like doing Newton’s method with an approximate Hessian that is refined over time.

**Summary:** Newton’s method uses second-order derivatives to achieve rapid (quadratic) convergence, at the expense of heavy per-iteration work. Quasi-Newton methods like BFGS approximate the second-order info with smart updating rules, achieving superlinear convergence in practice with much lower computational cost per iteration. They strike a balance between first-order and second-order methods and are often the fastest methods for smooth convex optimization when the problem size permits. The geometric insight is that both Newton and quasi-Newton are curvature-aware: they scale gradient directions according to the problem’s geometry, which dramatically improves convergence especially on ill-conditioned problems (where gradients alone struggle).