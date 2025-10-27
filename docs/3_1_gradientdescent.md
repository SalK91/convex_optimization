Gradient descent is the most fundamental first-order method for convex optimization. It capitalizes on the fact that for a differentiable convex function $f:\mathbb{R}^n\to\mathbb{R}$, the gradient $\nabla f(x)$ indicates the direction of steepest increase (and $-\nabla f(x)$ the steepest decrease) at $x$ (see Section A, Chapter 6). At a global minimizer $x^*$ of a convex differentiable $f$, the first-order optimality condition $\nabla f(x^*) = 0$ holds, and no descent direction exists. Gradient descent iteratively moves opposite to the gradient, seeking such a stationary point which, in convex problems, guarantees optimality.

### Gradient Descent: Algorithm and Geometry

Starting from an initial guess $x_0\in\mathcal{X}$, the gradient descent update (for unconstrained problems $\mathcal{X}=\mathbb{R}^n$) is:

$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$


where $\alpha>0$ is a step size (also called learning rate). Geometrically, this update follows the tangent plane of $f$ at $x_k$ a short distance in the negative gradient direction. Intuitively, one approximates $f(x)$ near $x_k$ by the first-order Taylor expansion $f(x)\approx f(x_k) + \langle \nabla f(x_k),,x - x_k\rangle$ and then minimizes this linear model plus a small quadratic regularization to keep the step local (see Section A’s discussion of Taylor expansions). The result is exactly the gradient step $x_{k+1}=x_k - \alpha \nabla f(x_k)$. This can be viewed as the steepest descent direction in the Euclidean metric (the gradient direction is orthogonal to level sets of $f$ and points toward lower values). By choosing $\alpha$ appropriately (e.g. via a line search or a bound on the Lipschitz constant of $\nabla f$), each step guarantees a sufficient decrease in $f$.

Recall: If $f$ is $L$-smooth (i.e. $\nabla f$ is Lipschitz continuous with constant $L$; see Section B, Lipschitz Continuity), then taking $\alpha \le 1/L$ ensures $f(x_{k+1}) \le f(x_k)$ for gradient descent. The descent lemma (from Section A) ensures the update is stable and $f$ decreases because the linear improvement $- \alpha |\nabla f(x_k)|^2$ dominates the quadratic error $\frac{L\alpha^2}{2}|\nabla f(x_k)|^2$ when $\alpha \le 1/L$.

Choosing the step size $\alpha$: A smaller $\alpha$ yields cautious, stable steps (useful if the landscape is ill-conditioned), whereas a larger $\alpha$ accelerates progress but can overshoot and even diverge if too large. In practice one may use backtracking or exact line search to adapt $\alpha$. Under constant $\alpha$ within the safe range, convergence is guaranteed for convex $f$.

### Convergence Rates: Convex vs Strongly Convex

For general convex and $L$-smooth $f$, gradient descent achieves a sublinear convergence rate in function value. In particular, after $k$ iterations one can guarantee

$$
f(x_k) - f(x^*) \le \frac{L \|x_0 - x^*\|^2}{2k},
\quad \text{so} \quad
f(x_k) - f(x^*) = O(1/k)
$$

This means to get within $\varepsilon$ of the optimal value, on the order of $1/\varepsilon$ iterations are needed. This $O(1/k)$ rate is often called sublinear convergence. If we further assume $f$ is $\mu$-strongly convex (see Section B: strong convexity ensures a unique minimizer and a quadratic lower bound), gradient descent’s convergence improves dramatically to a linear rate. In fact, for an $L$-smooth, $\mu$-strongly convex $f$, there exist constants $0<c<1$ such that

$$
f(x_k) - f(x^*) = O(c^k)
$$


i.e. $f(x_k)$ approaches $f(x^)$ geometrically fast. Equivalently, the error decreases by a constant factor $(1-\tfrac{\mu}{L})$ (or similar) at every iteration. For example, choosing $\alpha = 2/(\mu+L)$ yields $|x_{k+1}-x^| \le \frac{L-\mu}{L+\mu},|x_k - x^|$, so $|x_k - x^| = O((\frac{L-\mu}{L+\mu})^k)$. Such linear convergence (also called exponential convergence) implies $O(\log(1/\varepsilon))$ iterations to reach accuracy $\varepsilon$. Strong convexity essentially provides a uniformly curved bowl shape, preventing flat regions and guaranteeing that gradient steps won’t diminish to zero too early.

**Summary:** For convex $f$: $f(x_k)-f(x^) = O(1/k)$; if $f$ is $\mu$-strongly convex: $f(x_k)-f(x^) = O((1-\eta\mu)^k)$ (linear). These rates assume appropriate constant step sizes. (See Section B, “Function Properties,” for a detailed table of convergence rates of first-order methods.)

When does gradient descent perform poorly? If the problem is ill-conditioned (the Hessian has a high condition number $\kappa = L/\mu$), progress along some directions is much slower than others. The iterates may “zig-zag” through narrow valleys, requiring very small $\alpha$ to maintain stability. In fact, the iteration complexity of gradient descent is $O(\kappa \log(1/\varepsilon))$ for strongly convex problems – linearly dependent on the condition number. Chapter C3 on Newton’s method will address this issue by preconditioning with second-order information.

### Subgradient Method for Nondifferentiable Convex Functions

Gradient descent requires $\nabla f(x)$ to exist. Many convex problems in machine learning involve nondifferentiable objectives (e.g. hinge loss in SVMs, $L_1$-regularization in Lasso, ReLU activations). Subgradient methods generalize gradient descent to such functions using a subgradient in place of the gradient. Recall from Section B that for a convex function $f$, a subgradient $g$ at $x$ is any vector that supports $f$ at $x$, meaning $f(y) \ge f(x) + \langle g,,y-x\rangle$ for all $y$. The set of all subgradients at $x$ is the subdifferential $\partial f(x)$. If $f$ is differentiable at $x$, then $\partial f(x)={\nabla f(x)}$; if $f$ has a kink, $\partial f(x)$ is a whole set (e.g. for $f(u)=|u|$, any $g\in[-1,1]$ is a subgradient at $u=0$).

The subgradient method update mirrors gradient descent:

$$
x_{k+1} = x_k - \eta_k g_k
$$


where $g_k \in \partial f(x_k)$ is any chosen subgradient at $x_k$, and $\eta_k>0$ is a step size. If $x$ is constrained to a convex set $\mathcal{X}$, one includes a projection onto $\mathcal{X}$: $x_{k+1} = \Pi_{\mathcal{X}}!(x_k - \eta_k g_k)$ (this reduces to the unconstrained update if $\mathcal{X}=\mathbb{R}^n$). Geometrically, even though $f$’s graph may have corners, a subgradient $g_k$ defines a supporting hyperplane at $(x_k, f(x_k))$. The update $-\eta_k g_k$ is a feasible descent direction because $\langle g_k, x^* - x_k\rangle \ge f(x^*)-f(x_k)$ for the minimizer $x^*$, by convexity. Thus, moving a small amount opposite to $g_k$ tends to decrease $f$. One can still “snap back” to the feasible region via projection as needed, analogous to the projected gradient step for constraints. This guarantees $x_k$ stays in $\mathcal{X}$ (see below for projection geometry).

**Convergence of subgradient methods:** Unlike gradient descent, we cannot generally use a fixed $\eta$ to get convergence (the method won’t settle to a single point because of the zig-zagging on corners). Instead, a common strategy is a diminishing step size or an averaging of iterates. A typical result for minimizing a convex $f$ with Lipschitz-bounded subgradients ($|g_k|\le G$) is: using $\eta_k = \frac{R}{G\sqrt{k}}$ (with $R = |x_0 - x^*|$), the averaged iterate $\bar{x}T = \frac{1}{T}\sum{t=1}^T x_t$ satisfies

$$
f(\bar{x}_T) - f(x^*) \le \frac{R G}{T} = O(1/T)
$$


This is a sublinear rate slower than $O(1/T)$, reflecting the cost of nondifferentiability. In fact, $O(1/\sqrt{T})$ is the worst-case optimal rate for first-order methods on nonsmooth convex problems (no algorithm can generally do better than this without additional structure). If $f$ is also strongly convex, faster subgradient convergence is possible (e.g. $O(\log T / T)$ with specialized step schemes or using a known optimal value in Polyak’s step size), but it’s still slower than the smooth case. Importantly, the subgradient method does not converge to the exact minimizer unless $\eta_k\to 0$; typically one gets arbitrarily close but keeps bouncing around the optimum. This is why an ergodic average $\bar{x}_T$ is used in the guarantee above – it smooths out oscillations.

**Projection and feasibility:** When constraints $\mathcal{X}$ are present, the subgradient update includes a projection $\Pi_{\mathcal{X}}(\cdot)$. Recall from Section A (Geometry of Orthogonal Projection) that for a closed convex set $\mathcal{X}$, the projection $\Pi_{\mathcal{X}}(y) = \arg\min_{x\in\mathcal{X}}|x - y|$ yields the closest feasible point to $y$. The projection error $y - \Pi_{\mathcal{X}}(y)$ is orthogonal to $\mathcal{X}$ at the projection point (no improvement can be made along the feasible surface). Thus, $x_{k+1} = \Pi_{\mathcal{X}}(x_k - \eta g_k)$ can be seen as: take a step in the subgradient direction, then drop perpendicularly back into the set. This ensures feasibility of iterates while still achieving descent on $f$. For example, if $\mathcal{X}$ is the $\ell_2$ unit ball, the projection $\Pi_{\mathcal{X}}(y)$ simply scales $y$ to have norm 1 if it was outside.

**Use cases:** Subgradient methods shine in nonsmooth problems like L1-regularized models (Lasso), SVM hinge loss, and combinatorial convex relaxations, where gradients are not available. They are very simple (each step is like gradient descent), but one must carefully tune the step schedule. In practice, subgradient methods can be slow to get high accuracy; however, their simplicity and ability to handle nondifferentiability make them a go-to baseline. Techniques like momentum or adaptive step sizes can sometimes improve practical performance, but fundamentally $O(1/\sqrt{T})$ is the regime for nonsmooth convex minimization.

**Example:** Consider $f(x) = |x|$ (which is nonsmooth at 0). The subgradient method for $\min_x |x|1$ (with no smooth part) would at iteration $k$ choose some $g_k \in \partial |x_k|$ (which could be $g_k = \text{sign}(x_k)$ componentwise), and do $x{k+1}=x_k - \eta g_k$. This essentially performs soft-thresholding on each coordinate: if $x_k$ was positive, it decreases it, if negative, increases it, if zero, it stays within [-$\eta,\eta$]. Indeed, with an appropriate choice of $\eta$, one can show $x_k$ converges to 0 (the minimizer). This principle underlies the ISTA/FISTA algorithms for Lasso: each step is $x{k+1} = S{\lambda\eta}(x_k - \eta \nabla f_{\text{smooth}}(x_k))$, where $S_{\tau}(y)=\text{sign}(y)\max{|y|-\tau,,0}$ is the soft-thresholding operator. We will see proximal gradient methods in Chapter C4 that formalize this approach.

**Summary:** Gradient descent is the workhorse for smooth convex problems, enjoying $O(1/k)$ or better convergence and simplicity. Its limitation is the requirement of differentiability and sometimes slow progress in ill-conditioned settings. The subgradient method extends applicability to nondifferentiable convex functions at the expense of slower convergence. In both cases, the geometry of convexity (supporting hyperplanes, gradient orthogonality, etc.) underpins why moving opposite to a (sub)gradient leads toward the optimum.