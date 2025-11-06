# Chapter 3: Multivariable Calculus for Optimization

Optimisation is about finding points that minimise (or maximise) a function. To do that analytically, we need to understand gradients, Hessians, Taylor expansions, and first-/second-order optimality conditions.



## 3.1 Gradients, Jacobians, and Hessians

Let $f : \mathbb{R}^n \to \mathbb{R}$.  
The gradient of $f$ at $x$ is the column vector
$$
\nabla f(x) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1}(x) \\
\vdots \\
\frac{\partial f}{\partial x_n}(x)
\end{bmatrix}.
$$
It points in the direction of steepest increase of $f$.

The directional derivative in direction $u$ is $D_u f(x) = \lim_{t\to0} \frac{f(x+tu)-f(x)}{t} = \langle \nabla f(x), u\rangle$. This shows how the gradient inner product with $u$ gives the instantaneous rate of change of $f$ along $u$. In particular, $D_u f(x)$ is maximized when $u$ points along $\nabla f(x)$ (steepest ascent) and minimized when $u$ is opposite.


If $F : \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J_F(x)$ is the $m \times n$ matrix of partial derivatives.



> Gradient Lipschitz continuity: A concept often used in convergence analysis is Lipschitz continuity of the gradient. If there exists $L$ such that $|\nabla f(x) - \nabla f(y)| \le L |x-y|$ for all $x,y$, we say the gradient is $L$-Lipschitz (or $f$ is $L$-smooth). $L$ is essentially an upper bound on the Hessian eigenvalues (for $\ell_2$ norm): $L \ge \lambda_{\max}(\nabla^2 f(x))$ for all $x$. Smoothness is important because it ensures gradient descent with step $\alpha = 1/L$ converges, and it gives a bound $f(x_{k+1}) \le f(x_k) - \frac{1}{2L}|\nabla f(x_k)|^2$ (so the function value decreases at least proportionally to the squared gradient norm). Many convex functions in optimization are $L$-smooth (e.g. quadratic forms with $\lambda_{\max}(Q)=L$). Smoothness together with strong convexity (defined shortly) yields linear convergence rates for gradient descent.

> Strong convexity: A differentiable function $f$ is $\mu$-strongly convex if $f(y) \ge f(x) + \langle \nabla f(x), y-x \rangle + \frac{\mu}{2}|y-x|^2$ for all $x,y$. Equivalently, $f(x) - \frac{\mu}{2}|x|^2$ is convex, which implies $\nabla^2 f(x) \succeq \mu I$ (Hessian bounded below by $\mu$) when $f$ is twice differentiable. Strong convexity means $f$ has a quadratic curvature of at least $\mu$ – it grows at least as fast as a parabola. Strongly convex functions have unique minimizers (the bowl can’t flatten out). They also yield much faster convergence: for $\mu$-strongly convex and $L$-smooth $f$, gradient descent with $\alpha=1/L$ converges like $(1-\mu/L)^k$ (linear rate). Intuitively, the condition number $\kappa = L/\mu$ comes into play. Examples: the quadratic form above is strongly convex with $\mu = \lambda_{\min}(Q)$. Adding a small ridge term $\frac{\mu}{2}|x|^2$ to any convex $f$ makes it $\mu$-strongly convex and improves conditioning at the cost of bias.

The Hessian of $f$ is the $n \times n$ matrix of second partial derivatives:
$$
\nabla^2 f(x) =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}.
$$

If $f$ is twice continuously differentiable, then $\nabla^2 f(x)$ is symmetric (Clairaut’s theorem).

 
Example – quadratic function: $f(x) = \frac{1}{2}x^TQx - b^T x$. Here $\nabla f(x) = Qx - b$ (linear), and $\nabla^2 f(x) = Q$. Solving $\nabla f=0$ yields $Qx=b$, so if $Q \succ 0$ the unique minimizer is $x^* = Q^{-1}b$. The Hessian being $Q \succ 0$ confirms convexity. If $Q$ has large eigenvalues, gradient $Qx - b$ changes rapidly in some directions (steep narrow valley); if some eigenvalues are tiny, gradient hardly changes in those directions (flat valley). This aligns with earlier discussions: condition number of $Q$ controls difficulty of minimizing $f$.
 
## 3.2 First-order Taylor approximation

For differentiable $f$, we have the first-order Taylor expansion around $x$:
$$
f(x + d) \approx f(x) + \nabla f(x)^\top d~.
$$

Interpretation:

- $\nabla f(x)$ gives the best linear approximation.
- The linear model predicts how $f$ changes if we move by $d$.

This is the basis of first-order optimisation methods like gradient descent.

 
## 3.3 Second-order Taylor approximation

If $f$ is twice differentiable, then
$$
f(x + d) \approx f(x)
+ \nabla f(x)^\top d
+ \frac{1}{2} d^\top \nabla^2 f(x) d~.
$$

If $\nabla^2 f(x)$ is positive semidefinite, the quadratic term is always $\ge 0$. Locally, $x$ is in a “bowl”. If $\nabla^2 f(x)$ is indefinite, the landscape can curve up in some directions and down in others — typical of saddle points.

 
## 3.4 Unconstrained optimality conditions

Suppose we want to solve
$$
\min_x f(x)
$$
with no constraints.

A point $x^*$ is called a critical point if $\nabla f(x^*) = 0$.

- First-order necessary condition:  
  If $x^*$ is a (local) minimiser and $f$ is differentiable, then
  $$
  \nabla f(x^*) = 0~.
  $$

- Second-order necessary condition:  
  If $x^*$ is a (local) minimiser and $f$ is twice differentiable,
  $$
  \nabla f(x^*) = 0,
  \quad
  \nabla^2 f(x^*) \succeq 0
  \quad\text{(i.e. PSD).}
  $$

- Second-order sufficient condition:  
  If
  $$
  \nabla f(x^*) = 0,
  \quad
  \nabla^2 f(x^*) \succ 0
  \quad\text{(i.e. PD),}
  $$
  then $x^*$ is a strict local minimiser.

Now, here is where convexity changes everything.

> If $f$ is convex, then any point $x^*$ with $\nabla f(x^*) = 0$ is not just a local minimiser — it is a global minimiser (Boyd and Vandenberghe, 2004). No second-order check is even needed.

 
## 3.5 Gradients as normals to level sets

A level set of a differentiable function $f$ is
$$
L_c = \{ x : f(x) = c \}.
$$

At any point $x$ with $\nabla f(x) \ne 0$, the gradient $\nabla f(x)$ is orthogonal to the level set $L_{f(x)}$. Intuitively, the level set is like a contour line, and the gradient is perpendicular to it, pointing toward larger values of $f$.

In optimisation, if we want to decrease $f$, we move roughly in direction $-\nabla f(x)$.

This geometric fact recurs later in constrained optimisation and KKT: at optimality, the gradient of the objective lies in the span of gradients of active constraints.

 
## 3.6 Convexity and the Hessian

If $f : \mathbb{R}^n \to \mathbb{R}$ is twice differentiable, then:

- $f$ is convex if and only if $\nabla^2 f(x)$ is positive semidefinite for all $x$ in its domain (Boyd and Vandenberghe, 2004).
- $f$ is strictly convex if $\nabla^2 f(x)$ is positive definite for all $x$.

  