# Chapter 3: Multivariable Calculus for Optimization

Optimisation is about finding points that minimise (or maximise) a function. To do that analytically, we need to understand gradients, Hessians, Taylor expansions, and first-/second-order optimality conditions.

We work in $\mathbb{R}^n$.

---

## 3.1 Gradients, Jacobians, and Hessians

Let $f : \mathbb{R}^n \to \mathbb{R}$.  
The **gradient** of $f$ at $x$ is the column vector
$$
\nabla f(x) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1}(x) \\
\vdots \\
\frac{\partial f}{\partial x_n}(x)
\end{bmatrix}.
$$
It points in the direction of steepest increase of $f$.

If $F : \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** $J_F(x)$ is the $m \times n$ matrix of partial derivatives.

The **Hessian** of $f$ is the $n \times n$ matrix of second partial derivatives:
$$
\nabla^2 f(x) =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}.
$$

If $f$ is twice continuously differentiable, then $\nabla^2 f(x)$ is symmetric (Clairaut’s theorem).

---

## 3.2 First-order Taylor approximation

For differentiable $f$, we have the first-order Taylor expansion around $x$:
$$
f(x + d) \approx f(x) + \nabla f(x)^\top d~.
$$

Interpretation:

- $\nabla f(x)$ gives the best linear approximation.
- The linear model predicts how $f$ changes if we move by $d$.

This is the basis of first-order optimisation methods like gradient descent.

---

## 3.3 Second-order Taylor approximation

If $f$ is twice differentiable, then
$$
f(x + d) \approx f(x)
+ \nabla f(x)^\top d
+ \frac{1}{2} d^\top \nabla^2 f(x) d~.
$$

If $\nabla^2 f(x)$ is positive semidefinite, the quadratic term is always $\ge 0$. Locally, $x$ is in a “bowl”. If $\nabla^2 f(x)$ is indefinite, the landscape can curve up in some directions and down in others — typical of saddle points.

---

## 3.4 Unconstrained optimality conditions

Suppose we want to solve
$$
\min_x f(x)
$$
with no constraints.

A point $x^*$ is called a **critical point** if $\nabla f(x^*) = 0$.

- **First-order necessary condition**:  
  If $x^*$ is a (local) minimiser and $f$ is differentiable, then
  $$
  \nabla f(x^*) = 0~.
  $$

- **Second-order necessary condition**:  
  If $x^*$ is a (local) minimiser and $f$ is twice differentiable,
  $$
  \nabla f(x^*) = 0,
  \quad
  \nabla^2 f(x^*) \succeq 0
  \quad\text{(i.e. PSD).}
  $$

- **Second-order sufficient condition**:  
  If
  $$
  \nabla f(x^*) = 0,
  \quad
  \nabla^2 f(x^*) \succ 0
  \quad\text{(i.e. PD),}
  $$
  then $x^*$ is a strict local minimiser.

Now, here is where convexity changes everything.

> If $f$ is convex, then **any** point $x^*$ with $\nabla f(x^*) = 0$ is not just a local minimiser — it is a **global minimiser** (Boyd and Vandenberghe, 2004). No second-order check is even needed.

---

## 3.5 Gradients as normals to level sets

A **level set** of a differentiable function $f$ is
$$
L_c = \{ x : f(x) = c \}.
$$

At any point $x$ with $\nabla f(x) \ne 0$, the gradient $\nabla f(x)$ is orthogonal to the level set $L_{f(x)}$. Intuitively, the level set is like a contour line, and the gradient is perpendicular to it, pointing toward larger values of $f$.

In optimisation, if we want to decrease $f$, we move roughly in direction $-\nabla f(x)$.

This geometric fact recurs later in constrained optimisation and KKT: at optimality, the gradient of the objective lies in the span of gradients of active constraints.

---

## 3.6 Convexity and the Hessian

If $f : \mathbb{R}^n \to \mathbb{R}$ is twice differentiable, then:

- $f$ is **convex** if and only if $\nabla^2 f(x)$ is positive semidefinite for all $x$ in its domain (Boyd and Vandenberghe, 2004).
- $f$ is **strictly convex** if $\nabla^2 f(x)$ is positive definite for all $x$.

---


## 3.7 Takeaways

1. Gradients and Hessians describe local behaviour.
2. In unconstrained problems, $\nabla f(x^*)=0$ is necessary for optimality.
3. For convex $f$, $\nabla f(x^*)=0$ is also sufficient for *global* optimality.
4. The Hessian being PSD everywhere is the smooth-test for convexity.

In the next chapter, we step back from calculus and talk about *sets*. Convex sets give us the language to talk about feasible regions, constraints, and geometry of optimisation.
<!-- 
---

## References (Chapter 3)

- Boyd, S. and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.  
- Nesterov, Y. (2018). *Lectures on Convex Optimization*. Springer.

Calculus provides the analytical tools to optimize functions: it describes how functions change when we tweak the input. In convex optimization we often assume differentiability (at least for the objective, if not constraints), so we rely on gradients and Hessians to characterize optimal points and design algorithms.

**Gradient and directional derivative:** Let $f: \mathbb{R}^n \to \mathbb{R}$ be differentiable. The gradient $\nabla f(x)$ is the vector of partial derivatives

$$
\nabla f(x) =
\begin{bmatrix}
\dfrac{\partial f}{\partial x_1} \\
\vdots \\
\dfrac{\partial f}{\partial x_n}
\end{bmatrix}
$$


so that for a small step $h$, $f(x+h) \approx f(x) + \langle \nabla f(x),h\rangle$. This linear approximation is the first-order Taylor expansion. The gradient $\nabla f(x)$ points in the direction of steepest increase of $f$; $-\nabla f(x)$ is the direction of steepest decrease. Specifically, $\nabla f(x)$ is orthogonal to level sets of $f$ at $x$. In optimization, setting $\nabla f(x) = 0$ finds stationary points (candidates for optima). Gradient descent uses the update $x_{k+1} = x_k - \alpha \nabla f(x_k)$, taking a small step opposite the gradient to reduce $f$. The magnitude $|\nabla f(x)|$ indicates how steep $f$ is; when $\nabla f(\hat{x})=0$, the function is flat to first order at $\hat{x}$. For convex $f$, any stationary point is a global minimum.

The **directional derivative** in direction $u$ is $D_u f(x) = \lim_{t\to0} \frac{f(x+tu)-f(x)}{t} = \langle \nabla f(x), u\rangle$. This shows how the gradient inner product with $u$ gives the instantaneous rate of change of $f$ along $u$. In particular, $D_u f(x)$ is maximized when $u$ points along $\nabla f(x)$ (steepest ascent) and minimized when $u$ is opposite.

**Jacobian for vector-valued mappings:** If $g: \mathbb{R}^n \to \mathbb{R}^m$ (with $m>1$ outputs), the Jacobian matrix $J_g(x)$ is the $m \times n$ matrix of partial derivatives: its $(i,j)$ entry is $\partial g_i/\partial x_j$. The $i$ th row is $(\nabla g_i(x))^T$. For example, if $g(x) = Ax$ (linear map), then $J_g(x)=A$ constant. If $g(x) = (f(x), h(x))$ combines two scalars, the Jacobian has two rows: $\nabla f(x)^T$ and $\nabla h(x)^T$. The Jacobian represents the best linear approximation of $g$ near $x$: $g(x+h) \approx g(x) + J_g(x),h$. When $m=n$ and $J_g(x)$ is invertible, $g$ is locally invertible (by the Inverse Function Theorem) and the Jacobian’s determinant indicates how volumes scale under $g$. In optimization, Jacobians appear in constraints: if we have vector constraints $g(x)=0$, $J_g(x)$ is the constraint Jacobian matrix used in KKT conditions. They also appear when optimizing compositions of functions (via chain rule, below). In machine learning, the Jacobian of a network’s layers is used to propagate gradients backward (backpropagation is an application of chain rule on a composed function).

**Chain rule:** If $h(x) = f(g(x))$ is a composition $\mathbb{R}^n \xrightarrow{g} \mathbb{R}^m \xrightarrow{f} \mathbb{R}$, then by the chain rule the gradient is


$$
\nabla h(x) = J_g(x)^\top \, \nabla f(g(x))
$$




In coordinates, $\frac{\partial h}{\partial x_j} = \sum_{i=1}^m \frac{\partial f}{\partial y_i}(g(x)) \frac{\partial g_i}{\partial x_j}(x)$. This general rule shows that to compute the gradient of a nested function, we multiply the Jacobians going backward. For example, if $f(y)$ is scalar and $g(x)$ yields features, $\nabla_x f(g(x)) = J_g(x)^T \nabla f(y)|_{y=g(x)}$. This is exactly how backpropagation in neural networks works: the gradient w.r.t. inputs is obtained by propagating the output error gradient through each layer’s Jacobian (which are often simple elementwise operations or linear weight matrices). Thus, the chain rule is fundamental for efficient gradient calculations. In convex optimization, if $g(x)$ is an affine function and $f$ is convex and differentiable, then $h(x)=f(g(x))$ is convex and $\nabla h(x) = J_g(x)^T \nabla f(g(x))$ provides the needed gradient for algorithms.

**Hessian and second-order derivatives:** The Hessian of $f:\mathbb{R}^n\to\mathbb{R}$ is the $n \times n$ symmetric matrix of second partials, $\nabla^2 f(x)$, where $(\nabla^2 f(x))_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$. The Hessian matrix captures the quadratic curvature of $f$ around $x$. Specifically, the second-order Taylor expansion is

$$
f(x + h) \approx f(x)
+ \langle \nabla f(x), h \rangle
+ \tfrac{1}{2} \, h^\top (\nabla^2 f(x)) \, h
$$

The quadratic term $h^T \nabla^2 f(x) h / 2$ approximates how the gradient itself changes with $h$. 

**Properties of Hessian:**

- if $\nabla^2 f(x) \succeq 0$ (PSD) for all $x$ in a region, $f$ is convex on that region.

- If $\nabla^2 f(x) \succ 0$ (PD) for all $x$, $f$ is strictly convex (one minimizer). 

- On the other hand, if $\nabla^2 f(x)$ has a negative eigenvalue, $f$ is locally concave in that direction (not convex). Thus Hessian definiteness is a local convexity test. Many convex functions have constant Hessians (e.g. $f(x)=\frac{1}{2}x^TQx$ has $\nabla^2 f = Q$).

In optimization algorithms, Hessians are used in Newton’s method, which iteratively updates

$$
x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \, \nabla f(x_k)
$$


This uses the Hessian inverse as a linear approximation to the curvature, jumping to where the gradient would be zero if the quadratic model were exact. Newton’s method converges in a few iterations for quadratic objectives and generally superlinearly for well-behaved convex functions, but it requires solving linear systems involving $\nabla^2 f(x)$, which can be expensive for large $n$. Quasi-Newton methods (like BFGS) build approximations to the Hessian on the fly. Regardless, understanding Hessian is crucial for high-dimensional convex optimization: it tells us how sensitive the gradient is to changes in $x$, which directly affects step sizes and convergence.

**Example – quadratic function:** $f(x) = \frac{1}{2}x^TQx - b^T x$. Here $\nabla f(x) = Qx - b$ (linear), and $\nabla^2 f(x) = Q$. Solving $\nabla f=0$ yields $Qx=b$, so if $Q \succ 0$ the unique minimizer is $x^* = Q^{-1}b$. The Hessian being $Q \succ 0$ confirms convexity. If $Q$ has large eigenvalues, gradient $Qx - b$ changes rapidly in some directions (steep narrow valley); if some eigenvalues are tiny, gradient hardly changes in those directions (flat valley). This aligns with earlier discussions: condition number of $Q$ controls difficulty of minimizing $f$.

**Optimality conditions (unconstrained):** For an unconstrained differentiable problem $\min_x f(x)$, the first-order necessary condition is $\nabla f(x^) = 0$. If $f$ is convex, this is also sufficient: any $x$ with $\nabla f(x)=0$ is a global minimizer. If $f$ is twice differentiable, second-order conditions say: $\nabla f(x^)=0$ and $\nabla^2 f(x^*) \succeq 0$ for a local minimum. In convex problems the Hessian condition is automatically satisfied everywhere (since convex $f$ has PSD Hessian throughout), so checking $\nabla f(x)=0$ is enough.

**Gradient Lipschitz continuity:** A concept often used in convergence analysis is Lipschitz continuity of the gradient. If there exists $L$ such that $|\nabla f(x) - \nabla f(y)| \le L |x-y|$ for all $x,y$, we say the gradient is $L$-Lipschitz (or $f$ is $L$-smooth). $L$ is essentially an upper bound on the Hessian eigenvalues (for $\ell_2$ norm): $L \ge \lambda_{\max}(\nabla^2 f(x))$ for all $x$. Smoothness is important because it ensures gradient descent with step $\alpha = 1/L$ converges, and it gives a bound $f(x_{k+1}) \le f(x_k) - \frac{1}{2L}|\nabla f(x_k)|^2$ (so the function value decreases at least proportionally to the squared gradient norm). Many convex functions in optimization are $L$-smooth (e.g. quadratic forms with $\lambda_{\max}(Q)=L$). Smoothness together with strong convexity (defined shortly) yields linear convergence rates for gradient descent.

**Strong convexity:** A differentiable function $f$ is $\mu$-strongly convex if $f(y) \ge f(x) + \langle \nabla f(x), y-x \rangle + \frac{\mu}{2}|y-x|^2$ for all $x,y$. Equivalently, $f(x) - \frac{\mu}{2}|x|^2$ is convex, which implies $\nabla^2 f(x) \succeq \mu I$ (Hessian bounded below by $\mu$) when $f$ is twice differentiable. Strong convexity means $f$ has a quadratic curvature of at least $\mu$ – it grows at least as fast as a parabola. Strongly convex functions have unique minimizers (the bowl can’t flatten out). They also yield much faster convergence: for $\mu$-strongly convex and $L$-smooth $f$, gradient descent with $\alpha=1/L$ converges like $(1-\mu/L)^k$ (linear rate). Intuitively, the condition number $\kappa = L/\mu$ comes into play. Examples: the quadratic form above is strongly convex with $\mu = \lambda_{\min}(Q)$. Adding a small ridge term $\frac{\mu}{2}|x|^2$ to any convex $f$ makes it $\mu$-strongly convex and improves conditioning at the cost of bias.

In summary, the tools of calculus — gradients for direction of improvement, Hessians for curvature, Jacobians for constraint and composite mappings, and inequalities like Lipschitz bounds — all feed into understanding and solving convex optimization problems. The optimality conditions formalize the simple idea: at optimum, the gradient must vanish or be balanced by constraints. The next chapter will build on this by considering those constraints explicitly and introducing Lagrange multipliers and duality, giving deeper insight into optimality in constrained problems. -->