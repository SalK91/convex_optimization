# Chapter 3: Multivariable Calculus for Optimization
Optimization seeks to find points that minimize or maximize a real-valued function. To analyze and solve such problems, we rely on tools from multivariable calculus — gradients, Jacobians, Hessians, and Taylor expansions — which describe how a function changes locally.

This chapter provides the differential calculus foundation essential for convex analysis and gradient-based learning algorithms.  It links geometric intuition and analytical tools that underlie optimization methods such as gradient descent, Newton’s method, and backpropagation.





## 3.1 Gradients and Directional Derivatives

Let $f:\mathbb{R}^n\to\mathbb{R}$. Differentiability at $x$ means there exists a linear map (the gradient) such that

$$
f(x+h)=f(x)+\nabla f(x)^\top h+o(\|h\|).
$$

Equivalently, for every direction $v\in\mathbb{R}^n$, the directional derivative exists and matches the gradient pairing:

$$
D_v f(x)=\lim_{t \rightarrow 0}\frac{f(x+tv)-f(x)}{t}=\nabla f(x)^\top v.
$$


This shows that $\nabla f(x)$ is the unique vector giving the best linear approximation of $f$ near $x$.  
Among all unit directions $u$,
$$
D_u f(x) = \langle \nabla f(x), u \rangle
$$
is maximized when $u$ aligns with $\nabla f(x)$ — the direction of steepest ascent.  
The opposite direction, $- \nabla f(x)$, gives the steepest descent.

> A level set of a differentiable function $f$ is
$$
L_c = \{\, x \in \mathbb{R}^n : f(x) = c \,\}.
$$

> At any point $x$ with $\nabla f(x) \ne 0$, the gradient $\nabla f(x)$ is orthogonal to the level set $L_{f(x)}$. Geometrically, level sets are like contour lines on a topographic map, and the gradient points perpendicular to them — in the direction of the steepest ascent of $f$. If we wish to decrease $f$, we move roughly in the opposite direction, $-\nabla f(x)$ (the direction of steepest descent). This geometric fact becomes central in constrained optimization:  at optimality, the gradient of the objective lies in the span of gradients of active constraints.


## 3.2  Jacobians 
When dealing with optimization or learning, functions rarely map one number to another. They often map many inputs to many outputs — for example, a neural network layer, a physical model, or a vector-valued transformation. The Jacobian matrix captures *how each output reacts to each input* — the complete local sensitivity of a system.

Derivative to Gradient:  For a scalar function $f : \mathbb{R}^n \to \mathbb{R}$, the derivative generalizes to the gradient:

$$
\nabla f(x) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1}(x) \\
\vdots \\
\frac{\partial f}{\partial x_n}(x)
\end{bmatrix}.
$$

- Each component $\frac{\partial f}{\partial x_i}$ tells how $f$ changes as we vary $x_i$ alone.  
- Collectively, $\nabla f(x)$ forms the vector of steepest ascent, pointing toward the direction of maximal increase.  
- The magnitude $\|\nabla f(x)\|$ measures how sharply $f$ rises.


From Gradient to Jacobian — Many Inputs, Many Outputs: Now let $F : \mathbb{R}^n \to \mathbb{R}^m$ be a vector-valued function:

$$
F(x) =
\begin{bmatrix}
F_1(x) \\[4pt]
F_2(x) \\[4pt]
\vdots \\[4pt]
F_m(x)
\end{bmatrix}.
$$

Each $F_i(x)$ is a scalar function with its own gradient $\nabla F_i(x)^\top$.  
Stacking these row vectors gives the Jacobian matrix:
$$
J_F(x) =
\begin{bmatrix}
\frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_n}
\end{bmatrix}.
$$


For $F : \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is the $m \times n$ matrix
$$
J_F(x) =
\begin{bmatrix}
\frac{\partial F_1}{\partial x_1} & \dots & \frac{\partial F_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial F_m}{\partial x_1} & \dots & \frac{\partial F_m}{\partial x_n}
\end{bmatrix}.
$$

It represents the best linear map approximating $F$ near $x$:
$$
F(x + h) \approx F(x) + J_F(x) \, h.
$$

Just as a tangent line approximates a scalar curve, the Jacobian defines the tangent linear map that approximates $F$ near $x$:
$$
F(x + h) \approx F(x) + J_F(x) \, h.
$$

- The small displacement $h$ in input space is transformed linearly into an output change $J_F(x)h$.  
- Thus, $J_F(x)$ acts like the *microscopic blueprint* of $F$ around $x$.  
- Locally, $F$ behaves like a matrix transformation — stretching, rotating, or skewing space.

 | Part of $J_F(x)$ | Interpretation |
|-------------------|----------------|
| Row $i$ | Gradient of the $i$-th output $F_i(x)$ — how that output changes with each input variable. |
| Column $j$ | Sensitivity of all outputs to input $x_j$ — how changing $x_j$ influences the entire output vector. |
| Determinant (if $m=n$) | Local volume scaling — how much $F$ expands or compresses space near $x$. |
| Rank of $J_F(x)$ | Local dimensionality of the image of $F$ — tells if directions are lost or preserved. |


## 3.3 The Hessian and Curvature

If $f : \mathbb{R}^n \to \mathbb{R}$ is twice differentiable, its Hessian is

$$
\nabla^2 f(x) =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}.
$$


The Hessian encodes curvature information:
- $\nabla^2 f(x) \succeq 0$ (positive semidefinite) ⟹ $f$ is convex near $x$.  
- $\nabla^2 f(x) \succ 0$ ⟹ $f$ is strictly convex.  
- Negative eigenvalues ⟹ directions of local decrease.

 
Example – quadratic function: $f(x) = \frac{1}{2}x^TQx - b^T x$. Here $\nabla f(x) = Qx - b$ (linear), and $\nabla^2 f(x) = Q$. Solving $\nabla f=0$ yields $Qx=b$, so if $Q \succ 0$ the unique minimizer is $x^* = Q^{-1}b$. The Hessian being $Q \succ 0$ confirms convexity. If $Q$ has large eigenvalues, gradient $Qx - b$ changes rapidly in some directions (steep narrow valley); if some eigenvalues are tiny, gradient hardly changes in those directions (flat valley). This aligns with earlier discussions: condition number of $Q$ controls difficulty of minimizing $f$.

> Eigenvalues of the Hessian describe curvature along principal directions. Large eigenvalues correspond to steep curvature; small ones correspond to flat regions. Understanding this curvature is essential in designing stable optimization algorithms.

 
## 3.4 Taylor approximation

For differentiable $f$, we have the first-order Taylor expansion around $x$:
$$
f(x + d) \approx f(x) + \nabla f(x)^\top d~.
$$

The gradient gives the best local linear approximation, predicting how $f$ changes for a small move $d$. This is the foundation of first-order optimization methods such as gradient descent.

 
If f is twice differentiable, we have the second-order expansion
$$
f(x + d) \approx f(x)
+ \nabla f(x)^\top d
+ \frac{1}{2} d^\top \nabla^2 f(x) d~.
$$

If $\nabla^2 f(x)$ is positive semidefinite, the quadratic term is always $\ge 0$. Locally, $x$ is in a “bowl”. If $\nabla^2 f(x)$ is indefinite, the landscape can curve up in some directions and down in others — typical of saddle points.

## 3.5 Convexity and the Hessian

A twice-differentiable function $f$ is convex on a convex set if and only if

$$
\nabla^2 f(x) \succeq 0 \quad \forall x \text{ in the domain.}
$$

The Hessian describes how the gradient changes.

## 3.6 First and Second-Order Optimality Conditions

Suppose we want to solve an unconstrained optimization problem:
$$
\min_x f(x).
$$

A point $x^\star$ is called a critical point if
$$
\nabla f(x^\star) = 0.
$$


### First-Order Necessary Condition

If $f$ is differentiable and $x^\star$ is a local minimizer, then necessarily
$$
\nabla f(x^\star) = 0.
$$

Intuitively, this means the slope in every direction must vanish — there is no infinitesimal move that decreases $f$ further.


### Second-Order Necessary Condition

If $f$ is twice differentiable and $x^\star$ is a local minimizer, then
$$
\nabla f(x^\star) = 0,
\quad
\nabla^2 f(x^\star) \succeq 0,
$$

meaning the Hessian is positive semidefinite (PSD). The function curves upward (or flat) in all local directions.


### Second-Order Sufficient Condition

If
$$
\nabla f(x^\star) = 0,
\quad
\nabla^2 f(x^\star) \succ 0,
$$
i.e. the Hessian is positive definite (PD),  
then $x^\star$ is a strict local minimizer — the point lies at the bottom of a strictly convex bowl.


The gradient gives the best local linear approximation, predicting how f changes for a small move d.  This is the foundation of first-order optimization methods such as gradient descent. If $\nabla f(x^\star)$ has both positive and negative eigenvalues, $x^\star$ is a saddle point — neither a minimum nor a maximum.


Convexity makes everything simpler.If $f$ is convex, then *any* point $x^\star$ satisfying $\nabla f(x^\star) = 0$  is not only a local minimizer — it is a global minimizer.

## 3.7 Lipschitz Continuity and Smoothness

A function $f$ has a Lipschitz continuous gradient with constant $L > 0$ if
$$
\|\nabla f(x) - \nabla f(y)\| \le L \|x - y\|, \quad \forall x,y.
$$

Such a function is called L-smooth. This condition bounds how quickly the gradient can change, ensuring the function is not excessively curved.

This property implies the Descent Lemma:
$$
f(y) \le f(x) + \nabla f(x)^\top (y - x) + \tfrac{L}{2} \|y - x\|^2.
$$

Smoothness bounds how fast $f$ can curve.  
In gradient descent, choosing a step size $\eta \le 1/L$ guarantees convergence for convex functions.

> In ML training, $L$ controls how “aggressive” the learning rate can be — smoother losses allow larger steps.

## 3.8 Strong Convexity — Functions with Guaranteed Curvature

A differentiable function $f$ is said to be $\mu$-strongly convex if for some $\mu > 0$,
$$
f(y) \ge f(x) + \langle \nabla f(x),\, y - x \rangle + \frac{\mu}{2}\|y - x\|^2,
\quad \forall\, x, y.
$$

This inequality means f always lies above its tangent plane by at least a quadratic term of curvature μ.  
Strong convexity ensures a minimum curvature: f grows at least as fast as a parabola away from its minimizer.


## 3.9 Subgradients and Nonsmooth Extensions

Many useful convex functions are nonsmooth — e.g., hinge loss, $\ell_1$ norm, ReLU.  
They lack a gradient at certain points but admit a subgradient.

A vector $g$ is a subgradient of $f$ at $x$ if
$$
f(y) \ge f(x) + g^\top (y - x), \quad \forall y.
$$

The set of all such vectors is the subdifferential $\partial f(x)$.  

Examples:
- $f(x) = \|x\|_1$ has
  $$
  (\partial f(x))_i =
  \begin{cases}
  \operatorname{sign}(x_i), & x_i \ne 0, \\
  [-1, 1], & x_i = 0.
  \end{cases}
  $$
- For $f(x) = \max_i x_i$, any unit vector supported on the active index is a subgradient.

Subgradients generalize the gradient concept, allowing optimization even when derivatives do not exist.  
They are the backbone of nonsmooth convex optimization and proximal methods.  In machine learning, they make it possible to minimize losses such as the hinge or absolute deviation, where gradients are undefined at corners.

