# Chapter 3: Multivariable Calculus for Optimization

Optimization problems are ultimately questions about how a function changes when we move in different directions. To understand this behavior, we rely on multivariable calculus. Concepts such as gradients, Jacobians, Hessians, and Taylor expansions describe how a real-valued function behaves locally and how its value varies as we adjust its inputs.

These tools form the analytical backbone of modern optimization. Gradients determine descent directions and guide first-order algorithms such as gradient descent and stochastic gradient methods. Hessians quantify curvature and enable second-order methods like Newton’s method, which adapt their steps to the shape of the objective. Jacobians and chain rules underpin backpropagation in neural networks, linking calculus to large-scale machine learning practice.

This chapter develops the differential calculus needed for convex analysis and for understanding why many optimization algorithms work. We emphasize geometric intuition, how functions curve, how directions interact, and how local approximations guide global behavior, while providing the formal tools required to analyze convergence and stability in later chapters.


## Gradients and Directional Derivatives
Let $f : \mathbb{R}^n \to \mathbb{R}$. The function is differentiable at a point $x$ if there exists a vector $\nabla f(x)$ such that
$$
f(x + h)
=
f(x) + \nabla f(x)^\top h + o(\|h\|),
$$
meaning that the linear function $h \mapsto \nabla f(x)^\top h$ provides the best local approximation to $f$ near $x$. The gradient is the unique vector with this property.

A closely related concept is the directional derivative. For any direction $v \in \mathbb{R}^n$, the directional derivative of $f$ at $x$ in the direction $v$ is
$$
D_v f(x)
=
\lim_{t \to 0} \frac{f(x + tv) - f(x)}{t}.
$$
If $f$ is differentiable, then
$$
D_v f(x) = \nabla f(x)^\top v.
$$
Thus, the gradient encodes all directional derivatives simultaneously: its inner product with a direction $v$ tells us how rapidly $f$ increases when we move infinitesimally along $v$.

This immediately yields an important geometric fact. Among all unit directions $u$,
$$
D_u f(x) = \langle \nabla f(x), u \rangle
$$
is maximized when $u$ points in the direction of $\nabla f(x)$, the direction of steepest ascent. The steepest descent direction is therefore $-\nabla f(x)$, which motivates gradient-descent algorithms for minimizing functions.


> For any real number $c$, the level set of $f$ is 
$$
L_c = \{\, x \in \mathbb{R}^n : f(x) = c \,\}.
$$

> At any point $x$ with $\nabla f(x) \ne 0$, the gradient $\nabla f(x)$ is orthogonal to the level set $L_{f(x)}$. Geometrically, level sets are like contour lines on a topographic map, and the gradient points perpendicular to them — in the direction of the steepest ascent of $f$. If we wish to decrease $f$, we move roughly in the opposite direction, $-\nabla f(x)$ (the direction of steepest descent). This geometric fact becomes central in constrained optimization:  at optimality, the gradient of the objective lies in the span of gradients of active constraints.


##  Jacobians 
In optimization and machine learning, functions often map many inputs to many outputs for example, neural network layers, physical simulators, and vector-valued transformations. To understand how such functions change locally, we use the Jacobian matrix, which captures how each output responds to each input.

### From derivative to gradient

For a scalar function $ f : \mathbb{R}^n \to \mathbb{R} $, differentiability means that near any point $ x $,
$$
f(x + h) \approx f(x) + \nabla f(x)^\top h.
$$
The gradient vector
$$
\nabla f(x) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1}(x) \\
\vdots \\
\frac{\partial f}{\partial x_n}(x)
\end{bmatrix}
$$
collects all partial derivatives. Each component measures how sensitive $f$ is to changes in a single coordinate. Together, the gradient points in the direction of steepest increase, and its norm indicates how rapidly the function rises.

### From gradient to Jacobian

Now consider a vector-valued function $F : \mathbb{R}^n \to \mathbb{R}^m$,
$$
F(x) =
\begin{bmatrix}
F_1(x) \\
\vdots \\
F_m(x)
\end{bmatrix}.
$$
Each output $F_i$ has its own gradient. Stacking these row vectors yields the Jacobian matrix:
$$
J_F(x) =
\begin{bmatrix}
\frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_n}
\end{bmatrix}.
$$

The Jacobian provides the best linear approximation of $F$ near $x$:
$$
F(x + h) \approx F(x) + J_F(x)\, h.
$$
Thus, locally, the nonlinear map $F$ behaves like the linear map $h \mapsto J_F(x)h$. A small displacement $h$ in input space is transformed into an output change governed by the Jacobian.

### Interpreting the Jacobian

| Component of $J_F(x)$ | Meaning |
|--------------------------|---------|
| Row $i$               | Gradient of output $F_i(x)$: how the $i$-th output changes with each input variable. |
| Column $j$            | Sensitivity of all outputs to $x_j$: how varying input $x_j$ affects the entire output vector. |
| Determinant (when $m=n$) | Local volume scaling: how $F$ expands or compresses space near $x$. |
| Rank                    | Local dimension of the image: whether any input directions are lost or collapsed. |

The Jacobian is therefore a compact representation of local sensitivity. In optimization, Jacobians appear in gradient-based methods, backpropagation, implicit differentiation, and the analysis of constraints and dynamics.



## The Hessian and Curvature

For a twice–differentiable function $ f : \mathbb{R}^n \to \mathbb{R} $, the Hessian matrix collects all second-order partial derivatives:
$$
\nabla^{2} f(x) \;=\;
\begin{bmatrix}
\frac{\partial^{2} f}{\partial x_{1}^{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{1}\partial x_{n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial^{2} f}{\partial x_{n}\partial x_{1}} & \cdots & \frac{\partial^{2} f}{\partial x_{n}^{2}}
\end{bmatrix}.
$$

The Hessian describes the **local curvature** of the function. While the gradient indicates the direction of steepest change, the Hessian tells us *how that directional change itself varies*—whether the surface curves upward, curves downward, or remains nearly flat.

### Curvature and positive definiteness

The eigenvalues of the Hessian determine its geometric behavior:

- If $ \nabla^{2}f(x) \succeq 0 $ (all eigenvalues nonnegative), the function is locally convex near $x$.  
- If $ \nabla^{2}f(x) \succ 0 $, the surface curves upward in all directions, guaranteeing local (and for convex functions, global) uniqueness of the minimizer.  
- If the Hessian has both positive and negative eigenvalues, the point is a saddle: some directions curve up, others curve down.

Thus, curvature is directly encoded in the spectrum of the Hessian. Large eigenvalues correspond to steep curvature; small eigenvalues correspond to gently sloping or flat regions.

### Example: Quadratic functions

Consider the quadratic function
$$
f(x) = \tfrac{1}{2} x^\top Q x - b^\top x,
$$
where $Q$ is symmetric. The gradient and Hessian are
$$
\nabla f(x) = Qx - b, \qquad \nabla^2 f(x) = Q.
$$
Setting the gradient to zero gives the stationary point
$$
Qx = b.
$$
If $Q \succ 0$, the solution
$$
x^* = Q^{-1} b
$$
is the unique minimizer. The Hessian $Q$ being positive definite confirms strict convexity.

The eigenvalues of $Q$ also explain the difficulty of minimizing $f$:

- Large eigenvalues produce very steep, narrow directions—optimization methods must take small steps.  
- Small eigenvalues produce flat directions—progress is slow, especially for gradient descent.  

The ratio of largest to smallest eigenvalue, the **condition number**, governs the convergence speed of first-order methods on quadratic problems. Poor conditioning (large condition number) leads to zig-zagging iterates and slow progress.

### Why the Hessian matters in optimization

The Hessian provides second-order information that strongly influences algorithm behavior:

- Newton’s method uses the Hessian to rescale directions, effectively “whitening’’ curvature and often converging rapidly.  
- Trust-region and quasi-Newton methods approximate Hessian structure to stabilize steps.  
- In convex optimization, positive semidefiniteness of the Hessian is a fundamental characterization of convexity.

Understanding the Hessian therefore helps us understand the geometry of an objective, predict algorithm performance, and design methods that behave reliably on challenging landscapes.


    
## Taylor approximation

Taylor expansions provide local approximations of a function using its derivatives. These approximations form the basis of nearly all gradient-based optimization methods.

### First-order approximation

If $f$ is differentiable at $x$, then for small steps $d$,
$$
f(x + d)
\approx
f(x) + \nabla f(x)^\top d.
$$
The gradient gives the best linear model of the function near $x$. This linear approximation is the foundation of first-order methods such as gradient descent, which choose directions based on how this model predicts the function will change.

### Second-order approximation

If $f$ is twice differentiable, we can include curvature information:
$$
f(x + d)
\approx
f(x)
+ \nabla f(x)^\top d
+ \tfrac{1}{2} d^\top \nabla^2 f(x)\, d.
$$
The quadratic term measures how the gradient itself changes with direction. The behavior of this term depends on the Hessian:

- If $ \nabla^2 f(x) \succeq 0 $, the quadratic term is nonnegative and the function curves upward—locally bowl-shaped.
- If the Hessian has both positive and negative eigenvalues, the function bends up in some directions and down in others—characteristic of saddle points.

### Role in optimization algorithms

Second-order Taylor models are the basis of Newton-type methods. Newton’s method chooses $d$ by approximately minimizing the quadratic model,
$$
d \approx - \left(\nabla^2 f(x)\right)^{-1} \nabla f(x),
$$
which balances descent direction and local curvature. Trust-region and quasi-Newton methods also rely on this quadratic approximation, modifying or regularizing it to ensure stable progress.

Thus, Taylor expansions connect a function’s derivatives to practical optimization steps, bridging geometry and algorithm design.

## Smoothness and Strong Convexity

In optimization, the behavior of a function’s curvature strongly influences how algorithms perform. Two fundamental properties Lipschitz smoothness and strong convexity describe how rapidly the gradient can change and how much curvature the function must have.

### Lipschitz continuous gradients (L-smoothness)

A differentiable function $ f $ has an $L$-Lipschitz continuous gradient if
$$
\|\nabla f(x) - \nabla f(y)\| \le L \|x - y\| \qquad \forall x, y.
$$
This condition limits how quickly the gradient can change. Intuitively, an $L$-smooth function cannot have sharp bends or extremely steep local curvature. A key consequence is the Descent Lemma:
$$
f(y)
\le
f(x)
+
\nabla f(x)^\top (y - x)
+
\frac{L}{2}\|y - x\|^2.
$$
This inequality states that every $L$-smooth function is upper-bounded by a quadratic model derived from its gradient. It provides a guaranteed estimate of how much the function can increase when we take a step.

In gradient descent, smoothness directly determines a safe step size: choosing
$$
\eta \le \frac{1}{L}
$$
ensures that each update decreases the function value for convex objectives. In machine learning, the constant $L$ effectively controls how large the learning rate can be before training becomes unstable.

### Strong convexity

A differentiable function $ f $ is $ \mu $-strongly convex if, for some $ \mu > 0 $,
$$
f(y)
\ge
f(x)
+
\langle \nabla f(x),\, y - x \rangle
+
\frac{\mu}{2}\|y - x\|^2
\qquad \forall x, y.
$$
This condition guarantees that $f$ has at least $\mu$ amount of curvature everywhere. Geometrically, the function always lies above its tangent plane by a quadratic bowl, growing at least as fast as a parabola away from its minimizer.

Strong convexity has major optimization implications:

- The minimizer is unique.  
- Gradient descent converges linearly with step size $\eta \le 1/L$.  
- The ratio $L / \mu$ (the condition number) dictates convergence speed.

### Curvature in both directions

Together, smoothness and strong convexity bound the curvature of $f$:
$$
\mu I \;\preceq\; \nabla^2 f(x) \;\preceq\; L I.
$$
Smoothness prevents the curvature from being too large, while strong convexity prevents it from being too small. Many convergence guarantees in optimization depend on this pair of inequalities.

These concepts, imiting curvature from above via $L$ and from below via $\mu$, form the foundation for analyzing the performance of first-order algorithms and understanding how learning rates, conditioning, and geometry interact.


## Mental map
``` text
                Multivariable Calculus for Optimization
        How objectives change, how curvature shapes algorithms
                              │
                              ▼
                 Local change of a scalar function f(x)
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Differentiability & First-Order Model                     │
     │ f(x+h) = f(x) + ∇f(x)ᵀh + o(‖h‖)                          │
     │ - ∇f(x): best linear approximation                        │
     │ - Directional derivative: D_v f(x) = ∇f(x)ᵀv              │
     │ - Steepest descent: move along -∇f(x)                     │
     └───────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ Geometry of Level Sets                                      │
     │ L_c = {x : f(x)=c}                                          │
     │ - If ∇f(x) ≠ 0, then ∇f(x) ⟂ level set at x                 │
     │ - Connects to constrained optimality (later: KKT)           │
     └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ Vector-Valued Maps & Jacobians                              │
     │ F: ℝⁿ → ℝᵐ                                                  │
     │ - Jacobian J_F(x) stacks gradients of outputs               │
     │ - Linearization: F(x+h) ≈ F(x) + J_F(x) h                   │
     │ - Chain rule foundation for backprop / sensitivity analysis │
     └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Second-Order Structure: Hessian & Curvature               │
     │ ∇²f(x): matrix of second partials                         │
     │ - Curvature along v: vᵀ∇²f(x)v                            │
     │ - Eigenvalues quantify steep/flat directions              │
     │ - PSD/PD Hessian ties directly to convexity (Ch.5)        │
     └───────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Taylor Models → Algorithm Design                          │
     │ First-order:  f(x+d) ≈ f(x) + ∇f(x)ᵀd                     │
     │ Second-order: f(x+d) ≈ f(x) + ∇f(x)ᵀd + ½ dᵀ∇²f(x)d       │
     │ - Gradient descent uses the linear model                  │
     │ - Newton uses the quadratic model: d ≈ -(∇²f)^{-1}∇f      │
     │ - Trust-region / quasi-Newton approximate curvature       │
     └───────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌────────────────────────────────────────────────────────────────┐
     │ Global Control of Local Behavior: Smoothness & Strong Convexity│
     │ L-smooth: ‖∇f(x)-∇f(y)‖ ≤ L‖x-y‖                               │
     │ - Descent Lemma gives a quadratic upper bound                  │
     │ - Sets safe step size: η ≤ 1/L (for convex objectives)         │
     │ μ-strongly convex: f lies above tangents by (μ/2)‖y-x‖²        │
     │ - Unique minimizer, linear convergence of gradient descent     │
     │ Combined curvature bounds: μI ⪯ ∇²f(x) ⪯ LI                   │
     │ - Condition number κ = L/μ governs difficulty                  │
     └────────────────────────────────────────────────────────────────┘
```