# Calculus Essentials: Gradients, Hessians, and Taylor Expansions

Calculus provides the foundation for optimization algorithms. Gradients indicate directions of steepest ascent or descent, Hessians encode curvature, and Taylor expansions give local approximations of functions. This chapter develops these concepts with an eye toward convex optimization and machine learning applications.

## Gradient and Directional Derivative

Let $f: \mathbb{R}^n \to \mathbb{R}$ be differentiable. The **gradient** of $f$ at $x$ is the vector

$$
\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}.
$$

Properties:

- Linear approximation: $f(x + h) \approx f(x) + \langle \nabla f(x), h \rangle$ for small $h$  
- Steepest ascent direction: $\nabla f(x)$  
- Steepest descent direction: $-\nabla f(x)$

The directional derivative of $f$ at $x$ in direction $u$ is

$$
D_u f(x) = \lim_{t \to 0} \frac{f(x + t u) - f(x)}{t} = \langle \nabla f(x), u \rangle.
$$

Optimization application: Gradient descent updates are

$$
x_{k+1} = x_k - \alpha \nabla f(x_k),
$$

where $\alpha$ is a step size, moving in the direction of steepest decrease.


## Hessian and Second-Order Directional Derivatives

The Hessian of $f$ at $x$ is the symmetric matrix of second partial derivatives:

$$
\nabla^2 f(x) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}.
$$

Properties:

- Symmetric: $\nabla^2 f(x) = (\nabla^2 f(x))^\top$  
- Quadratic form: $u^\top \nabla^2 f(x) u$ measures curvature along $u$  
- Positive semidefinite Hessian $\implies$ local convexity

Optimization application: Newton’s method updates

$$
x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)
$$

use curvature information to accelerate convergence, especially near minimizers.

## Taylor Expansions

The second-order Taylor expansion of $f$ around $x$ is

$$
f(x + h) \approx f(x) + \langle \nabla f(x), h \rangle + \frac{1}{2} h^\top \nabla^2 f(x) h.
$$

Properties:

- Linear term captures slope (first-order approximation)  
- Quadratic term captures curvature  
- Provides local quadratic models used in Newton and quasi-Newton methods.  


## Optimization Connections

- Gradients determine update directions in first-order methods  
- Hessians determine curvature, step scaling, and Newton updates.  
- Taylor expansions provide local approximations, guiding line search and trust-region methods.  
- Eigenvalues of the Hessian determine convexity, step sizes, and conditioning.
