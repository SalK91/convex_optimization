# Smoothness and Strong Convexity

Smoothness and strong convexity are fundamental concepts in optimization that describe the behavior of gradients and curvature of functions. They determine step sizes, convergence rates, and preconditioning strategies in both first-order and second-order methods. This chapter develops these concepts with explicit links to convex optimization algorithms.

## Lipschitz Continuity and Smoothness

A differentiable function $f: \mathbb{R}^n \to \mathbb{R}$ is **$L$-smooth** (or has Lipschitz continuous gradient) if there exists $L > 0$ such that

$$
\|\nabla f(x) - \nabla f(y)\| \le L \|x - y\| \quad \forall x, y \in \mathbb{R}^n.
$$

Equivalent quadratic upper bound:

$$
f(y) \le f(x) + \langle \nabla f(x), y - x \rangle + \frac{L}{2} \|y - x\|^2.
$$

Interpretation:

- $L$ controls how rapidly the gradient can change  
- Larger $L$ implies steep curvature and smaller safe step sizes

Optimization application: For gradient descent, the step size $\alpha$ must satisfy $\alpha \le 1/L$ to guarantee decrease of $f$.

Example: $f(x) = \frac{1}{2} x^\top A x$ with $A \succeq 0$ has $L = \lambda_{\max}(A)$.

## Strong Convexity

A differentiable function $f$ is $\mu$-strongly convex if there exists $\mu > 0$ such that

$$
f(y) \ge f(x) + \langle \nabla f(x), y - x \rangle + \frac{\mu}{2} \|y - x\|^2 \quad \forall x, y \in \mathbb{R}^n.
$$

Equivalent condition: The Hessian satisfies $\nabla^2 f(x) \succeq \mu I$ for all $x$.

Interpretation:

- $\mu$ provides a lower bound on curvature  
- Strong convexity ensures a unique minimizer  
- Condition number: $\kappa = L/\mu$ measures problem conditioning

Optimization application: Gradient descent on a $\mu$-strongly convex, $L$-smooth function satisfies linear convergence:

$$
\|x_k - x^*\| \le \left(1 - \frac{\mu}{L}\right)^k \|x_0 - x^*\|.
$$

Example: $f(x) = x_1^2 + 2x_2^2$ is $1$-strongly convex along $x_1$ and $2$-strongly convex along $x_2$, with $L = 2$ and $\kappa = 2$.

## Geometric Intuition

- Smoothness bounds how steeply gradients can change; it controls the “flatness” of the function landscape  
- Strong convexity** ensures the function curves upward everywhere, providing a “bowl shape” that guarantees a unique minimizer  
- The ratio $L/\mu$ (condition number) indicates how stretched the bowl is; larger $\kappa$ implies elongated level sets and slower gradient descent convergence

## Connections to Optimization Algorithms

- Step-size selection: $\alpha \le 1/L$ ensures monotone decrease of $f$  
- Convergence rates: linear convergence in strongly convex functions, sublinear in general convex functions  
- Preconditioning: transforming variables to reduce $\kappa = L/\mu$ accelerates convergence  
- Applicable to gradient descent, accelerated gradient methods (Nesterov), and Newton-type methods
