# Function Properties for Optimization: Strong Convexity, Smoothness, and Conditioning


## Strong Convexity

A function $f$ is **$\mu$-strongly convex** if

$$
f(y) \ge f(x) + \nabla f(x)^\top (y-x) + \frac{\mu}{2}\|y-x\|^2.
$$

If $f$ is twice differentiable, this is equivalent to

$$
\nabla^2 f(x) \succeq \mu I \quad \text{for all } x.
$$

- Guarantees a **unique minimizer**.  
- Gradient-based methods achieve **linear convergence**.  
- Prevents flat regions where optimization would stall.  

### Why Convergence May Be Slow Without Strong Convexity
- If $f$ is convex but not strongly convex, it can have **flat regions** (zero curvature).  
- Gradients may be very small in these directions → **gradient steps shrink**, and convergence becomes **sublinear**:  

$$
f(x_t) - f(x^\star) = O\left(\frac{1}{t}\right).
$$

- Example: $f(x) = x^4$ is convex but not strongly convex near $x=0$. Gradient descent steps become tiny near the minimum → slow convergence.  

- Contrast: $f(x) = x^2$ is strongly convex ($\mu=2$) → linear convergence.  

### Examples

1. **Quadratic function:**  
   $f(x) = x^2$ → $\mu=2$, strongly convex → fast convergence.  

2. **Quartic function:**  
   $f(x) = x^4$ → convex but not strongly convex near $0$ → slow convergence.  

3. **Ridge Regression (L2 Regularization):**  
   $$
   f(w) = \|Xw - y\|^2 + \lambda \|w\|^2, \quad \lambda > 0.
   $$
    - The first term $\|Xw - y\|^2$ is convex.  
    - The L2 term $\lambda \|w\|^2$ is strongly convex ($\nabla^2 (\lambda\|w\|^2) = 2\lambda I \succeq \lambda I$).  
    - **Adding the L2 penalty makes the entire objective strongly convex** with $\mu = \lambda$.  
    - **Implications:**  
        - Unique solution:  
        $$
        w^\star = (X^\top X + \lambda I)^{-1} X^\top y
        $$  
        even if $X^\top X$ is singular or ill-conditioned.  
        - Stable optimization: gradient-based methods converge linearly.  
        - Prevents overfitting by controlling the size of weights.  

 

## Smoothness (L-smoothness)

A function $f$ is **$L$-smooth** if

$$
\|\nabla f(x) - \nabla f(y)\| \le L \|x-y\|.
$$

If twice differentiable:

$$
\nabla^2 f(x) \preceq L I \quad \text{for all } x.
$$

- Limits how steep $f$ can be.  
- Ensures gradients change gradually → **stable gradient steps**.  
- Guarantees safe step sizes: $\alpha < 1/L$ for gradient descent.  

### Why Smoothness Matters for Convergence
- Without smoothness, the gradient can change abruptly.  
- A large gradient could lead to **overshooting**, oscillation, or divergence.  
- Smoothness ensures **predictable, stable progress** along the gradient.  

**Examples:**  
- Quadratic $f(x) = \frac{1}{2}x^\top Qx$: $L = \lambda_{\max}(Q)$.  
- Logistic regression loss: smooth with $L$ depending on $\|X\|^2$.  
- Non-smooth case: $f(x) = |x|$ → gradient jumps at $x=0$, cannot guarantee smooth progress → need subgradient methods.  


## Condition Number

The **condition number** is defined as

$$
\kappa = \frac{L}{\mu}.
$$

- Measures how “stretched” the optimization landscape is.  
- High $\kappa$ → narrow, elongated valleys → gradient descent zig-zags, converges slowly.  
- Low $\kappa$ → round bowl → fast convergence.  

**Examples:**  
- $Q=I$: $\mu=L=1$, $\kappa=1$ → fastest convergence.  
- $Q=\text{diag}(1,1000)$: $\mu=1$, $L=1000$, $\kappa=1000$ → ill-conditioned, very slow.  
- In ML, normalization (batch norm, feature scaling, whitening) reduces $\kappa$, improving training speed.  

 
## Use Cases and Benefits

### Strong Convexity
- Unique solution (ridge regression).  
- Linear convergence of gradient-based methods.  
- Stabilizes optimization by avoiding flatness.  

### Smoothness
- Ensures safe and predictable step sizes.  
- Avoids overshooting or divergence.  
- Justifies constant learning rates for many ML losses.  

### Condition Number
- Predicts convergence speed.  
- Guides preprocessing: scaling, normalization, whitening.  
- Central in designing adaptive optimizers and preconditioning methods.  

---

## Convergence Rates of First-Order Methods

| Function Property            | Gradient Descent Rate                         | Accelerated Gradient (Nesterov) | Subgradient Method Rate          |
|-------------------------------|-----------------------------------------------|---------------------------------|---------------------------------|
| Convex (not strongly convex) | $O(1/t)$                                      | $O(1/t^2)$                      | $O(1/\sqrt{t})$                 |
| $\mu$-Strongly Convex        | Linear: $O\big((1-\eta\mu)^t\big)$            | Linear: faster than GD          | $O(\log t / t)$                 |
| Condition Number $\kappa$    | Iterations $\sim O(\kappa \log(1/\epsilon))$   | Iterations $\sim O(\sqrt{\kappa}\log(1/\epsilon))$ | – |

 

## Intuitive Summary
- **Strong convexity**: bowl is always curved enough → unique and fast convergence.  
- **Smoothness**: bowl is not too steep → safe steps, avoids overshooting.  
- **Condition number**: how round vs stretched the bowl is → dictates optimization difficulty.  
- Without strong convexity → flat regions → slow sublinear convergence.  
- Without smoothness → steep gradient changes → possible divergence or oscillations.
