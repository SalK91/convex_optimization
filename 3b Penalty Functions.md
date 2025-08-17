# Convex Optimization Notes: Penalty Function Approximation

## Penalty Function Approximation
We solve:

$\min \; \phi(r_1) + \cdots + \phi(r_m) \quad \text{subject to} \quad r = Ax - b$

where:
- $A \in \mathbb{R}^{m \times n}$
- $\phi : \mathbb{R} \to \mathbb{R}$ is a **convex penalty function**

The choice of $\phi$ determines how residuals are penalized.



## Common Penalty Functions

### 1. Quadratic (Least Squares)
$\phi(u) = u^2$

- Strongly convex, smooth.  
- Penalizes large residuals heavily.  
- Equivalent to **Gaussian noise model** in statistics.  
- Leads to **unique minimizer**.  


### 2. Absolute Value (Least Absolute Deviations)
$\phi(u) = |u|$

- Convex but **nonsmooth** at $u=0$ (subgradient methods needed).  
- Robust to outliers compared to quadratic.  
- Equivalent to **Laplace noise model** in statistics.  

#### Why does it lead to sparsity?
- The sharp corner at $u=0$ makes it favorable for optimization to set many residuals (or coefficients) exactly to zero.  
- In contrast, quadratic penalties ($u^2$) only shrink values toward zero but rarely make them exactly zero.  
- Geometric intuition: the $\ell_1$ ball has corners aligned with coordinate axes → solutions land on axes → sparse.  
- Statistical interpretation: corresponds to a **Laplace prior**, which induces sparsity, whereas $\ell_2$ corresponds to a Gaussian prior (no sparsity).  

👉 This property is the foundation of **Lasso regression** and many **compressed sensing** methods.  


### 3. Deadzone-Linear
$\phi(u) = \max \{ 0, |u| - \alpha \}$, where $\alpha > 0$

- Ignores small deviations ($|u| < \alpha$).  
- Linear growth outside the “deadzone.”  
- Used in **support vector regression (SVR)** with $\epsilon$-insensitive loss.  
- Convex, but not strictly convex → possibly multiple minimizers.  



### 4. Log-Barrier
$\phi(u) =
\begin{cases}
-\alpha^2 \log \left(1 - (u/\alpha)^2 \right), & |u| < \alpha \\
\infty, & \text{otherwise}
\end{cases}$

- Smooth, convex inside domain $|u| < \alpha$.  
- Grows steeply as $|u| \to \alpha$.  
- Effectively enforces **constraint $|u| < \alpha$**.  


## Histograms of Residuals (Effect of Penalty Choice)

For $A \in \mathbb{R}^{100 \times 30}$, the residual distribution $r$ depends on $\phi$:

- **Quadratic ($u^2$):** residuals spread out (Gaussian-like).  
- **Absolute value ($|u|$):** sharper peak at 0, heavier tails (Laplace-like).  
- **Deadzone:** many residuals exactly at 0 (ignored region).  
- **Log-barrier:** residuals concentrate away from the boundary $|u| = 1$.  

👉 **Takeaway:** Choice of $\phi$ directly shapes residual distribution.



## Huber Penalty Function
The **Huber penalty** combines quadratic and linear growth:

$\phi_{\text{huber}}(u) =
\begin{cases}
u^2, & |u| \leq M \\
2M|u| - M^2, & |u| > M
\end{cases}$

### Properties
- Quadratic near 0 ($|u| \leq M$) → efficient for small noise.  
- Linear for large $|u|$ → robust to outliers.  
- Smooth, convex.  
- Interpolates between **least squares** and **least absolute deviations**.  

👉 Called a **robust penalty**, widely used in robust regression.


## Summary: Choosing a Penalty Function

- **Quadratic:** efficient, but sensitive to outliers.  
- **Absolute value:** robust, but nonsmooth.  
- **Deadzone:** ignores small errors, good for sparse modeling (e.g., SVR).  
- **Log-barrier:** enforces domain constraints smoothly.  
- **Huber:** best of both worlds → quadratic for small residuals, linear for large ones.  

