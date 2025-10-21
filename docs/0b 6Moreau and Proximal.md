# Moreau Envelopes & Proximal Operators

 Moreau envelopes and proximal operators are central concepts in **nonsmooth convex optimization**. They allow us to:

- Smooth nonsmooth functions for easier optimization  
- Define **proximal updates** that generalize gradient steps  
- Connect primal and dual problems via **Fenchel conjugates**  

Intuition: A Moreau envelope is a **smoothed version** of a convex function that approximates it while retaining its convexity. The proximal operator finds the **point closest to a given input while balancing the original function**, effectively performing a “soft minimization.”

 

## Definitions and Formal Statements

### Moreau Envelope

For a proper convex function $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ and parameter $\lambda > 0$, the **Moreau envelope** $f_\lambda$ is defined as:

\[
f_\lambda(x) = \min_{y \in \mathbb{R}^n} \left\{ f(y) + \frac{1}{2\lambda} \|y - x\|_2^2 \right\}
\]

- $f_\lambda(x)$ is **smooth**, even if $f$ is nonsmooth.  
- It provides a **lower bound on $f$**, approaching $f$ as $\lambda \to 0$.  

### Proximal Operator

The **proximal operator** of $f$ is defined as:

\[
\text{prox}_{\lambda f}(x) = \arg\min_{y \in \mathbb{R}^n} \left\{ f(y) + \frac{1}{2\lambda} \|y - x\|_2^2 \right\}
\]

- The proximal operator returns the point $y$ that **balances minimizing $f$ with staying close to $x$**.  
- Relation: $f_\lambda(x) = f(\text{prox}_{\lambda f}(x)) + \frac{1}{2\lambda} \| \text{prox}_{\lambda f}(x) - x \|_2^2$

 
## Step-by-Step Analysis / How to Use

1. Identify the convex function $f$ and choose a parameter $\lambda > 0$.  
2. Compute the **proximal operator**:  

\[
y^* = \text{prox}_{\lambda f}(x) = \arg\min_y \left\{ f(y) + \frac{1}{2\lambda} \|y - x\|_2^2 \right\}
\]

3. Compute the **Moreau envelope** if a smooth approximation is desired:  

\[
f_\lambda(x) = f(y^*) + \frac{1}{2\lambda} \|y^* - x\|_2^2
\]

4. Use in optimization algorithms:  
   - **Proximal gradient descent:** $x_{k+1} = \text{prox}_{\lambda g}(x_k - \lambda \nabla f(x_k))$  
   - **Splitting methods:** Handle $f$ and $g$ separately using their proximal operators  

 
## Examples

### Example 1: $\ell_1$ Norm (Soft Thresholding)

Let $f(x) = \|x\|_1$. Then

\[
\text{prox}_{\lambda \| \cdot \|_1}(x)_i = \text{sign}(x_i) \cdot \max(|x_i| - \lambda, 0)
\]

- Each component is **shrunk toward zero**, promoting sparsity.  
- Widely used in **LASSO regression** and **compressed sensing**.

### Example 2: Indicator Function

Let $f(x) = \delta_C(x)$, the indicator of a convex set $C$. Then

\[
\text{prox}_{\lambda \delta_C}(x) = \arg\min_{y \in C} \frac{1}{2\lambda} \|y - x\|_2^2 = P_C(x)
\]

- The proximal operator reduces to the **metric projection** onto $C$.  
- Intuition: move $x$ to the closest feasible point in $C$.

### Example 3: Quadratic Function

Let $f(x) = \frac{1}{2} \|x\|_2^2$. Then

\[
\text{prox}_{\lambda f}(x) = \frac{x}{1 + \lambda}, \quad f_\lambda(x) = \frac{1}{2(1+\lambda)} \|x\|_2^2
\]

- Smooths the function and scales down the input.  

 

## Applications / Implications

- **Proximal Gradient Methods:** Handle composite objectives $f(x) + g(x)$ where $f$ is smooth and $g$ is nonsmooth.  
- **Splitting Algorithms:** Alternating updates with proximal operators allow decomposition in high-dimensional problems.  
- **Regularization:** $\ell_1$, nuclear norm, and indicator functions are easily handled using proximal operators.  
- **Duality:** Proximal operators relate to Fenchel conjugates via the **Moreau decomposition**:

\[
x = \text{prox}_{\lambda f}(x) + \lambda \, \text{prox}_{f^*/\lambda}(x/\lambda)
\]

 
## Summary / Key Takeaways

- Moreau envelopes provide **smooth approximations** of nonsmooth convex functions.  
- Proximal operators generalize gradient steps to **nonsmooth settings**.  
- Proximal updates often have **closed-form solutions** for many common functions.  
- They are central to **modern optimization algorithms**, including proximal gradient, ADMM, and primal-dual splitting.  
- The connection with duality and conjugates makes them a **versatile and powerful tool** in convex optimization.
