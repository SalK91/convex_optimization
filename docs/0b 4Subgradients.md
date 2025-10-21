# Subgradients

Subgradients generalize the concept of gradients to **nonsmooth convex functions**, allowing us to perform optimization even when the function is not differentiable. They are crucial for:

- Nonsmooth convex optimization (e.g., $\ell_1$ regularization, hinge loss)  
- Defining optimality conditions for convex functions  
- Linking primal and dual formulations via Fenchel conjugates  

Intuition: For a convex function $f$, the subgradient at a point $x$ is the slope of a **supporting hyperplane** that lies below the graph of $f$. It tells you a **direction along which the function does not decrease**, even if the gradient does not exist.


## Definitions and Formal Statements

Let $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ be convex. A vector $g \in \mathbb{R}^n$ is a **subgradient** of $f$ at $x$ if:

\[
f(y) \ge f(x) + \langle g, y - x \rangle \quad \forall y \in \mathbb{R}^n
\]

The **subdifferential** $\partial f(x)$ is the set of all subgradients at $x$:

\[
\partial f(x) = \{ g \in \mathbb{R}^n : f(y) \ge f(x) + \langle g, y - x \rangle, \forall y \in \mathbb{R}^n \}
\]

Key properties:

- If $f$ is differentiable at $x$, then $\partial f(x) = \{\nabla f(x)\}$.  
- The subdifferential is **convex and closed**.  
- $0 \in \partial f(x)$ if and only if $x$ is a **global minimizer** of $f$.  


## Step-by-Step Analysis / How to Use

To compute or apply subgradients:

1. Identify if the function $f$ is convex.  
2. Check if $f$ is differentiable at the point $x$:  
   - If yes, the gradient is the only subgradient.  
   - If no, identify supporting hyperplanes that satisfy the subgradient inequality.  
3. Use subgradients in algorithms:  
   - **Subgradient descent:** update $x_{k+1} = x_k - \alpha_k g_k$ with $g_k \in \partial f(x_k)$  
   - **Optimality checks:** $0 \in \partial f(x)$ implies $x$ is optimal  
4. Leverage relationships with conjugates:  
   - $g \in \partial f(x) \iff x \in \partial f^*(g)$ (duality link)

 
## Examples

### Example 1: Absolute Value

Let $f(x) = |x|$:

- For $x > 0$, $\partial f(x) = \{1\}$  
- For $x < 0$, $\partial f(x) = \{-1\}$  
- For $x = 0$, $\partial f(0) = [-1, 1]$  

Intuition: At 0, any slope between -1 and 1 forms a supporting hyperplane.

 
### Example 2: $\ell_1$ Norm

Let $f(x) = \|x\|_1$ for $x \in \mathbb{R}^n$:

\[
\partial \|x\|_1 = \{ g \in \mathbb{R}^n : g_i = \text{sign}(x_i) \text{ if } x_i \neq 0, \, g_i \in [-1,1] \text{ if } x_i = 0 \}
\]

- Each component can vary in $[-1,1]$ if the corresponding $x_i = 0$.  
- This property promotes **sparsity** in optimization problems like LASSO.

 
### Example 3: Indicator Function

Let $f(x) = \delta_C(x)$, the indicator of a convex set $C$:

\[
\partial \delta_C(x) =
\begin{cases}
\{ g : \langle g, y - x \rangle \le 0, \forall y \in C \} & x \in C \\
\emptyset & x \notin C
\end{cases}
\]

- Interpretation: the subdifferential is the **normal cone** to the set at $x$.

 
## Applications / Implications

- **Nonsmooth Optimization:** Subgradient descent generalizes gradient descent to nonsmooth convex functions.  
- **Optimality Conditions:** $0 \in \partial f(x)$ characterizes global minimizers.  
- **Duality:** Subgradients connect primal and dual variables via Fenchel conjugates.  
- **Sparsity and Regularization:** Subdifferentials of norms define constraints and thresholds in proximal algorithms.

 
## Summary / Key Takeaways

- Subgradients extend the concept of gradients to nonsmooth convex functions.  
- The subdifferential is convex, closed, and provides **supporting hyperplanes**.  
- Zero subgradient characterizes global optimality.  
- Subgradients are foundational in nonsmooth optimization, duality, and proximal algorithms.  
- Examples include absolute value, $\ell_1$ norms, and indicator functions of convex sets.
