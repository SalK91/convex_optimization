# First-Order Optimization Methods

In machine learning, especially at large scale, we often cannot afford to solve convex problems using heavy, exact solvers (like simplex or interior-point methods). Instead, we rely on **first-order methods** — algorithms that use only **function values** and **gradients** to iteratively approach the solution.



## Gradient Descent (GD)

Gradient descent is the most fundamental algorithm for minimizing differentiable convex functions.

**Basic Idea:**  
At each iteration, move in the direction **opposite** to the gradient (the steepest descent direction), because it points toward lower values of the function.

**Update Rule:**
$$
x^{(k+1)} = x^{(k)} - \alpha_k \nabla f(x^{(k)})
$$
where:
- $\nabla f(x^{(k)})$ — gradient at the current point.
- $\alpha_k$ — step size (learning rate).

**Convergence (Convex Case):**
- If $f$ is convex and has **Lipschitz-continuous gradients**, gradient descent converges at rate $O(1/k)$ for constant step size.
- If $f$ is also **strongly convex**, the rate improves to **linear convergence**.

**Step Size Selection:**
- **Constant**: simple but requires tuning.
- **Diminishing**: $\alpha_k \to 0$ ensures convergence but may be slow.
- **Backtracking Line Search**: adaptively chooses $\alpha_k$ for efficiency.


## Subgradient Methods

Many important convex functions in ML are **non-differentiable** (e.g., $\|x\|_1$).  
The subgradient generalizes the gradient for such functions.

**Subgradient Definition:**  
A vector $g$ is a **subgradient** of $f$ at $x$ if:
$$
f(y) \ge f(x) + g^T (y - x), \quad \forall y
$$

**Update Rule:**
$$
x^{(k+1)} = x^{(k)} - \alpha_k g^{(k)}, \quad g^{(k)} \in \partial f(x^{(k)})
$$

**Key Trade-Off:** Subgradient methods are **robust** but converge more slowly ($O(1/\sqrt{k})$ in general).



## Stochastic Gradient Descent (SGD)

SGD is the workhorse of large-scale machine learning.

**When to Use:**  
When the objective is a **sum over many data points**:
$$
f(x) = \frac{1}{N} \sum_{i=1}^N f_i(x)
$$

**Update Rule:**
- Pick a random index $i_k$
- Use $\nabla f_{i_k}(x^{(k)})$ as an **unbiased** estimate of the full gradient:
$$
x^{(k+1)} = x^{(k)} - \alpha_k \nabla f_{i_k}(x^{(k)})
$$

**Advantages:**
- Much faster per iteration for large $N$.
- Enables online learning.

**Disadvantages:**
- Introduces variance; iterates “bounce” around the optimum.
- Requires careful learning rate schedules.


## Accelerated Gradient Methods

Nesterov’s Accelerated Gradient (NAG) achieves the **optimal** convergence rate for smooth convex functions: $O(1/k^2)$.

**Key Idea:**  
Introduce a **momentum term** that anticipates the next position, correcting the gradient direction.

**Update:**
$$
\begin{aligned}
y^{(k)} &= x^{(k)} + \beta_k (x^{(k)} - x^{(k-1)}) \\
x^{(k+1)} &= y^{(k)} - \alpha_k \nabla f(y^{(k)})
\end{aligned}
$$

**When Useful:**  
- Smooth convex problems where plain gradient descent is too slow.
- Large-scale ML tasks with batch updates.



### Why First-Order Methods Matter for ML
- Handle **huge datasets** efficiently.
- Require only **gradient information**, which is cheap for many models.
- Naturally fit into **streaming and online learning** setups.
- Form the backbone of **deep learning optimizers** (SGD, Adam, RMSProp — though deep nets are non-convex).
