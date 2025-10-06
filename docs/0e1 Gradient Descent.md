# Gradient Descent: Derivation and Convergence

We aim to minimize a differentiable function $f$ over a feasible set $\mathcal{X}$:

$$
\min_{x \in \mathcal{X}} f(x).
$$

 
## 1. Local Approximation

At iteration $t$, we have the current point $x_t$.  
To make optimization tractable, we build a **first-order Taylor approximation** of $f$ around $x_t$:

$$
f(x) \approx f(x_t) + \langle \nabla f(x_t), x - x_t \rangle.
$$

- This is a **linear approximation** of $f$ at $x_t$.  
- Directly minimizing this linear model is unbounded below; it would push $x$ infinitely in the negative gradient direction.

 
## 2. Adding a Quadratic Regularization Term

To prevent taking arbitrarily large steps, we add a **quadratic penalty** that discourages moving far from $x_t$:

$$
f(x) \approx f(x_t) + \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{2\eta} \|x - x_t\|^2,
$$

where $\eta > 0$ is the **step size (learning rate)**.

> **Intuition**:  
> - Limits trust in the linear approximation (local approximation only).  
> - Creates a trade-off between decreasing the linear term and staying near $x_t$.  
> - $\frac{1}{2\eta}$ controls the strength of the penalty:  
>   - Small $\eta$: conservative, small steps.  
>   - Large $\eta$: aggressive, larger steps.  

This is conceptually related to **proximal methods** or **trust-region approaches** in optimization.

 

## 3. Deriving the Gradient Descent Update

We update $x_{t+1}$ by minimizing the quadratic model:

$$
x_{t+1} = \arg\min_{x \in \mathcal{X}} \Big[ f(x_t) + \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{2\eta} \|x - x_t\|^2 \Big].
$$

- Ignore $f(x_t)$ since it's constant with respect to $x$.  
- Take the gradient of the remaining terms and set it to zero:

$$
\nabla f(x_t) + \frac{1}{\eta}(x - x_t) = 0
$$

Solve for $x$:

$$
x - x_t = -\eta \nabla f(x_t)
$$

Hence the **gradient descent update**:

$$
\boxed{x_{t+1} = x_t - \eta \nabla f(x_t)}
$$

---

## 4. Convergence Analysis

### Assumptions

We assume:

1. **$L$-smoothness (Lipschitz-continuous gradient):**

$$
\|\nabla f(x) - \nabla f(y)\| \le L \|x - y\|, \quad \forall x, y
$$

2. **$\mu$-strong convexity**:

$$
f(y) \ge f(x) + \langle \nabla f(x), y-x \rangle + \frac{\mu}{2} \|y-x\|^2, \quad \forall x, y
$$

 
### 4.1 Descent Lemma

For $L$-smooth functions, we have:

$$
f(x_{t+1}) \le f(x_t) + \langle \nabla f(x_t), x_{t+1}-x_t \rangle + \frac{L}{2} \|x_{t+1}-x_t\|^2
$$

Substitute $x_{t+1} = x_t - \eta \nabla f(x_t)$:

$$
\begin{aligned}
f(x_{t+1}) &\le f(x_t) + \langle \nabla f(x_t), -\eta \nabla f(x_t) \rangle + \frac{L}{2} \|\eta \nabla f(x_t)\|^2 \\
&= f(x_t) - \eta \|\nabla f(x_t)\|^2 + \frac{L \eta^2}{2} \|\nabla f(x_t)\|^2 \\
&= f(x_t) - \left( \eta - \frac{L\eta^2}{2} \right) \|\nabla f(x_t)\|^2
\end{aligned}
$$

> **Implication**: If $\eta \le \frac{1}{L}$, the term $\eta - \frac{L\eta^2}{2} > 0$, so $f(x_{t+1}) \le f(x_t)$. Each step decreases the function value.

 
### 4.2 Convergence for Strongly Convex Functions

If $f$ is $\mu$-strongly convex, gradient descent converges **linearly**. Specifically:

$$
\begin{aligned}
\|x_{t+1} - x^*\|^2 &= \|x_t - \eta \nabla f(x_t) - x^*\|^2 \\
&= \|x_t - x^*\|^2 - 2\eta \langle \nabla f(x_t), x_t - x^* \rangle + \eta^2 \|\nabla f(x_t)\|^2
\end{aligned}
$$

From strong convexity:

$$
\langle \nabla f(x_t), x_t - x^* \rangle \ge \frac{\mu L}{\mu + L} \|x_t - x^*\|^2 + \frac{1}{\mu + L} \|\nabla f(x_t)\|^2
$$

Choosing step size $\eta = \frac{2}{\mu + L}$ yields:

$$
\|x_{t+1} - x^*\|^2 \le \left( \frac{L - \mu}{L + \mu} \right)^2 \|x_t - x^*\|^2
$$

> **Linear convergence**: Distance to optimum shrinks by a constant factor each step.

 

### 4.3 Convergence Rate Summary

- For $L$-smooth **convex** (not strongly convex):

$$
f(x_T) - f(x^*) \le \frac{L \|x_0 - x^*\|^2}{2T}
$$

Sublinear rate $O(1/T)$.

- For $L$-smooth **$\mu$-strongly convex**:

$$
\|x_T - x^*\| \le \left( \frac{L - \mu}{L + \mu} \right)^T \|x_0 - x^*\|
$$

Linear rate $O(\rho^T)$, $\rho < 1$.
