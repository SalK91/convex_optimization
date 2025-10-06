## Projections and Proximal Operators in Constrained Convex Optimization

In many convex optimization problems, we want to minimize a convex, differentiable function $f(x)$ subject to some constraint that limits $x$ to a feasible region $\mathcal{X} \subseteq \mathbb{R}^n$:

$$
\min_{x \in \mathcal{X}} f(x)
$$

A standard gradient descent step is

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

but this update might move $x_{t+1}$ **outside** the feasible region $\mathcal{X}$.  
To fix that, we add a **projection step** that brings the point back into the allowed set.

 
### Projection Operator

The **projection** of a point $y$ onto a convex set $\mathcal{X}$ is the closest point in the set to $y$:

$$
\text{Proj}_{\mathcal{X}}(y) = \arg\min_{x \in \mathcal{X}} \|x - y\|^2
$$

So the **projected gradient descent** update becomes

$$
x_{t+1} = \text{Proj}_{\mathcal{X}}\big(x_t - \eta \nabla f(x_t)\big)
$$

#### Geometric intuition
Think of taking a gradient step in the direction of steepest descent, possibly leaving the feasible region.  
The projection then “snaps” that point back to the **nearest feasible location**. This ensures all iterates $x_t$ stay within $\mathcal{X}$ while still moving downhill with respect to $f$.

**Example:**  
If $\mathcal{X} = \{x : \|x\|_2 \le 1\}$ (the unit ball), the projection is

$$
\text{Proj}_{\mathcal{X}}(y) = \frac{y}{\max(1, \|y\|_2)}
$$

That means:
- If $y$ is inside the ball, it stays there ($\|y\|_2 \le 1$).
- If $y$ is outside, scale it down to lie exactly on the boundary.

 

## **From Projections to Proximal Operators**

Projection helps when constraints are *explicitly defined* by a set (e.g., nonnegativity or norm bounds). But many optimization problems include *non-smooth regularization terms* instead — for example, $g(x) = \lambda \|x\|_1$ to promote sparsity.

The **proximal operator** generalizes projection to handle such non-smooth functions directly.


### **Definition**

For a convex (possibly non-differentiable) function $g(x)$, its **proximal operator** is defined as:

$$
\text{prox}_{\lambda g}(y)
= \arg\min_x \left( g(x) + \frac{1}{2\lambda}\|x - y\|^2 \right)
$$

#### **Interpretation**
The proximal operator finds a point $x$ that balances **two objectives**:

1. **Stay close to $y$** — enforced by the squared term $\frac{1}{2\lambda}\|x - y\|^2$.
2. **Reduce $g(x)$** — the regularization or penalty term.

The parameter $\lambda > 0$ controls this trade-off:

- A small $\lambda$ → stronger pull toward $y$ (less movement).  
- A large $\lambda$ → more freedom to reduce $g(x)$.

The squared distance term acts as a **soft tether**, keeping $x$ near $y$ while allowing it to move toward regions where $g(x)$ is smaller or structured.

 
### **Indicator Function and Connection to Projection**

Let’s see how projection appears as a *special case* of the proximal operator.

Define the **indicator function** of a convex set $\mathcal{X}$ as:

$$
I_{\mathcal{X}}(x) =
\begin{cases}
0, & x \in \mathcal{X} \\
+\infty, & x \notin \mathcal{X}
\end{cases}
$$

Now, substitute $g(x) = I_{\mathcal{X}}(x)$ into the definition of the proximal operator:

$$
\text{prox}_{\lambda I_{\mathcal{X}}}(y)
= \arg\min_x \Big( I_{\mathcal{X}}(x) + \frac{1}{2\lambda}\|x - y\|^2 \Big)
$$

Because $I_{\mathcal{X}}(x)$ is infinite outside $\mathcal{X}$, the minimization is effectively **restricted to** $x \in \mathcal{X}$.  
Thus we get:

$$
\text{prox}_{\lambda I_{\mathcal{X}}}(y)
= \arg\min_{x \in \mathcal{X}} \|x - y\|^2
= \text{Proj}_{\mathcal{X}}(y)
$$

✅ Therefore, **projection is just a proximal operator** for the indicator of a set.

## **Understanding the Proximal Step**

The proximal operator can be viewed as a **correction step**:

- The **gradient step** moves toward minimizing the smooth part $f(x)$.
- The **proximal step** adjusts that move to respect the structure imposed by $g(x)$ — e.g., sparsity, nonnegativity, or feasibility.

When combining both, we get the **proximal gradient method**:

$$
x_{t+1} = \text{prox}_{\eta g}\big(x_t - \eta \nabla f(x_t)\big)
$$

This algorithm generalizes projected gradient descent — it works for both *constraint sets* (through indicator functions) and *regularizers* (like $\ell_1$-norms).

  

### Example: Proximal of the $\ell_1$-Norm

We want to compute the proximal operator of the $\ell_1$-norm:

$$
\text{prox}_{\lambda \|\cdot\|_1}(y)
= \arg\min_x \left( \lambda \|x\|_1 + \frac{1}{2}\|x - y\|^2 \right)
$$


### Step 1. Coordinate-wise separation

Because both $\|x\|_1$ and $\|x - y\|^2$ are separable across coordinates, we can solve for each component independently:

$$
\min_x \left( \lambda |x| + \frac{1}{2}(x - y)^2 \right)
$$

Thus, we only need to handle the **scalar problem** for one coordinate $y \in \mathbb{R}$:

$$
\phi(x) = \lambda |x| + \frac{1}{2}(x - y)^2
$$

and find

$$
x^\star = \arg\min_x \phi(x)
$$

---

### Step 2. Subgradient optimality condition

Since $\phi$ is convex (but not differentiable at $x = 0$), the optimality condition is

$$
0 \in \partial \phi(x^\star)
$$

Compute the subgradient:

$$
\partial \phi(x) = \lambda \, \partial |x| + (x - y)
$$

where

$$
\partial |x| =
\begin{cases}
\{1\}, & x > 0 \\[4pt]
[-1, 1], & x = 0 \\[4pt]
\{-1\}, & x < 0
\end{cases}
$$

Hence, the optimality condition becomes

$$
0 \in \lambda s + (x^\star - y), \quad s \in \partial |x^\star|
$$

Rewriting:

$$
x^\star = y - \lambda s, \quad s \in \partial |x^\star|
$$

---

### Step 3. Case Analysis

#### **Case 1:** $x^\star > 0$

Then $s = 1$, so

$$
x^\star = y - \lambda
$$

This is valid only if $x^\star > 0 \implies y > \lambda$.

Hence, when $y > \lambda$, the minimizer is:

$$
x^\star = y - \lambda
$$

---

#### **Case 2:** $x^\star < 0$

Then $s = -1$, so

$$
x^\star = y + \lambda
$$

This is valid only if $x^\star < 0 \implies y < -\lambda$.

Hence, when $y < -\lambda$, the minimizer is:

$$
x^\star = y + \lambda
$$

---

#### **Case 3:** $x^\star = 0$

Then $s \in [-1, 1]$, and the condition

$$
0 \in \lambda s + (0 - y)
$$

means there exists $s \in [-1, 1]$ such that $y = \lambda s$.  
This happens exactly when $y \in [-\lambda, \lambda]$.

Hence, when $|y| \le \lambda$, the minimizer is:

$$
x^\star = 0
$$

---

### Step 4. Combine the cases

Putting the three cases together:

$$
\text{prox}_{\lambda |\cdot|}(y) =
\begin{cases}
y - \lambda, & y > \lambda \\[6pt]
0, & |y| \le \lambda \\[6pt]
y + \lambda, & y < -\lambda
\end{cases}
$$

Or equivalently, in compact form:

$$
\boxed{
\text{prox}_{\lambda |\cdot|}(y)
= \text{sign}(y) \cdot \max(|y| - \lambda, 0)
}
$$

---

### Step 5. Extend to vector case

For a vector $y \in \mathbb{R}^n$, the proximal operator applies **coordinate-wise**:

$$
\big(\text{prox}_{\lambda \|\cdot\|_1}(y)\big)_i
= \text{sign}(y_i) \cdot \max(|y_i| - \lambda, 0)
$$

---

### Step 6. Intuition

- When $|y_i| \le \lambda$, the quadratic term cannot compensate for the $\ell_1$ penalty, so the coordinate shrinks to **zero** (sparsity).
- When $|y_i| > \lambda$, the coordinate is **shrunk** by $\lambda$ toward zero but remains nonzero.
- This behavior is called **soft-thresholding**, and it is the key to algorithms like LASSO and ISTA for sparse recovery.
