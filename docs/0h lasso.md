## LASSO and Optimization Methods

The **LASSO** problem is formulated as:

$$
\min_{x \in \mathbb{R}^p} \ \|Ax - y\|_2^2 + \lambda \|x\|_1
$$

- $A \in \mathbb{R}^{n \times p}$: measurement/design matrix.
- $y \in \mathbb{R}^n$: observations.
- $\|x\|_1 = \sum_{i=1}^p |x_i|$: promotes **sparsity**.
- Assumption: $x$ is **$s$-sparse**, meaning only $s \ll p$ entries are nonzero → **compressed sensing**.


### 1.  LASSO via Subgradient Descent

Since $\|x\|_1$ is **non-smooth**, we use **subgradient descent**:

**Update rule:**
$$
x_{t+1} = x_t - \eta \left( 2A^\top(Ax_t - y) + \lambda z \right)
$$

where the **subgradient** $z \in \partial \|x\|_1$ is:

$$
z_i =
\begin{cases}
+1, & x_i > 0 \\
-1, & x_i < 0 \\
[-1, 1], & x_i = 0
\end{cases}
$$

- Convergence rate: $\mathcal{O}(1/\sqrt{t})$ — **slow** for practical use.
- Still important conceptually but inefficient compared to proximal methods.

---

### 2. Proximal Gradient for LASSO (ISTA)

To efficiently handle the non-smooth $\ell_1$ term, we use the **proximal gradient (ISTA)** method.

#### **Step 1 — Gradient update on smooth term $\|Ax - y\|_2^2$:**

$$
y_t = x_t - \eta \cdot 2A^\top(Ax_t - y)
$$

#### **Step 2 — Apply proximal operator of $\ell_1$ (soft thresholding):**

$$
x_{t+1} = \text{prox}_{\eta \lambda \|\cdot\|_1}(y_t)
$$

The **soft-thresholding operator**:

$$
\text{prox}_{\alpha \|\cdot\|_1}(z)_i = \text{sign}(z_i) \cdot \max(|z_i| - \alpha, 0)
$$

Thus the **ISTA update** becomes:

$$
x_{t+1} = \text{sign}(y_{t,i}) \cdot \max(|y_{t,i}| - \eta \lambda, 0)
$$

> Interpretation: **Gradient descent + shrinkage toward zero** → automatically induces sparsity.

---

### 3. FISTA — Accelerated Proximal Gradient for LASSO

ISTA has convergence rate $\mathcal{O}(1/t)$. **FISTA (Fast ISTA)** improves it to:

$$
\mathcal{O}(1/t^2) \quad \text{(optimal for first-order methods)}
$$

**Algorithm (FISTA):**

- Initialize: $x_0$, set $y_0 = x_0$, $t_0 = 1$.
- For $k = 0, 1, 2, \dots$:

1. **Proximal gradient step:**
$$
x_{k+1} = \text{prox}_{\eta \lambda \|\cdot\|_1} \left( y_k - \eta \cdot 2A^\top(Ay_k - y) \right)
$$

2. **Update momentum parameter:**
$$
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}
$$

3. **Nesterov extrapolation step:**
$$
y_{k+1} = x_{k+1} + \frac{t_k - 1}{t_{k+1}} (x_{k+1} - x_k)
$$

> **Key idea:** Instead of updating only from $x_k$ (like ISTA), FISTA uses a **look-ahead point $y_k$** to inject momentum and accelerate convergence.

---

### 📊 Method Comparison

| Method                     | Handles $\ell_1$? | Uses Prox? | Convergence Rate |
|-------------------------|------------------|-----------|------------------|
| Gradient Descent        | ❌               | ❌        | Fast (smooth only) |
| Subgradient Descent     | ✅               | ❌        | $\mathcal{O}(1/\sqrt{t})$ (slow) |
| **ISTA (Proximal GD)**  | ✅               | ✅        | $\mathcal{O}(1/t)$ |
| **FISTA (Accelerated)** | ✅ ✅             | ✅        | **$\mathcal{O}(1/t^2)$** ✅ |

---

