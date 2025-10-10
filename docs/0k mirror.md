## Mirror Descent — Geometry-Aware Optimization

### 🔍 Motivation

Standard Gradient Descent assumes **Euclidean geometry**, implicitly using the $\ell_2$ norm:

- **Updates:** $x_{t+1} = x_t - \eta \nabla f(x_t)$  
- **Distance notion:** Euclidean distance $\|x - y\|_2$

This is well-suited for problems in $\mathbb{R}^n$ without additional structure.  
However, in many modern optimization problems:

- Parameters lie in the **probability simplex** (e.g., distributions, mixture weights).
- **Sparsity or positivity** constraints are important.
- **KL divergence** or **entropy** is a more natural distance than Euclidean norm.
- Euclidean projection can **destroy structure** or be computationally expensive.

> ✅ **Mirror Descent** introduces a geometry-aware update mechanism using a **Bregman divergence**, allowing updates that are natural to the problem domain.




### Mirror Descent Formulation

Choose a **mirror map** $\psi(x)$: a **strictly convex, differentiable function** defining the geometry.

- **Bregman Divergence:**
$$
D_\psi(x \| y) = \psi(x) - \psi(y) - \langle \nabla \psi(y), x - y \rangle
$$

- **Primal update form:**
$$
x_{t+1} = \arg\min_{x \in \mathcal{X}} \left\{ \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{\eta} D_\psi(x \| x_t) \right\}
$$

- **Dual-space interpretation:**
\[
\begin{aligned}
u_t &= \nabla \psi(x_t) \quad \text{(mirror map to dual space)} \\
u_{t+1} &= u_t - \eta \nabla f(x_t) \quad \text{(gradient step in dual coordinates)} \\
x_{t+1} &= \nabla \psi^*(u_{t+1}) \quad \text{(map back via convex conjugate)}
\end{aligned}
\]

---

### 📌 Examples of Mirror Maps

| Geometry / Constraint | Mirror Map $\psi(x)$ | Bregman Divergence $D_\psi$ | Resulting Update |
|----------------------|---------------------|-----------------------------|------------------|
| **Euclidean (GD)** | $\frac{1}{2}\|x\|_2^2$ | $\frac{1}{2}\|x - y\|_2^2$ | Additive: $x - \eta \nabla f$ |
| **Simplex (probabilities)** | $\sum_i x_i \log x_i$ (**negative entropy**) | KL divergence: $\sum_i x_i \log \frac{x_i}{y_i}$ | **Multiplicative weights / exponentiated gradient** |
| **Sparse / $\ell_1$ geometry** | $\|x\|_1 \log \|x\|_1$ | Divergence encouraging sparsity | Sparse-aware descent |

---

### ⭐ Example: Exponentiated Gradient on the Simplex

Consider $x \in \Delta^n = \{x \ge 0, \sum_i x_i = 1\}$ and loss $f(x)$.

- Choose **mirror map**:  
$$
\psi(x) = \sum_{i=1}^n x_i \log x_i \quad (\text{negative entropy})
$$

- Compute $\nabla \psi(x) = [1 + \log x_i]_{i=1}^n$

- Dual update:
\[
u_{t+1,i} = u_{t,i} - \eta \nabla_i f(x_t)
\]

- Map back using $\psi^*(u)$:
\[
x_{t+1,i} = \frac{\exp(u_{t+1,i} - 1)}{\sum_j \exp(u_{t+1,j} - 1)}
\]

Substituting $u_t = 1 + \log x_t$, we get the **multiplicative update**:
$$
x_{t+1,i} = \frac{x_{t,i} \exp(-\eta \nabla_i f(x_t))}{\sum_j x_{t,j} \exp(-\eta \nabla_j f(x_t))}
$$

> ✅ This is known as **Exponentiated Gradient**, **Multiplicative Weights**, or **Hedge Algorithm** in Online Learning.

---

### 🚀 Comparison Table

| Method | Update Rule | Geometry | Notes |
|--------|------------|---------|------|
| Gradient Descent | $x - \eta \nabla f$ | Euclidean | Can leave simplex, requires projection |
| Projected GD | Take GD → project | Euclidean + hard constraints | Projection may break structure |
| **Mirror Descent** | Gradient in dual space + Bregman step | **Flexible (entropy, KL, etc.)** | **Structure-preserving**, cheaper projection |

---

### 🧠 Key Takeaways

- Mirror Descent = **Gradient Descent in dual (mirror) space**.
- Choice of **mirror map defines geometry**.
- With **entropy**, we get **multiplicative updates**, which **preserve simplex structure naturally**.
- With **Euclidean $\psi$**, Mirror Descent reduces to standard GD.

---

Would you like me to:
- ✅ Add a **visual intuition** diagram explanation?
- ✅ Include **regret bound interpretation in Online Learning**?
- ✅ Generate a **clean PDF chapter-style export**?

Just say **"continue"**, and I’ll build the next section!
