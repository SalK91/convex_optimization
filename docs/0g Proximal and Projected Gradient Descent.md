## **Projections and Proximal Operators in Constrained Convex Optimization**

In many convex optimization problems, we aim to minimize a differentiable convex function $f(x)$ over a closed convex set $\mathcal{X} \subseteq \mathbb{R}^n$:

$$
\min_{x \in \mathcal{X}} f(x)
$$

A simple gradient step of the form

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

may produce a point that lies **outside** the feasible region $\mathcal{X}$. To ensure feasibility, we introduce the **projection operator** onto $\mathcal{X}$:

$$
\text{Proj}_{\mathcal{X}}(y) = \arg\min_{x \in \mathcal{X}} \|x - y\|^2
$$

The **projected gradient descent** update is then written as:

$$
x_{t+1} = \text{Proj}_{\mathcal{X}}(x_t - \eta \nabla f(x_t))
$$

This projection step ensures that each iterate remains within $\mathcal{X}$ while still following the descent direction of $f$.  
Geometrically, it “pulls” the point back to the nearest feasible position in the set after taking a gradient step.  
This is especially important when $\mathcal{X}$ encodes constraints such as non-negativity, norm bounds, or affine restrictions.

---

### **From Projections to Proximal Operators**

While projection handles explicit *set constraints*, many optimization problems involve **non-smooth regularization terms** instead.  
For such cases, the idea of projection generalizes to the **proximal operator**.

For a convex (possibly non-differentiable) function $g(x)$, the **proximal operator** is defined as:

$$
\text{prox}_{\lambda g}(y) = \arg\min_x \Big( g(x) + \frac{1}{2\lambda} \|x - y\|^2 \Big)
$$

If $g$ is the **indicator function** of a convex set $\mathcal{X}$—that is, $g(x) = 0$ for $x \in \mathcal{X}$ and $+\infty$ otherwise—then the proximal operator reduces exactly to the projection operator:

$$
\text{prox}_{\lambda g}(y) = \text{Proj}_{\mathcal{X}}(y)
$$

Hence, projections are a **special case** of proximal operators.

---

### **Intuition and Role in Optimization**

Both operators serve as **correction mechanisms** during iterative optimization:

- The **projection operator** ensures **feasibility** with respect to constraints.  
- The **proximal operator** enforces **regularization** or **penalty structure** when dealing with non-smooth functions.

Intuitively, these operators help balance *descent direction* with *constraint or regularization structure*, maintaining the geometry required for convergence in convex optimization.

---

### **Example**

If $\mathcal{X} = \{x \mid \|x\|_2 \le 1\}$, the projection onto $\mathcal{X}$ is:

$$
\text{Proj}_{\mathcal{X}}(y) = \frac{y}{\max(1, \|y\|_2)}
$$

Similarly, if $g(x) = \lambda \|x\|_1$, then the proximal operator corresponds to the **soft-thresholding** function:

$$
\text{prox}_{\lambda \| \cdot \|_1}(y)_i = \text{sign}(y_i) \cdot \max(|y_i| - \lambda, 0)
$$

---

### **Summary**

| Concept | Mathematical Definition | Purpose |
|----------|--------------------------|----------|
| Projection $\text{Proj}_{\mathcal{X}}(y)$ | $\arg\min_{x \in \mathcal{X}} \|x - y\|^2$ | Keeps iterates within the feasible set |
| Proximal operator $\text{prox}_{\lambda g}(y)$ | $\arg\min_x \big( g(x) + \frac{1}{2\lambda}\|x - y\|^2 \big)$ | Handles non-smooth penalties or implicit constraints |

---

**Figure:** *Illustration of a projected gradient step.*  
After taking a gradient descent step that moves $x_t$ outside the feasible set $\mathcal{X}$, the projection operator maps it back to the nearest point in $\mathcal{X}$, ensuring feasibility of the next iterate.
