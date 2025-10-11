Gradient Descent (GD) is the de facto method for minimizing differentiable functions, it implicitly assumes Euclidean geometry, which may not respect the natural structure of many problems. Mirror Descent generalizes GD by incorporating **geometry-aware updates** via a **mirror map** and **Bregman divergence**, making it particularly suitable for constrained, probabilistic, and sparse domains. 


## 1. Introduction and Motivation

Gradient Descent is often introduced as the default optimization method:

$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

This seemingly simple update assumes that the underlying optimization space is **Euclidean**, where distance is measured using the $\ell_2$ norm:

$$\|x - y\|_2 = \sqrt{\sum_i (x_i - y_i)^2}$$

Works well for unconstrained problems in $\mathbb{R}^n$ with no additional structure.  

However, in real-world machine learning and optimization problems:

- Parameters often live in **structured spaces** like probability simplices or sparse domains.
- Euclidean distance is often **not the most natural notion of distance**.
- Applying Euclidean updates can **destroy problem structure** or create **instabilities**.

> **Key insight:** Gradient Descent is not inherently “wrong”—it’s just geometry-specific. Mirror Descent generalizes GD to respect the **intrinsic geometry** of the problem.


## 2. Geometry in Optimization — Why It Matters
Standard GD treats all directions equally. The steepest descent direction is simply the gradient $\nabla f(x)$, derived from **Euclidean distance**. This is equivalent to asking: "In which direction does $f(x)$ decrease fastest if distance is measured by the $\ell_2$ norm?"

While sufficient for many unconstrained problems, this implicitly assumes:

- The feasible set is unbounded or flat.
- Movement along all axes is equally “costly”.
- There are no constraints like positivity or normalization.

### When Euclidean Geometry Fails

Many modern optimization problems involve **structured domains**:

| Scenario | Constraint / Structure | Natural Geometry |
|----------|-----------------------|-----------------|
| Probability vectors | $x_i \ge 0, \sum_i x_i = 1$ | KL divergence / simplex geometry |
| Attention weights | Positive and normalized | Entropy geometry |
| Sparse models | Preference for zeros | $\ell_1$ geometry |
| Online learning | Avoid drastic updates | Multiplicative weights / log-space |

Using Euclidean GD in these settings can lead to:

- **Harsh projections** that instantly zero out components.
- **Violation of sparsity or positivity constraints**.
- **Loss of smoothness or natural probabilistic interpretation**.

> **Observation:** Gradient Descent works “locally,” but may be incompatible with **global geometry** of the feasible domain.



## 3. Mirror Descent: A Geometric Generalization

Mirror Descent adapts Gradient Descent to **non-Euclidean geometries**, encoding the structure of the optimization space.

### Mirror Maps and Dual Coordinates

A **mirror map** $\psi(x)$ is a **strictly convex, differentiable function** representing the geometry:

$$u = \nabla \psi(x)$$

- $x$ = primal variable  
- $u$ = dual variable (coordinates in transformed space)

Updates occur in the **dual space**, then are mapped back using the **convex conjugate**:

$$x = \nabla \psi^*(u)$$

This allows GD-like updates while respecting geometry constraints.



### Bregman Divergence and Interpretation

The **Bregman divergence** associated with $\psi$ generalizes squared Euclidean distance:

$$D_\psi(x \| y) = \psi(x) - \psi(y) - \langle \nabla \psi(y), x - y \rangle$$

#### Intuition:

- Think of $D_\psi(x \| y)$ as a **geometry-aware distance measure**.  
- It captures **how far $x$ is from $y$ in the space defined by $\psi$**, not just in straight-line Euclidean distance.
- Conceptually, it measures the **error between the linear approximation of $\psi$ at $y$ and the true value at $x$**:

  - $\psi(y) + \langle \nabla \psi(y), x - y \rangle$ = linear approximation  
  - $\psi(x) - (\text{linear approximation})$ = how "nonlinear" the space feels from $y$ to $x$  

- When $\psi(x) = \frac12 \|x\|_2^2$, the Bregman divergence reduces to **Euclidean distance squared**.  

- For **negative entropy** (common in probability spaces), it reduces to **KL divergence**.

- **Purpose in MD:** Ensures updates respect the **intrinsic geometry**, balancing movement in the objective with staying “close” in the right geometry.

**Mirror Descent update (primal form):**

$$x_{t+1} = \arg\min_{x \in \mathcal{X}} \left\{ \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{\eta} D_\psi(x \| x_t) \right\}$$

> Move in a descent direction while staying “close” according to Bregman divergence, not Euclidean distance.


## 4. Gradient Descent vs Mirror Descent
### Primal View (Projection vs Bregman Step)

- **GD:** Step in Euclidean space; may leave the feasible domain → project back.  
- **MD:** Step along geometry-aware Bregman divergence; **no harsh projection needed**.

| Method | Update Rule | Notes |
|--------|------------|------|
| Gradient Descent | $x - \eta \nabla f$ | Euclidean, may leave domain |
| Projected GD | $\text{Proj}(x - \eta \nabla f)$ | Projection may destroy smoothness |
| Mirror Descent | $\arg\min_x \langle \nabla f, x - x_t \rangle + \frac{1}{\eta} D_\psi(x\|x_t)$ | Structure-preserving |


### Dual View (GD in Dual Space)

Mirror Descent can also be understood as **Gradient Descent in dual coordinates**:

$$
\begin{aligned}
u_t &= \nabla \psi(x_t) \\
u_{t+1} &= u_t - \eta \nabla f(x_t) \\
x_{t+1} &= \nabla \psi^*(u_{t+1})
\end{aligned}
$$

✅ MD is **GD in a warped coordinate system**, where distance and directions are geometry-aware.


## 5. Intuitive Example on the Simplex

Consider $x \in \Delta^2 = \{ x \ge 0, x_1 + x_2 = 1 \}$ and objective:

$$f(x) = x_1^2 + 2 x_2$$

Initial point: $x = (0.5, 0.5)$, step size $\eta = 0.3$.

---

### Behavior of GD + Projection

1. Gradient: $\nabla f = (2x_1, 2) = (1,2)$
2. Step: $y = x - \eta \nabla f = (0.2, -0.1)$
3. Project onto simplex: $x_{\text{new}} = (1, 0)$

> ❌ Projection abruptly kills one component. Smoothness and probabilistic structure are lost.

---

### Behavior of Mirror Descent (KL / Negative Entropy)

Mirror map: $\psi(x) = \sum_i x_i \log x_i$  

Update rule:

$x_i^{\text{new}} \propto x_i \exp(-\eta \nabla_i f(x))$

Normalized:

$x \approx (0.57, 0.43)$

> ✅ Smooth, positive, stays in the simplex, no harsh projection.

---

### Interpretation

| Method | Intuition |
|--------|----------|
| GD + Projection | Walks straight → hits boundary → forced back |
| Mirror Descent | Walks along curved, geometry-aware space → never violates constraints |

> Mirror Descent = **optimization with geometry turned ON**.



## 6. Choosing the Mirror Map — Geometry as a Design Choice

### Entropy Geometry (Simplex)

- Mirror map: $\psi(x) = \sum_i x_i \log x_i$
- Divergence: KL divergence
- Update: multiplicative weights  
- Applications: probability vectors, attention mechanisms

### $\ell_1$ Geometry (Sparsity)

- Mirror map encourages sparse updates
- Useful in compressed sensing, feature selection

### Euclidean as a Special Case

- Mirror map: $\psi(x) = \frac12 \|x\|_2^2$
- Divergence: squared Euclidean distance
- Recovers standard GD

---

## 7. Practical Guidance for Practitioners

### When to Prefer Mirror Descent

- Structured domains (simplex, positive vectors, sparse spaces)
- Smooth, structure-preserving updates required
- Avoiding costly or disruptive Euclidean projections

### Computational Remarks

- Choice of mirror map affects **efficiency** (some dual mappings are cheap, others expensive)
- Often simple closed-form updates exist (multiplicative weights, exponentiated gradient)
- Integration with adaptive step sizes or momentum is possible


Mirror Descent is a **powerful generalization of Gradient Descent**, making the **geometry of the domain explicit** in the update rule. By carefully choosing a **mirror map**, one can design updates that:

- Preserve constraints naturally
- Avoid projection shocks
- Respect sparsity or probability structure
- Connect elegantly to modern ML methods like attention, boosting, and natural gradient
