# Quasi-Newton Methods

Optimization often struggles not because of bad algorithms but because **the geometry of the loss landscape is distorted** — stretched, skewed, ill-conditioned.

Newton’s method fixes this by using curvature, but computing full Hessians is expensive.

> **Quasi-Newton methods are a clever hack:**  
> They *learn* curvature from past gradients **without ever computing second derivatives explicitly**.


## Newton's Update

The classical Newton update is:

$$
x_{t+1} = x_t - H^{-1}(x_t) \, \nabla f(x_t)
$$

- $H(x_t)$ is the Hessian (curvature matrix).
- Fast convergence but **requires computing and inverting $H$**.
- For dimension $d$, storing $H$ costs $O(d^2)$ and inverting costs $O(d^3)$ → **impractical at scale**.


## Key Idea of Quasi-Newton Methods

Instead of computing $H^{-1}$, **build an approximation**, call it $B_t \approx H^{-1}(x_t)$.

We extract curvature information **from how gradients change** after each step.

Define:

- **Step vector:**
  $$
  s_t = x_{t+1} - x_t
  $$
- **Gradient change:**
  $$
  y_t = \nabla f(x_{t+1}) - \nabla f(x_t)
  $$

Then we impose the **secant condition**:

$$
B_{t+1} \, y_t = s_t
$$

> 💬 Interpretation: *"If moving by $s_t$ caused gradient to change by $y_t$, then our internal curvature model should map $y_t \mapsto s_t$."*

This **imitates Newton's relation** without computing actual Hessians.

 

## BFGS — The Core Quasi-Newton Algorithm

BFGS maintains and updates an approximation of the **inverse Hessian**.

The update rule is:

$$
B_{t+1} = \left(I - \frac{s_t y_t^\top}{y_t^\top s_t}\right) B_t 
\left(I - \frac{y_t s_t^\top}{y_t^\top s_t}\right) 
+ \frac{s_t s_t^\top}{y_t^\top s_t}
$$

Properties:
- **Rank-2 update** → efficient
- Preserves **symmetry and positive definiteness**
- **No Hessian needed** — only gradients

Update the parameters using:

$$
x_{t+1} = x_t - B_{t+1} \nabla f(x_t)
$$

 
## Intuitive Geometry

- **Gradient Descent** assumes the world is a **sphere** → same learning rate in all directions.
- **Newton** knows the **true ellipse shape** of level sets and **reshapes space** so it becomes spherical.
- **BFGS** starts blind like GD but **learns to reshape space** gradually based on past gradient motion.

> Think of BFGS as an optimizer that **reconstructs a mental map of terrain curvature from memory**.

---

## Why BFGS Works — Memory of Curvature

Each $(s_t, y_t)$ pair captures **1D curvature information in the direction of motion**.

Storing many of these directions lets $B_t$ approximate **true curvature** more and more accurately.  
Eventually, the loss surface **feels spherical**, and optimization becomes **fast and direct**.


## L-BFGS — Making BFGS Scalable

Storing full $B_t$ takes $O(d^2)$ memory → too large when $d$ is in millions.

**Limited-memory BFGS (L-BFGS):**
- Keep only the **last $m$ pairs** $(s_t, y_t)$ (with $m \ll d$)
- Reconstruct the **vector product** $B_t \cdot \nabla f(x_t)$ using a **two-loop recursion**
- **Memory: $O(md)$ instead of $O(d^2)$**

This makes L-BFGS **practical for high-dimensional problems** — used in:
- **Logistic regression**
- **NLP models**
- **Deep learning fine-tuning**
- **SciPy & PyTorch optimizers**

 
## Comparison Table

| Method | Memory | Uses Hessian? | Update Direction | Convergence Speed | Affine-Aware? |
|--------|--------|--------------|------------------|------------------|---------------|
| Gradient Descent | $O(d)$ | ❌ | $-\nabla f$ | Linear | ❌ |
| Newton | $O(d^2)$ | ✅ Full | $-H^{-1} \nabla f$ | Quadratic | ✅ |
| BFGS | $O(d^2)$ | ✅ Approx. | $-B_t \nabla f$ | Superlinear | ✅ (approx) |
| **L-BFGS** | $O(md)$ | ✅ Approx. (limited) | Fast via recursion | Superlinear | ✅ (approx) |

 
## Final Mental Model

> **Gradient Descent**: Walks downhill with **fixed stride** — doesn't care about terrain shape.  
> **Newton**: Has a **full curvature map** — picks the fastest path, but expensive.  
> **BFGS/L-BFGS**: Starts blind like GD but **learns terrain structure from past steps**, gradually adapting stride and direction like Newton — **without ever seeing the actual Hessian**.

