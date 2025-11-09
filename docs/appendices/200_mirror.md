# Appendix H: Mirror Descent and Bregman Geometry

Gradient Descent (GD) is the de facto method for minimizing differentiable functions, but it implicitly assumes Euclidean geometry.  
In many structured domains—such as probability simplices or sparse models—Euclidean updates can destroy problem structure or cause instability.  

Mirror Descent (MD) generalizes GD by incorporating geometry-aware updates via a mirror map and Bregman divergence.  
It performs gradient-like updates in a dual space, respecting the *intrinsic geometry* of the domain.

 
## H.1 Motivation and Limitations of Euclidean GD

Standard GD update:
\[
x_{t+1} = x_t - \eta \nabla f(x_t)
\]
assumes Euclidean distance
\[
\|x-y\|_2 = \sqrt{\sum_i (x_i - y_i)^2}.
\]

This works well in $\mathbb{R}^n$ without structure, but fails to respect constraints or sparsity.

In practice:

- Many parameters are nonnegative or normalized (probabilities, weights).  
- Euclidean steps can violate constraints or zero out coordinates.  
- The “flat” $\ell_2$ geometry treats all directions equally.

> Insight: Gradient Descent is geometry-specific. Mirror Descent generalizes it by changing the *metric* via a mirror map.

 

## H.2 Geometry in Optimization

The “steepest descent” direction depends on the notion of distance.  
GD implicitly minimizes a *linearized loss* plus a Euclidean proximity term.

| Scenario | Natural Constraint | Appropriate Geometry |
|-----------|--------------------|----------------------|
| Probability vectors | $x_i\ge0, \sum_i x_i=1$ | KL / entropy geometry |
| Sparse models | $\|x\|_1$-structured | $\ell_1$ geometry |
| Online learning | multiplicative updates | log-space geometry |

Using Euclidean projections in these domains can cause:

- abrupt projection onto boundaries,
- loss of positivity or sparsity,
- geometric inconsistency.

 

## H.3 Mirror Descent Framework

Let $\psi(x)$ be a mirror map — a strictly convex, differentiable potential encoding the geometry.

Define the dual coordinate:
\[
u = \nabla \psi(x),
\]
and its inverse mapping through the convex conjugate $\psi^*$:
\[
x = \nabla \psi^*(u).
\]

### Bregman Divergence
The geometry is quantified by the Bregman divergence:
\[
D_\psi(x \| y)
= \psi(x) - \psi(y) - \langle \nabla\psi(y), x - y \rangle.
\]

- Measures how nonlinear $\psi$ is between $x$ and $y$.  
- When $\psi(x)=\tfrac12\|x\|_2^2$, $D_\psi$ becomes $\tfrac12\|x-y\|_2^2$.  
- When $\psi(x)=\sum_i x_i\log x_i$, $D_\psi$ becomes KL divergence.

 
## H.4 Mirror Descent Update Rule

Mirror Descent minimizes a linearized loss plus a geometry-aware regularizer:
\[
x_{t+1}
= \arg\min_{x\in\mathcal{X}}
\Big\{ \langle \nabla f(x_t), x - x_t\rangle
+ \tfrac{1}{\eta} D_\psi(x \| x_t) \Big\}.
\]

Equivalent dual-space form:
\[
\begin{aligned}
u_t &= \nabla \psi(x_t),\\
u_{t+1} &= u_t - \eta \nabla f(x_t),\\
x_{t+1} &= \nabla \psi^*(u_{t+1}).
\end{aligned}
\]

✅ MD is gradient descent in dual coordinates, where distances are measured by $D_\psi$ instead of $\|x-y\|_2$.

 
## H.5 Comparing GD, Projected GD, and MD

| Method | Update Rule | Geometry | Comments |
|---------|--------------|-----------|-----------|
| Gradient Descent | $x - \eta\nabla f$ | Euclidean | may leave feasible set |
| Projected GD | $\text{Proj}(x - \eta\nabla f)$ | Euclidean + projection | can cause discontinuous jumps |
| Mirror Descent | $\arg\min_x \langle\nabla f, x - x_t\rangle + \frac{1}{\eta}D_\psi(x\|x_t)$ | Bregman | smooth, structure-preserving |

 

## H.6 Simplex Example (KL Geometry)

Let $x\in\Delta^2=\{x\ge0, x_1+x_2=1\}$, objective $f(x)=x_1^2+2x_2$, $\eta=0.3$.

### Euclidean GD + Projection
1. $\nabla f=(2x_1,2)=(1,2)$,  
2. $y=x-\eta\nabla f=(0.2,-0.1)$,  
3. Project → $x_{new}=(1,0)$.

→ Projection kills one coordinate ⇒ lost smoothness.

### Mirror Descent with Negative Entropy
Mirror map $\psi(x)=\sum_i x_i\log x_i$.  
Update:
\[
x_i^{new}\propto x_i\exp(-\eta\nabla_i f(x)),
\quad \text{then normalize.}
\]
Gives $x\approx(0.57,0.43)$ — smooth, positive, stays in simplex.

> MD follows the manifold of the simplex naturally—no harsh projection.

 

## H.7 Choosing the Mirror Map

| Mirror Map $\psi(x)$ | Bregman Divergence $D_\psi$ | Typical Domain / Application |
|-----------------------|-----------------------------|------------------------------|
| $\tfrac12\|x\|_2^2$ | Euclidean distance | unconstrained $\mathbb{R}^n$ |
| $\sum_i x_i\log x_i$ | KL divergence | simplex, probabilities |
| $\|x\|_1$ or variants | $\ell_1$ geometry | sparse models |
| log-barrier $\sum_i -\log x_i$ | barrier divergence | positive orthant |

Mirror maps act as design choices defining the optimization geometry.

 
## H.8 Practical Remarks

When to prefer Mirror Descent:

- Structured domains (simplex, positive vectors, sparse spaces)
- Smooth, structure-preserving updates desired
- Avoiding discontinuous projections

Computational notes:

- Some $\psi$ yield closed-form updates (e.g. multiplicative weights).  
- Works with adaptive or momentum step-size schemes.  
- Often underlies algorithms in online learning, boosting, and natural gradient methods.

---

## H.9 Convergence at a Glance

For convex $f$ with bounded gradients $\|\nabla f\|\le G$ and strong convex mirror map $\psi$,
Mirror Descent achieves the same sublinear rate as projected subgradient methods:
\[
f(\bar{x}_T)-f(x^*)
\le O\!\left(\frac{1}{\sqrt{T}}\right),
\]
but with improved *geometry-adapted* constants that exploit curvature of $\psi$.

 