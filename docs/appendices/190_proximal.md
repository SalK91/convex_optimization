# Appendix G | Projections and Proximal Operators in Constrained Convex Optimization

Many convex optimization problems involve constraints or nonsmooth penalties.  
This appendix unifies both under the framework of projections and proximal operators, which extend gradient-based methods to constrained or regularized settings.

 
## G.1 Problem Setup

We wish to minimize a convex, differentiable function \( f(x) \) subject to a convex feasible set \( \mathcal{X} \subseteq \mathbb{R}^n \):

\[
\min_{x \in \mathcal{X}} f(x).
\]

A plain gradient step,

\[
x_{t+1} = x_t - \eta \nabla f(x_t),
\]

may leave \( x_{t+1} \notin \mathcal{X} \).  
We fix this by projecting the iterate back into the feasible region.

 

## G.2 Projection Operator

The projection of a point \(y\) onto a convex set \(\mathcal{X}\) is

\[
\text{Proj}_{\mathcal{X}}(y)
= \arg\min_{x \in \mathcal{X}} \|x - y\|^2.
\]

Hence, the projected gradient descent update is

\[
x_{t+1} = \text{Proj}_{\mathcal{X}}\big(x_t - \eta \nabla f(x_t)\big).
\]

### Geometric Insight

- Take a descent step possibly outside the feasible set.  
- Project back to the closest feasible point.  
- The update direction remains aligned with the negative gradient while maintaining feasibility.

Example — Euclidean ball:  
If \( \mathcal{X} = \{x : \|x\|_2 \le 1\} \), then

\[
\text{Proj}_{\mathcal{X}}(y) = \frac{y}{\max(1, \|y\|_2)}.
\]

- Inside the ball → unchanged.  
- Outside → scaled back to the boundary.

---

## G.3 From Projections to Proximal Operators

Projections handle explicit constraints, but many problems use implicit penalties — e.g. sparsity (\(\|x\|_1\)), total variation, or nonnegativity penalties.

The proximal operator generalizes projection to handle such nonsmooth regularization directly.

### Definition

For a convex (possibly nondifferentiable) function \( g(x) \),

\[
\text{prox}_{\lambda g}(y)
= \arg\min_x \Big( g(x) + \tfrac{1}{2\lambda}\|x - y\|^2 \Big),
\]
where \( \lambda > 0 \) balances regularization vs. proximity.

### Interpretation

- The quadratic term \( \tfrac{1}{2\lambda}\|x - y\|^2 \) keeps \(x\) close to \(y\).  
- The function \( g(x) \) encourages structure (sparsity, smoothness, feasibility).  
- Small \(\lambda\): conservative correction; large \(\lambda\): stronger regularization.

The proximal step acts as a soft correction after a gradient step.

---

## G.4 Projection as a Special Case

Define the indicator function of a convex set \(\mathcal{X}\):

\[
I_{\mathcal{X}}(x) =
\begin{cases}
0, & x \in \mathcal{X}, \\[4pt]
+\infty, & x \notin \mathcal{X}.
\end{cases}
\]

Substitute \(g(x)=I_{\mathcal{X}}(x)\) into the proximal definition:

\[
\text{prox}_{\lambda I_{\mathcal{X}}}(y)
= \arg\min_x \Big( I_{\mathcal{X}}(x) + \tfrac{1}{2\lambda}\|x - y\|^2 \Big)
= \arg\min_{x \in \mathcal{X}} \|x - y\|^2
= \text{Proj}_{\mathcal{X}}(y).
\]

✅ Projection is just a proximal operator for an indicator function.

 
## G.5 Proximal Gradient Method

When minimizing a composite convex objective
\[
\min_x \; f(x) + g(x),
\]
where \(f\) is smooth and \(g\) convex (possibly nonsmooth), the proximal gradient method updates:

\[
x_{t+1} = \text{prox}_{\eta g}\big(x_t - \eta \nabla f(x_t)\big).
\]

- The gradient step reduces the smooth part \(f(x)\).  
- The proximal step enforces structure via \(g(x)\).  
This method generalizes projected gradient descent to include penalties and constraints seamlessly.

 
## G.6 Example: Proximal Operator of the \(\ell_1\)-Norm

We seek

\[
\text{prox}_{\lambda \|\cdot\|_1}(y)
= \arg\min_x \left( \lambda\|x\|_1 + \tfrac{1}{2}\|x - y\|^2 \right).
\]

### Step 1. Coordinate Separation

The problem is separable across coordinates:
\[
\min_x \sum_i \Big(\lambda |x_i| + \tfrac{1}{2}(x_i - y_i)^2\Big),
\]
so each coordinate solves
\[
\min_x \phi(x) = \lambda|x| + \tfrac{1}{2}(x - y)^2.
\]

 

### Step 2. Subgradient Optimality

Optimality condition:
\[
0 \in \partial\phi(x^\star) = \lambda \partial|x^\star| + (x^\star - y).
\]
Thus,
\[
x^\star = y - \lambda s, \quad s \in \partial |x^\star|.
\]

 
### Step 3. Case Analysis

| Case | Condition | Solution |
|------|------------|-----------|
| \(x^\star>0\) | \(y>\lambda\) | \(x^\star = y - \lambda\) |
| \(x^\star<0\) | \(y<-\lambda\) | \(x^\star = y + \lambda\) |
| \(x^\star=0\) | \(|y|\le\lambda\) | \(x^\star = 0\) |

 

### Step 4. Compact Form

\[
\boxed{
\text{prox}_{\lambda|\cdot|}(y)
= \text{sign}(y) \cdot \max(|y| - \lambda,\, 0)
}
\]

This is the soft-thresholding operator.

 
### Step 5. Vector Case

For \(y \in \mathbb{R}^n\),

\[
\big(\text{prox}_{\lambda\|\cdot\|_1}(y)\big)_i
= \text{sign}(y_i)\cdot\max(|y_i| - \lambda, 0).
\]

Each coordinate is independently shrunk toward zero — producing sparse solutions.

 
### Step 6. Interpretation

- Coordinates with \(|y_i| \le \lambda\) → set to zero (promotes sparsity).  
- Coordinates with \(|y_i| > \lambda\) → shrink by \(\lambda\).  
- The proximal operator thus blends denoising and regularization: it keeps large coefficients but trims small ones.

 

## G.7 Geometry and Connection to Algorithms

- Projection = nearest feasible point → handles *hard constraints*.  
- Proximal operator = nearest structured point → handles *soft regularization*.  
- Proximal gradient = combines both, yielding algorithms like:
  - ISTA / FISTA (sparse recovery, LASSO),
  - Projected gradient (feasibility),
  - ADMM (splitting into subproblems).

Proximal methods lie at the core of modern convex optimization and machine learning, offering flexibility for nonsmooth and constrained problems alike.

 

## G.8 Summary

- Projections and proximal operators generalize gradient steps to respect constraints and structure.  
- Projection is a special case of the proximal operator for an indicator function.  
- Proximal mappings handle nonsmooth regularizers (e.g., \(\ell_1\)-norm).  
- The proximal gradient method unifies constrained and regularized optimization.  
- Many state-of-the-art ML algorithms are built upon these proximal foundations.

