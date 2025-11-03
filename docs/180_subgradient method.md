# Appendix F: Subgradient Method: Derivation, Geometry, and Convergence

This appendix presents the subgradient method—the fundamental algorithm for minimizing nonsmooth convex functions.  
It generalizes gradient descent to functions such as the $\ell_1$ norm, hinge loss, and ReLU penalties that appear frequently in machine learning and signal processing.

## F.1 Problem Setup

We consider

$$
\min_{x \in \mathcal{X}} f(x),
$$

where $f$ is convex but possibly nondifferentiable and $\mathcal{X}$ is a convex feasible set.

 
## F.2 Subgradients and Geometry

A subgradient $g_t \in \partial f(x_t)$ satisfies

$$
f(y) \ge f(x_t) + \langle g_t,\, y - x_t \rangle, \quad \forall y \in \mathcal{X}.
$$

- If $f$ is differentiable, $\partial f(x_t) = \{\nabla f(x_t)\}$.  
- At a nonsmooth point (e.g. $|x|$ at $x=0$), $\partial f(x_t)$ is a set of supporting slopes.  
- Each subgradient defines a supporting hyperplane below the graph of $f$.

Hence a subgradient gives a descent direction even when $f$ lacks a unique gradient.

 
## F.3 Update Rule and Projection View

The projected subgradient step is

$$
x_{t+1} = \Pi_{\mathcal{X}}\!\big(x_t - \eta_t g_t\big),
$$

where
- $g_t \in \partial f(x_t)$,  
- $\eta_t>0$ is the step size,  
- $\Pi_{\mathcal{X}}$ projects onto $\mathcal{X}$.

If $\mathcal{X} = \mathbb{R}^n$, projection disappears:
\[
x_{t+1} = x_t - \eta_t g_t.
\]

Geometric view: move in a subgradient direction, then project back to feasibility.  
The method “slides” along the edges of $f$’s epigraph.

 

## F.4 Distance Analysis

Let $x^\star$ be an optimal solution. Expanding the squared distance:

\[
\|x_{t+1}-x^\star\|^2
= \|x_t - x^\star\|^2
- 2\eta_t\langle g_t, x_t - x^\star\rangle
+ \eta_t^2 \|g_t\|^2.
\]

By convexity,
\[
f(x_t) - f(x^\star) \le \langle g_t, x_t - x^\star\rangle.
\]

Substitute to get

\[
\|x_{t+1}-x^\star\|^2
\le
\|x_t - x^\star\|^2
- 2\eta_t\big(f(x_t)-f(x^\star)\big)
+ \eta_t^2 \|g_t\|^2.
\]

 

## F.5 Bounding Suboptimality

Rearranging:

\[
f(x_t)-f(x^\star)
\le
\frac{\|x_t-x^\star\|^2 - \|x_{t+1}-x^\star\|^2}{2\eta_t}
+ \frac{\eta_t}{2}\|g_t\|^2.
\]

This shows a trade-off:

- Large $\eta_t$ → faster steps but higher error term.  
- Small $\eta_t$ → more precise but slower progress.

 

## F.6 Convergence Rate

Assume $\|g_t\| \le G$. Summing over $t=0,\dots,T-1$:

\[
\sum_{t=0}^{T-1}\!\big(f(x_t)-f(x^\star)\big)
\le
\frac{\|x_0-x^\star\|^2}{2\eta}
+ \frac{\eta G^2 T}{2}.
\]

Define $\bar{x}_T = \tfrac{1}{T}\sum_{t=0}^{T-1} x_t$.  
By convexity of $f$,

\[
f(\bar{x}_T)-f(x^\star)
\le
\frac{\|x_0-x^\star\|^2}{2\eta T}
+ \frac{\eta G^2}{2}.
\]

Choosing $\eta_t = \tfrac{R}{G\sqrt{T}}$ with $R=\|x_0-x^\star\|$ yields

\[
f(\bar{x}_T)-f(x^\star)
\le
\frac{RG}{\sqrt{T}},
\]
i.e. a sublinear rate $O(1/\sqrt{T})$.

 
## F.7 Interpretation and Practice

- Works for any convex function, smooth or not.  
- Converges slower than smooth-gradient methods ($O(1/T)$ or linear), but applies more generally.  
- Step size schedule is crucial:  
  $\eta_t \!\downarrow 0$ for convergence, or fixed $\eta$ for steady error.  
- Averaging $\bar{x}_T$ improves stability.

### Typical ML Uses
| Model | Objective | Nonsmooth Term |
|--------|------------|----------------|
| LASSO | $\tfrac12\|Ax-b\|_2^2 + \lambda\|x\|_1$ | $\ell_1$ penalty |
| SVM | $\tfrac12\|w\|^2 + C\sum_i \max(0,1-y_i w^\top x_i)$ | hinge loss |
| Robust regression | $\sum_i |a_i^\top x - b_i|$ | absolute deviation |
| Neural nets | $\|w\|_1$ or ReLU activations | piecewise linear |

 

## F.8 Beyond Basic Subgradients

Many advanced methods refine or accelerate the basic idea:

- Stochastic subgradients: sample-based updates for large-scale ML.  
- Mirror descent: adapt geometry via Bregman divergences.  
- Proximal methods: replace step with proximal operator (see Appendix B).  
- Dual averaging & AdaGrad: adapt step sizes to coordinate scaling.

 

## F.9 Summary

- Subgradients generalize gradients to nondifferentiable convex functions.  
- The projected subgradient method provides a universal, robust minimization algorithm.  
- Achieves $O(1/\sqrt{T})$ convergence under bounded subgradients.  
- Foundation for stochastic, proximal, and mirror-descent algorithms explored in Chapters 9–10.
 