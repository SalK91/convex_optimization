# Chapter 5: Convex Functions

This chapter develops the basic tools for understanding convex functions: their definitions, geometric characterisations, first- and second-order tests, and operations that preserve convexity. These tools will later support duality, optimality conditions, and algorithmic analysis.

 
## Definitions of convexity

A function $f : \mathbb{R}^n \to \mathbb{R}$ is convex if for all $x,y$ in its domain and all $\theta \in [0,1]$,
$$
f(\theta x + (1-\theta) y)
\le
\theta f(x) + (1-\theta) f(y).
$$

The graph of $f$ never dips below the straight line between $(x,f(x))$ and $(y,f(y))$. If the inequality is strict whenever $x \neq y$, the function is *strictly convex*.

A powerful geometric viewpoint comes from the epigraph:
$$
\mathrm{epi}(f) = 
\{ (x,t) \in \mathbb{R}^n \times \mathbb{R} : f(x) \le t \}.
$$
The function $f$ is convex if and only if its epigraph is a convex set. This connects convex functions to the convex sets studied earlier.

 
## First-order characterisation

If $f$ is differentiable, then $f$ is convex if and only if
$$
f(y) \ge f(x) + \nabla f(x)^\top (y - x)
\quad\text{for all } x,y.
$$

Interpretation:

- The tangent plane at any point $x$ lies below the function everywhere.
- $\nabla f(x)$ defines a supporting hyperplane to the epigraph.
- The gradient provides a global linear underestimator of $f$.

This geometric picture is crucial in optimisation:
at a minimiser $x^\star$, convexity implies  
$$
\nabla f(x^\star) = 0 \quad \Longleftrightarrow \quad x^\star \text{ is a global minimiser}.
$$

For nondifferentiable convex functions, the gradient is replaced by a subgradient, which plays the same role in forming supporting hyperplanes.

 
## Second-order characterisation

If $f$ is twice continuously differentiable, then convexity can be checked via curvature:

$$
f \text{ is convex } \iff \nabla^2 f(x) \succeq 0 \text{ for all } x.
$$

- If the Hessian is positive semidefinite everywhere, the function bends upward.  
- If $\nabla^2 f(x) \succ 0$ everywhere, the function is strictly convex.  
- Negative eigenvalues indicate directions of negative curvature — impossible for convex functions.

This characterisation connects convexity to the spectral properties of the Hessian discussed earlier.

 
## Examples of convex functions

1. Affine functions:  
   $$
   f(x) = a^\top x + b.
   $$  
   Always convex (and concave). They define supporting hyperplanes.

2. Quadratic functions with PSD Hessian:  
   $$
   f(x) = \tfrac12 x^\top Q x + c^\top x + d,\quad Q \succeq 0.
   $$  
   Convex because the curvature matrix $Q$ is PSD.

3. Norms:  
   $$
   f(x) = \|x\|_p \quad (p \ge 1).
   $$  
   All norms are convex; in ML, norms induce regularisers (Lasso, ridge).

4. Maximum of affine functions:  
   $$
   f(x) = \max_i (a_i^\top x + b_i).
   $$  
   Convex because the maximum of convex functions is convex.  
   (Important in SVM hinge loss.)

5. Log-sum-exp:  
   $$
   f(x) = \log\!\left( \sum_{i=1}^k \exp(a_i^\top x + b_i) \right).
   $$  
   A smooth approximation to the max; convex by Jensen’s inequality. Appears in softmax, logistic regression, partition functions.

 
## Jensen’s inequality

Let $f$ be convex and $X$ a random variable in its domain. Then:
$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)].
$$

This generalises the definition of convexity from finite averages to expectations.  
Practically:

- convex functions “pull upward” under averaging,
- log-sum-exp is convex because exponential is convex,
- EM and variational methods rely on Jensen to construct lower bounds.

As a finite form, for $\theta_i \ge 0$ with $\sum \theta_i = 1$,
$$
f\!\left(\sum_i \theta_i x_i\right)
\le
\sum_i \theta_i f(x_i).
$$

 
## Operations that preserve convexity

Convexity is preserved under many natural constructions:

- Nonnegative scaling:  
  If $f$ is convex and $\alpha \ge 0$, then $\alpha f$ is convex.

- Addition:  
  If $f$ and $g$ are convex, then $f+g$ is convex.

- Maximum:  
  $\max\{f,g\}$ is convex.

- Affine pre-composition:  
  If $A$ is a matrix,
  $$
  x \mapsto f(Ax + b)
  $$
  is convex.

- Monotone composition rule:  
  If $f$ is convex and nondecreasing in each argument, and each $g_i$ is convex,  
  then $x \mapsto f(g_1(x), \dots, g_k(x))$ is convex.

 
## Level sets of convex functions

For $\alpha \in \mathbb{R}$, the sublevel set is
$$
\{ x : f(x) \le \alpha \}.
$$

If $f$ is convex, every sublevel set is convex.  
This property is crucial because inequalities $f(x) \le \alpha$ are ubiquitous in constraints.

Examples:

- Norm balls:  
  $\{ x : \|x\|_2 \le r \}$  
- Linear regression confidence ellipsoids:  
  $\{ x : \|Ax - b\|_2 \le \epsilon \}$

These sets enable convex constrained optimisation formulations.

 
## Strict and strong convexity

### Strict convexity

A function is strictly convex if
$$
f(\theta x + (1-\theta) y)
<
\theta f(x) + (1-\theta) f(y)
$$
for all $x \neq y$ and $\theta \in (0,1)$.

Strict convexity implies unique minimisers.

### Strong convexity

A differentiable function is $\mu$-strongly convex if
$$
f(y) 
\ge 
f(x) + \nabla f(x)^\top (y - x) + \frac{\mu}{2}\|y - x\|_2^2.
$$

Strong convexity adds *quantitative curvature*: the function grows at least quadratically away from its minimiser.

Consequences:

- unique minimiser,
- gradient descent achieves linear convergence rate,  
  error shrinks as  
  $$
  \|x_{k+1} - x^\star\| \le (1 - \eta\mu)\|x_k - x^\star\| ,
  $$
- conditioning ($\kappa = L/\mu$) governs algorithmic difficulty.

Strong convexity is frequently induced by regularisation (e.g., ridge regression adds $\tfrac{\lambda}{2}\|x\|_2^2$).
 
## Mental Map

```text
                      Convex Functions
      Objective landscapes with predictable geometry and guarantees
                              │
                              ▼
                Core idea: no bad local minima
        (every local minimum is global; geometry is well-behaved)
                              │
                              ▼
     ┌───────────────────────────────────────────────────────────┐
     │ Definition of Convexity                                   │
     │ f(θx+(1−θ)y) ≤ θf(x)+(1−θ)f(y)                            │
     │ - Graph lies below all chords                             │
     │ - Strict convexity: inequality is strict                  │
     │ - Epigraph view: f convex ⇔ epi(f) is a convex set       │
     └───────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌────────────────────────────────────────────────────────────┐
     │ First-Order Geometry (Supporting Hyperplanes)              │
     │ f(y) ≥ f(x)+∇f(x)ᵀ(y−x)                                    │
     │ - Tangent plane globally underestimates f                  │
     │ - ∇f(x) defines a supporting hyperplane to epi(f)          │
     │ - Optimality: ∇f(x*)=0 ⇔ x* global minimizer (smooth case)│
     │ - Nonsmooth extension: subgradients (next chapter)         │
     └────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌────────────────────────────────────────────────────────────┐
     │ Second-Order Characterisation                              │
     │ ∇²f(x) ⪰ 0  for all x                                      │
     │ - PSD Hessian ⇔ upward curvature everywhere               │
     │ - PD Hessian ⇔ strict convexity                           │
     │ - Links convexity to eigenvalues and curvature (Ch.3)      │
     └────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ Canonical Examples                                          │
     │ - Affine functions: supporting hyperplanes                  │
     │ - Quadratics (Q⪰0): curvature from Hessian                  │
     │ - Norms: regularization geometry                            │
     │ - Max of affine functions: hinge loss, LPs                  │
     │ - Log-sum-exp: smooth max, softmax, logistic regression     │
     └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ Jensen’s Inequality                                         │
     │ f(E[X]) ≤ E[f(X)]                                           │
     │ - Convex functions penalize variability                     │
     │ - Basis for EM, variational bounds, log-sum-exp convexity   │
     │ - Extends convexity from points to expectations             │
     └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ Convexity-Preserving Operations                             │
     │ - Scaling (α≥0), addition                                   │
     │ - Max of convex functions                                   │
     │ - Affine pre-composition f(Ax+b)                            │
     │ - Monotone composition rules                                │
     │ Role: modular construction of convex models                 │
     └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ Sublevel Sets                                               │
     │ {x : f(x) ≤ α}                                              │
     │ - Always convex for convex f                                │
     │ - Enables convex inequality constraints                     │
     │ - Norm balls, confidence ellipsoids, feasibility regions    │
     └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
     ┌─────────────────────────────────────────────────────────────┐
     │ Strict vs Strong Convexity                                  │
     │ Strict convexity: unique minimizer                          │
     │ Strong convexity: f ≥ tangent + (μ/2)‖x−y‖²                 │
     │ - Quantitative curvature                                    │
     │ - Linear convergence of gradient descent                    │
     │ - Conditioning κ=L/μ governs difficulty                     │
     │ - Often induced via ℓ₂ regularization                       │
     └─────────────────────────────────────────────────────────────┘
                              
```