# Chapter 5: Convex Functions

Convex functions are the objectives we minimise. Understanding them is essential, because convexity of the objective is what turns an optimisation problem from "possibly impossible" to "provably solvable".

---

## 5.1 Definitions of convexity

### 5.1.1 Basic definition
A function $f : \mathbb{R}^n \to \mathbb{R}$ is **convex** if for all $x,y$ in its domain and all $\theta \in [0,1]$,
$$
f(\theta x + (1-\theta) y)
\le
\theta f(x) + (1-\theta) f(y)~.
$$

If the inequality is strict whenever $x \ne y$ and $\theta \in (0,1)$, then $f$ is **strictly convex**.

### 5.1.2 Epigraph definition
The **epigraph** of $f$ is
$$
\mathrm{epi}(f) = \{ (x, t) \in \mathbb{R}^n \times \mathbb{R} : f(x) \le t \}.
$$

**Key fact:** $f$ is convex if and only if $\mathrm{epi}(f)$ is a convex set (Boyd and Vandenberghe, 2004).  
This links convex *functions* to convex *sets*, and unlocks the geometry: tangent hyperplanes to $\mathrm{epi}(f)$ correspond to subgradients (Chapter 6).

---

## 5.2 First-order characterisation

If $f$ is differentiable, then $f$ is convex if and only if
$$
f(y) \ge f(x) + \nabla f(x)^\top (y - x)
\quad \text{for all } x,y.
$$

Interpretation:
- The first-order Taylor approximation is always a global underestimator.
- The gradient at $x$ defines a supporting hyperplane to the epigraph of $f$ at $(x, f(x))$.

This inequality is sometimes called the **first-order condition for convexity**.

---

## 5.3 Second-order characterisation

If $f$ is twice continuously differentiable, then $f$ is convex if and only if its Hessian is positive semidefinite everywhere:
$$
\nabla^2 f(x) \succeq 0
\quad \text{for all } x~.
$$

If $\nabla^2 f(x) \succ 0$ for all $x$, then $f$ is strictly convex.

This is a computational test for convexity of smooth functions: check the Hessian.

---

## 5.4 Examples of convex functions

1. **Affine functions:**  
   $f(x) = a^\top x + b$.  
   Always convex (and concave).

2. **Quadratic functions with PSD Hessian:**  
   $f(x) = \tfrac12 x^\top Q x + c^\top x + d$,  
   where $Q \succeq 0$ (symmetric positive semidefinite).  
   Convex because $\nabla^2 f(x) = Q \succeq 0$.

3. **Norms:**  
   $f(x) = \|x\|$ for any norm.  
   All norms are convex.

4. **Maximum of affine functions:**  
   $f(x) = \max_i (a_i^\top x + b_i)$.  
   Convex because it is the pointwise maximum of convex functions.

5. **Log-sum-exp function:**  
   $f(x) = \log \left( \sum_{i=1}^k \exp(a_i^\top x + b_i) \right)$.  
   This function is convex and is ubiquitous in statistics and machine learning (softmax, logistic regression). The convexity follows from Jensen’s inequality and properties of the exponential (Boyd and Vandenberghe, 2004).

---

## 5.5 Jensen’s inequality

Let $f$ be convex, and let $X$ be a random variable taking values in the domain of $f$. Then
$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]~.
$$

This is **Jensen’s inequality**. It generalises the definition of convexity from two-point averages to arbitrary expectations. As a special case, for scalars $x_1,\dots,x_n$ and weights $\theta_i \ge 0$ with $\sum_i \theta_i = 1$,
$$
f\!\left(\sum_i \theta_i x_i\right)
\le
\sum_i \theta_i f(x_i).
$$

Jensen’s inequality is foundational in probability, statistics, and information theory.

---

## 5.6 Operations that preserve convexity

If $f$ and $g$ are convex, then:

- $f + g$ is convex.
- $\alpha f$ is convex for any $\alpha \ge 0$.
- $\max\{f,g\}$ is convex.
- Composition with an affine map preserves convexity:  
  If $A$ is a matrix and $b$ a vector, then $x \mapsto f(Ax + b)$ is convex.

If $f$ is convex and nondecreasing in each argument, and each $g_i$ is convex, then the composition $x \mapsto f(g_1(x), \dots, g_k(x))$ is convex. This helps you build new convex functions from known ones.

---

## 5.7 Level sets of convex functions

For $\alpha \in \mathbb{R}$, define the **sublevel set**
$$
\{ x : f(x) \le \alpha \}.
$$

If $f$ is convex, then every sublevel set is convex (Boyd and Vandenberghe, 2004). This is crucial: constraints of the form $f(x) \le \alpha$ are convex constraints.

For example, the set
$$
\{ x : \|Ax - b\|_2 \le \epsilon \}
$$
is convex because $x \mapsto \|Ax-b\|_2$ is convex.

---

## 5.8 Strict and strong convexity

- $f$ is **strictly convex** if
$$
f(\theta x + (1-\theta) y) < \theta f(x) + (1-\theta) f(y)
$$
for all $x \ne y$ and $\theta \in (0,1)$.

- $f$ is **strongly convex** with parameter $m>0$ if
$$
f(y) \ge f(x) + \nabla f(x)^\top (y-x) + \frac{m}{2} \|y-x\|_2^2.
$$

Strong convexity implies a unique minimiser and gives quantitative convergence rates for gradient methods.

---

## 5.9 Takeaways

1. Convexity of $f$ can be checked:
    - by definition,
    - by first-order condition,
    - by Hessian PSD (if smooth).
2. Convex functions have convex sublevel sets, which become convex feasible regions.
3. Jensen’s inequality is the probabilistic face of convexity.
4. Strong convexity gives uniqueness and fast convergence guarantees.

Next, we will drop differentiability altogether. Convex optimisation does not need smoothness. That is where subgradients come in.
 
<!-- Convex functions are those that lie below their chords. Formally, $f: \mathbb{R}^n \to \mathbb{R}$ is convex if for all $x,y$ and $0\le \lambda \le 1$,

$$
f(\lambda x + (1 - \lambda) y)
\le
\lambda f(x) + (1 - \lambda) f(y)
$$

This means the graph of $f$ bends upwards (or is a straight line in degenerate cases), never above the straight line between $(x,f(x))$ and $(y,f(y))$. Equivalently, $f$ has a convex epigraph: $\mathrm{epi}(f) = {(x,t): f(x)\le t}$ is a convex set. Convex functions generalize familiar shapes like lines, parabolas opening upward, exponential, etc., to $\mathbb{R}^n$. They are the objective functions we can minimize efficiently because any local minimum is global.

**Examples of convex functions:**

- **Linear functions:** $f(x) = a^T x + b$ are convex (in fact, both convex and concave). Their epigraph is a halfspace. Linear programming objectives are convex.

- **Quadratic functions:** $f(x) = \frac{1}{2}x^T Q x + c^T x + d$ is convex if and only if $Q \succeq 0$ (positive semidefiniteness ensures the curvature is upward). E.g. least squares $f(x)=|Ax - b|_2^2$ has Hessian $2A^T A \succeq 0$, so it’s convex.

- **Norms:** $f(x)=|x|_p$ is convex for $p \ge 1$. Norms are actually sublinear (positive homogeneous and triangle inequality), hence convex. The $\ell_1$ norm is convex (forming a “V” in 1D), which is why we can add it as a regularizer for inducing sparsity. $\ell_2$ norm is strictly convex (its Hessian is identity except at 0 which is nondifferentiable for $\ell_1$).

- **Exponential:** $f(x)=e^x$ (on $\mathbb{R}$) is convex since $e^{\lambda x+(1-\lambda)y} = e^{\lambda x}e^{(1-\lambda)y} \le \lambda e^x + (1-\lambda)e^y$ by weighted AM-GM inequality. In $\mathbb{R}^n$, $f(x) = \sum_{i} e^{x_i}$ is convex (as sum of convex). This appears in logistic regression or softmax.

- **Negative entropy:** $f(p) = \sum_{i} p_i \ln p_i$ (with $p_i \ge 0$, $\sum p_i=1$) is convex in the probability simplex. This is used in maximum entropy problems and is strongly convex (with respect to appropriate norm).

- **Indicator functions of convex sets:** $f(x) = \begin{cases}0 & x\in C, \ +\infty & x\notin C,\end{cases}$ is convex if $C$ is convex. These allow one to enforce constraints via optimization (minimize $f(x)+$ something will force $x\in C$ in optimum). They are not differentiable but have subgradients related to normals of $C$.

**First-order characterization:** If $f$ is differentiable, $f$ is convex if and only if

$$
f(y) \ge f(x) + \langle \nabla f(x), \, y - x \rangle
\quad \forall\, x, y
$$


This is a powerful characterization: it says the tangent hyperplane at any point $x$ lies below the graph of $f$ everywhere. In other words, the gradient at $x$ provides a global underestimator of $f$ (supporting hyperplane to epigraph). This inequality is essentially the definition above in the limit $\lambda\to0$. Geometrically, this means no tangent line ever goes above the function. For a convex differentiable $f$, we have $f(y) - f(x) \ge \nabla f(x)^T (y-x)$, so moving from $x$ in any direction, the actual increase in $f$ is at least as large as the linear prediction by $\nabla f(x)$ (since the function bends upward or straight). At optimum $\hat{x}$, a necessary and sufficient condition (for convex differentiable $f$) is $\nabla f(x^) = 0$. This ties to optimality: $\nabla f(x^)=0$ means $f(y)\ge f(\hat{x}) + \nabla f(\hat{x})^T (y-\hat{x}) = f(\hat{x})$ for all $y$, so $\hat{x}$ is global minimizer.

If $f$ is not differentiable, a similar condition holds with subgradients: $f$ is convex iff for all $x,y$ there exists a (sub)gradient $g \in \partial f(x)$ such that $f(y) \ge f(x) + g^T(y-x)$. The set of all subgradients $\partial f(x)$ is a convex set (the subdifferential). At optimum, $0 \in \partial f(x^)$ is the condition. For example, $f(x)=|x|$ has subgradient $g=\mathrm{sign}(x)$ (multivalued at 0, where any $g\in[-1,1]$ is subgradient). Setting $0$ in subdifferential yields $0\in[-1,1]$, so indeed $x^=0$ is minimizer.

**Second-order test:** If $f$ is twice differentiable, then $f$ is convex if and only if its Hessian is positive semidefinite everywhere: $\nabla^2 f(x) \succeq 0$ for all $x$. This is often the easiest way to verify convexity for smooth functions. E.g., for $f(x) = \ln(1+e^x)$, $\nabla^2 f(x) = \frac{e^x}{(1+e^x)^2} > 0$, so it’s convex (in fact, logistic loss). If Hessian has even one negative eigenvalue somewhere, $f$ is not convex (it’s bending downward in that direction at that point).

**Jensen’s inequality:** A very important property of convex functions is Jensen’s inequality, which in one form states: for a random variable $X$ (or any weighting), $f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$ if $f$ is convex. In other words, the function of an average is no more than the average of function values. This is really the definition of convexity extended to infinite mixtures (integrals). For discrete weights $\lambda_i$ summing to 1, Jensen’s inequality is

$$
f\!\left(\sum_i \lambda_i x_i\right)
\le
\sum_i \lambda_i f(x_i)
$$


which is exactly convexity. Jensen’s inequality has many uses: in machine learning, it justifies algorithms like EM (which use the inequality to create surrogate objectives), and it provides bounds like $\log(\mathbb{E}[e^X]) \ge \mathbb{E}[X]$ (by convexity of $\log$ or $e^x$). As a simple example, taking $f(x)=x^2$ and $X$ uniform in ${-1,1}$, Jensen says $(\mathbb{E}[X])^2 = 0^2 \le \mathbb{E}[X^2] = 1$, which is true. Or $f(x)=\frac{1}{x}$ convex on $(0,\infty)$ implies $\frac{1}{\mathbb{E}[X]} \le \mathbb{E}[\frac{1}{X}]$ for positive $X$. In optimization, Jensen’s inequality often helps in proving convexity of expectations: if you mix some distributions or uncertain inputs, the expected loss is convex if the loss function is convex.

**Operations preserving convexity:** Convexity is preserved under nonnegative weighted sum: if $f_1,\dots,f_m$ are convex, then $f(x)=\sum_i \alpha_i f_i(x)$ with $\alpha_i \ge 0$ is convex. This is useful for regularization: objective + $\lambda$*regularizer is convex if both are convex. Also, composition with an affine mapping: $g(x)=f(Ax+b)$ is convex if $f$ is convex (because $g(x_1+x_2) = f(A(x_1+x_2)+b) = f(Ax_1+b + Ax_2+b - b) = f(\tilde{x}_1 + \tilde{x}_2)$ essentially, which preserves inequality). If $f$ is convex and increasing, then $f(g(x))$ is convex whenever $g(x)$ is convex (this often appears for norms: e.g. $f(t)=t^p$ (increasing for $p\ge1$) and $g(x)=|x|$ convex, so $|x|^p$ is convex). More generally, rules like perspective function and partial minimization preserve convexity (the latter means if $F(x,y)$ is convex in $(x,y)$, then $h(x) = \inf_y F(x,y)$ is convex — important for Lagrange dual function derivations).

**Strong convexity and Lipschitz gradient:** As mentioned earlier, if $f$ is $\mu$-strongly convex, then it has a unique minimizer and satisfies $f(y) \ge f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}|y-x|^2$. Geometrically, $f$ grows at least as a quadratic of curvature $\mu$. This property gives error bounds: any $x$ implies $f(x)-f^* \ge \frac{\mu}{2}|x-\hat{x}|^2$, so in convex optimization theory one derives convergence rates like $|x_k-\hat{x}|^2$ contracts each iteration for strongly convex functions. Many loss functions in ML are strongly convex after adding a small ridge (e.g. squared loss with $|w|^2$ regularizer).

If $f$ has $L$-Lipschitz continuous gradient, it satisfies $f(y) \le f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}|y-x|^2$ (the other side of the inequality chain), meaning it doesn’t grow faster than a quadratic of curvature $L$. Together, $\mu$-strong convexity and $L$-Lipschitz gradient imply $\mu I \preceq \nabla^2 f(x) \preceq L I$. Such functions have condition number $L/\mu$, and gradient descent has linear convergence rate $(1-\mu/L)$.

**Convexity tests summary: **Use first-order test if you have an expression for $\nabla f$ (verify $f(y) \ge f(x)+\nabla f(x)^T(y-x)$). Use second-order (Hessian PSD) if $f$ is twice differentiable. Use known building blocks and rules for composite functions otherwise.

**Sublevel sets:** A nice property: for a convex function $f$, the sublevel sets ${x: f(x) \le \alpha}$ are convex sets. So convex functions have convex “contours” (not necessarily the level sets themselves if the function isn’t strictly convex at that level, but all points where $f \le$ constant form a convex region). This matters because if you minimize a convex function, ${x: f(x) \le f(x_0)}$ shrinks convexly around the minimizer as $f(x_0)$ decreases. Many convergence proofs consider how far one can be from optimum by looking at such sublevel sets intersections with balls, etc.

To develop intuition, consider that convex functions are “bowl-shaped”. For one-dimensional convex $f$, the line segment between any two points on the curve lies above the curve: imagine a cup shape or a straight line. A concave function is the opposite (like a cap shape, satisfying $f(\lambda x + (1-\lambda)y) \ge \lambda f(x)+(1-\lambda)f(y)$). In economics, utility functions are concave (risk aversion), but we often maximize them, converting to a convex minimization (max concave = min -concave which is convex). Thus convexity simplifies optimization because it eliminates local minima.

Finally, convex analysis provides more advanced tools (conjugate functions, subgradient calculus, etc.) which we won't fully delve into, but one key idea is convex conjugate $f^*(y) = \sup_x (y^T x - f(x))$. This is always convex even if $f$ is not, and for convex $f$ it encodes its geometry (supporting hyperplanes). Fenchel’s duality arises from this, providing another pathway to dual problems (as we’ll see next chapter). -->