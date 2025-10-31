# Appendix A: Common Inequalities and Identities

This appendix collects important inequalities used throughout convex analysis and optimisation. These are the “algebraic tools” you reach for in proofs, optimality arguments, and convergence analysis (Boyd and Vandenberghe, 2004; Hiriart-Urruty and Lemaréchal, 2001).

---

## A.1 Cauchy–Schwarz inequality

For any $x,y \in \mathbb{R}^n$,
$$
|x^\top y| \le \|x\|_2 \, \|y\|_2.
$$

Equality holds if and only if $x$ and $y$ are linearly dependent.

Consequences:

- Defines the notion of angle between vectors.
- Justifies dual norms.

---

## A.2 Jensen’s inequality

Let $f$ be convex, and let $X$ be a random variable. Then
$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)].
$$

In finite form: for $\theta_i \ge 0$ with $\sum_i \theta_i = 1$,
$$
f\!\left(\sum_i \theta_i x_i\right)
\le
\sum_i \theta_i f(x_i).
$$

Jensen’s inequality is equivalent to convexity: it says “the function at the average is no more than the average of the function values.” It is used constantly to prove convexity of expectations and log-sum-exp.

---

## A.3 AM–GM inequality

For $x_1,\dots,x_n \ge 0$,
$$
\frac{1}{n}\sum_{i=1}^n x_i
\ge
\left(\prod_{i=1}^n x_i \right)^{1/n}.
$$

This can be proved using Jensen’s inequality with $f(t) = \log t$, which is concave. AM–GM appears frequently in inequality-constrained optimisation, e.g. bounding products by sums.

---

## A.4 Hölder’s inequality (generalised Cauchy–Schwarz)

For $p,q \ge 1$ with $\frac{1}{p} + \frac{1}{q} = 1$ (conjugate exponents),
$$
\sum_{i=1}^n |x_i y_i|
\le
\left( \sum_{i=1}^n |x_i|^p \right)^{1/p}
\left( \sum_{i=1}^n |y_i|^q \right)^{1/q}.
$$

- When $p=q=2$, Hölder becomes Cauchy–Schwarz.
- Hölder underlies dual norms: the dual of $\ell_p$ is $\ell_q$.

---

## A.5 Young’s inequality

For $a,b \ge 0$ and $p,q > 1$ with $\frac{1}{p} + \frac{1}{q} = 1$,
$$
ab \le \frac{a^p}{p} + \frac{b^q}{q}.
$$

This is useful in bounding cross terms in convergence proofs.

---

## A.6 Fenchel’s inequality

Let $f$ be a convex function and let $f^*$ be its convex conjugate:
$$
f^*(y) = \sup_x (y^\top x - f(x)).
$$

Then for all $x,y$,
$$
f(x) + f^*(y) \ge y^\top x.
$$

Fenchel’s inequality is at the heart of convex duality. In fact, weak duality in Chapter 8 is essentially an application of Fenchel’s inequality.

---

## A.7 Supporting hyperplane inequality

If $f$ is convex, then for any $x$ and any $g \in \partial f(x)$,
$$
f(y) \ge f(x) + g^\top (y-x)
\quad \text{for all } y.
$$

This can be viewed as “$f$ lies above all its tangent hyperplanes,” even when it’s not differentiable. This is both a characterisation of convexity and the definition of subgradients.

---

## A.8 Summary

- Cauchy–Schwarz and Hölder bound inner products.
- Jensen shows convexity and expectation interact cleanly.
- Fenchel’s inequality is the algebra of duality.
- Supporting hyperplane inequality is the geometry of convexity.

These inequalities are used implicitly all over convex optimisation.

<!-- Throughout convex optimization theory, certain inequalities appear repeatedly to bound quantities or prove convexity. We collect a few fundamental ones that every practitioner should know, emphasizing their intuitive meaning and use cases.

**Cauchy–Schwarz Inequality:** For any vectors $x, y \in \mathbb{R}^n$

$$
|\langle x, y \rangle| \le \|x\|_2 \, \|y\|_2
$$


with equality if and only if $x$ and $y$ are collinear (one is a scalar multiple of the other). This inequality is a cornerstone of Euclidean geometry. It can be seen as a statement that the projection of $x$ onto $y$ cannot exceed the lengths product, or that the cosine of the angle between $x,y$ is at most 1. In optimization, Cauchy–Schwarz justifies many step-size bounds and duality relations. For example, in gradient descent: $f(x_{k+1}) = f(x_k) + \langle \nabla f(x_k), x_{k+1}-x_k\rangle + \cdots$ and choosing $x_{k+1}-x_k = -\alpha \nabla f(x_k)$ gives the maximal decrease in the linear approximation sense because it negates the gradient direction. Cauchy–Schwarz is also used to derive error bounds: $|g^T(x - \hat{x})| \le |g|_2 |x-\hat{x}|_2$, which plugged into optimality conditions yields, for example, primal-dual gap bounds or sensitivity analysis results.

**Triangle Inequality (Minkowski’s inequality):** For any vectors $x,y$ in a normed space,

$$
\|x + y\| \le \|x\| + \|y\|
$$


Geometrically, the direct path from 0 to $x+y$ is no longer than going from 0 to $x$ and then $x$ to $x+y$ (which is same as 0 to $y$ after shifting origin to $x$). In $\ell_2$, this is the everyday triangle rule. In $\ell_1$, it says $|x_1+y_1| + \cdots + |x_n+y_n| \le (|x_1|+\cdots+|x_n|)+(|y_1|+\cdots+|y_n|)$, obvious since term by term it's true. In $\ell_\infty$, $\max_i |x_i+y_i| \le \max_i |x_i| + \max_i |y_i|$. The triangle inequality ensures that norm balls are convex (their defining inequality is linear in two points). In optimization algorithms, the triangle inequality is often used to decompose error terms: $|x_k - \hat{x}| \le |x_k - \bar{x}| + |\bar{x} - \hat{x}|$ for some intermediate $\bar{x}$, or to prove convergence by bounding how far the current iterate is after one step plus how far that step could overshoot. Minkowski’s inequality is a generalization: for $p\ge1$, $|x+y|_p^p \le |x|_p^p + |y|_p^p +$ cross terms (which lead to the basic form above after taking pth root).

**Jensen’s Inequality:** If $f$ is convex and $x_1,\dots,x_m$ are points with weights $\lambda_i \ge 0$, $\sum \lambda_i=1$, then

$$
f\!\left(\sum_i \lambda_i x_i\right)
\le
\sum_i \lambda_i f(x_i)
$$


Equivalently, $f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$. We introduced this in the convex function chapter. It’s extremely useful for expectations and probabilistic analysis. A common application: AM–GM inequality: take $f(x) = e^x$ convex, and $x_i = \ln a_i$, $\lambda_i = 1/n$. Then Jensen gives $e^{\frac{1}{n}\sum \ln a_i} \le \frac{1}{n}\sum e^{\ln a_i}$, i.e. $(\prod_{i} a_i)^{1/n} \le \frac{1}{n}\sum a_i$. This is the statement that the geometric mean is at most the arithmetic mean. Equality holds if all $a_i$ equal (so logs equal, function is linear segment). Another: using $f(x)=x^2$ convex and weights, one shows RMS-AM: $\sqrt{\frac{1}{n}\sum a_i^2} \ge \frac{1}{n}\sum a_i$. Jensen’s inequality is often used in convex optimization proofs to move expectation inside a function or vice versa. For instance, in demonstrating convexity of a function that is defined as an expectation: $F(\theta) = \mathbb{E}_\xi[\phi(\theta,\xi)]$ is convex in $\theta$ if each $\phi(\theta,\xi)$ is convex, because $F(\lambda \theta_1+(1-\lambda)\theta_2) = \mathbb{E}[\phi(\lambda \theta_1+(1-\lambda)\theta_2,\xi)] \le \mathbb{E}[\lambda \phi(\theta_1,\xi)+(1-\lambda)\phi(\theta_2,\xi)] = \lambda F(\theta_1) + (1-\lambda) F(\theta_2)$. This is Jensen in action with probability weights. In stochastic gradient methods, one often uses Jensen to argue that the expected objective is decreased even if each step only guarantees decrease in expectation, etc.

**Hölder’s Inequality:** This generalizes Cauchy–Schwarz to $\ell_p$ and $\ell_q$ spaces. If $\frac{1}{p} + \frac{1}{q} = 1$ (with $p,q\ge1$), then for vectors (or sequences) $a,b$ of appropriate dimension,

$$
\sum_i |a_i b_i|
\le
\left(\sum_i |a_i|^p\right)^{1/p}
\left(\sum_i |b_i|^q\right)^{1/q}
$$


For example, with $p=2,q=2$, this is Cauchy–Schwarz. With $p=1,q=\infty$, it says $\sum |a_i b_i| \le (\sum |a_i|)|b|\infty$ (makes sense: $|b|\infty$ is the max component, factor it out). With $p=q=4/3$ and $q=4$, it’s something more exotic. Hölder’s inequality is very useful in analysis and dual norms: it exactly shows that the dual norm of $\ell_p$ is $\ell_q$. Indeed, take $x$ with $|x|_p=1$, then maximize $\sum x_i y_i$ over such $x$. Setting $x_i = |y_i|^{q-2} y_i / |y|_q^{q-1}$ (the extremizer found by equality case) yields $\sum_i x_i y_i = |y|q$. So $\sup{|x|_p\le1} x^T y = |y|_q$. This is why in Chapter 3 we stated dual norm formula and that $\ell_p$ dual is $\ell_q$; the proof is Hölder’s inequality. In optimization, Hölder is often used to bound sums or integrals when dealing with error terms. For instance, in optimizing integrals or in mirror descent analysis, one often gets a term like $\int g(x) h(x),dx$ and uses Hölder to split it into $|g|_p |h|_q$. In linear programming, Hölder’s inequality appears in the derivation of LP duality: maximizing $c^T x$ subject to $Ax \le b$, $x\ge0$ and its dual $b^T y$ s.t. $A^T y \ge c, y\ge0$. Weak duality follows from $c^T x \le y^T A x \le y^T b$ by $x\ge0$, $A^T y\ge c$, which is essentially an instance of Hölder: $c^T x \le (A^T y)^T x = y^T (Ax) \le y^T b$. Here we used $\sum_i c_i x_i \le \sum_i y_i (A x)_i$ since $c_i \le (A^T y)_i$ and $x_i\ge0$. That step is like $c_i x_i \le y_i (A x)_i$ summing yields the inequality. It’s a discrete form but conceptually related to Hölder’s type argument of splitting products.

**Other inequalities:** There are many more (Markov, Chebyshev, Chernoff – probabilistic; Young’s inequality for products; Lipschitz implies bounded differences, etc.), but the ones above are most directly tied to convex analysis. Another notable one is Fenchel’s inequality: for any convex $f$ and its convex conjugate $\hat{f}$, $f(x) + \hat{f}(y) \ge x^T y$. This is essentially a consequence of the definition of conjugate (supremum form) and is used in deriving duality gaps.


To tie it together: suppose we have a suboptimal point $x$ and optimum $x^*$. Then using a subgradient $g\in\partial f(x)$, convexity (first-order condition) gives $f(x) - f(x^*) \le g^T(x - x^*)$. Now Cauchy–Schwarz (or Hölder for appropriate norm choices) gives $g^T(x-x^*) \le |g| |x-x^*|$. Thus


$$
f(x) - f(x^*) \le \|g\| \, \|x - x^*\|
$$


This inequality is often used: it relates suboptimality to a dual norm of subgradient and distance to solution. If we have a bound on $|g|$ (say by optimality conditions or Lipschitz continuity), then we get a residual bound on $f(x)-f(x^*)$. This is one small example of how combining convexity with Cauchy–Schwarz yields useful insights. -->