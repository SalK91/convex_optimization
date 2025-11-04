# Appendix A: Common Inequalities and Identities

This appendix collects important inequalities used throughout convex analysis and optimisation. These are the “algebraic tools” you reach for in proofs, optimality arguments, and convergence analysis (Boyd and Vandenberghe, 2004; Hiriart-Urruty and Lemaréchal, 2001).

 
## A.1 Cauchy–Schwarz inequality

For any $x,y \in \mathbb{R}^n$,
$$
|x^\top y| \le \|x\|_2 \, \|y\|_2.
$$

Equality holds if and only if $x$ and $y$ are linearly dependent.

Consequences:

- Defines the notion of angle between vectors.
- Justifies dual norms.

 
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

 

## A.3 AM–GM inequality

For $x_1,\dots,x_n \ge 0$,
$$
\frac{1}{n}\sum_{i=1}^n x_i
\ge
\left(\prod_{i=1}^n x_i \right)^{1/n}.
$$

This can be proved using Jensen’s inequality with $f(t) = \log t$, which is concave. AM–GM appears frequently in inequality-constrained optimisation, e.g. bounding products by sums.

 
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

 

## A.5 Young’s inequality

For $a,b \ge 0$ and $p,q > 1$ with $\frac{1}{p} + \frac{1}{q} = 1$,
$$
ab \le \frac{a^p}{p} + \frac{b^q}{q}.
$$

This is useful in bounding cross terms in convergence proofs.

 

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

 

## A.7 Supporting hyperplane inequality

If $f$ is convex, then for any $x$ and any $g \in \partial f(x)$,
$$
f(y) \ge f(x) + g^\top (y-x)
\quad \text{for all } y.
$$

This can be viewed as “$f$ lies above all its tangent hyperplanes,” even when it’s not differentiable. This is both a characterisation of convexity and the definition of subgradients.

 

## A.8 Summary

- Cauchy–Schwarz and Hölder bound inner products.
- Jensen shows convexity and expectation interact cleanly.
- Fenchel’s inequality is the algebra of duality.
- Supporting hyperplane inequality is the geometry of convexity.

These inequalities are used implicitly all over convex optimisation.
