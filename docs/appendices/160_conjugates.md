# Appendix D: Convex Conjugates and Fenchel Duality

Convex conjugates and Fenchel duality form the functional heart of convex analysis.  
They provide a powerful unifying view of optimization by connecting geometry, algebra, and duality.  

- Convex conjugates convert a function into its “slope-space” representation — capturing its tightest linear overestimates.  
- Fenchel duality uses these conjugates to derive dual optimization problems that often reveal structure, efficiency, or interpretability hidden in the primal form.  

Together, they form the bridge between the geometry of convex sets (Appendix C) and the duality theory of optimization (Chapter 8).

 
## D.1 Intuitive Picture

Imagine a convex function \(f(x)\) drawn as a bowl in space.  
Each point \(y\) defines a line (or hyperplane) of slope \(y\):
\[
x \mapsto \langle y, x \rangle - b.
\]
The convex conjugate \(f^*(y)\) is the smallest height \(b\) such that this line always stays above \(f(x)\).  
In other words:

> \(f^*(y)\) measures the tightest linear overestimate of \(f\) in direction \(y\).

So \(f^*\) encodes how “steep” \(f\) can be in every direction — it transforms the geometry of \(f\) into a new convex function on slope-space.

 
## D.2 Definition and Key Properties

Let \(f : \mathbb{R}^n \to \mathbb{R}\cup\{+\infty\}\) be a proper convex function.  
Its convex (Fenchel) conjugate is
\[
f^*(y) = \sup_{x \in \mathbb{R}^n} \big( \langle y, x \rangle - f(x) \big).
\]

Interpretation
- \(y\): a slope or linear functional.
- The supremum seeks the largest gap between the linear function \(\langle y,x\rangle\) and the graph of \(f\).
- \(f^*(y)\) is always convex, even if \(f\) isn’t strictly convex.

### Fundamental Identities

1. Fenchel–Young inequality
   \[
   \langle y,x\rangle \le f(x) + f^*(y),
   \]
   with equality iff \(y \in \partial f(x)\).

2. Biconjugation
   \[
   f^{} = f \quad \text{if $f$ is proper, convex, and lower semicontinuous.}
   \]
   This tells us the conjugate transform loses no information for convex functions.

3. Order reversal
   \(f \le g \;\Rightarrow\; f^* \ge g^*\).

4. Scaling and shift
   - \((f + a)^*(y) = f^*(y) - a\),
   - \((\alpha f)^*(y) = \alpha f^*(y/\alpha)\) for \(\alpha>0.\)

---

## D.3 Canonical Examples

| Function \(f(x)\) | Conjugate \(f^*(y)\) | Notes |
|--------------------|----------------------|-------|
| \( \tfrac{1}{2}\|x\|_2^2 \) | \( \tfrac{1}{2}\|y\|_2^2 \) | Self-conjugate quadratic |
| \( \|x\|_1 \) | \( \delta_{\{\|y\|_\infty \le 1\}}(y) \) | Dual norm indicator |
| \( \delta_C(x) \) | \( \sigma_C(y)=\sup_{x\in C}\langle y,x\rangle \) | Support function of set \(C\) |
| \( e^x \) | \( y\log y - y,\, y>0 \) | Appears in entropy and KL-divergence |

These examples illustrate how conjugation connects:
- Norms ↔ dual norms,  
- Sets ↔ support functions,  
- Exponentials ↔ entropy,  
- Quadratics ↔ themselves.

 

## D.4 Geometric Interpretation

- Each point on \(f\) has a tangent hyperplane whose slope is a subgradient.  
- The collection of all such hyperplanes forms the epigraph of \(f^*\).  
- The transformation \(f \mapsto f^*\) swaps the roles of “position” and “slope”:  
  convex geometry ↔ supporting hyperplanes.

Visually:  
- \(f\) describes a bowl in \((x,t)\)-space.  
- \(f^*\) describes the envelope of tangent planes to that bowl.

 

## D.5 From Conjugates to Duality — Fenchel Duality

Many convex optimization problems can be written as
\[
\min_x \; f(x) + g(Ax),
\]
where \(f,g\) are convex and \(A\) is linear.  
Fenchel duality uses conjugates to build a dual problem in terms of \(f^*\) and \(g^*\).

### The Fenchel Dual Problem
\[
\max_y \; -f^*(A^\top y) - g^*(-y).
\]

Interpretation
- \(y\) is the dual variable (similar to Lagrange multipliers).  
- The dual objective collects the best linear lower bounds on the primal cost.

 

## D.6 Weak and Strong Duality

- Weak duality: For any \(x,y\),
  \[
  f(x)+g(Ax) \ge -f^*(A^\top y) - g^*(-y).
  \]
  So the dual value always underestimates the primal value.

- Strong duality:  
  If \(f,g\) are closed convex and a mild constraint qualification holds (e.g. Slater’s condition — existence of strictly feasible \(x\)), then
  \[
  \min_x [f(x)+g(Ax)] = \max_y [-f^*(A^\top y) - g^*(-y)].
  \]

At the optimum:
\[
A^\top y^* \in \partial f(x^*), 
\qquad
-y^* \in \partial g(Ax^*).
\]
These are the Fenchel–KKT conditions, directly linking primal and dual subgradients.

 

## D.7 Illustrative Examples

### (a) Linear Programming

Primal:
\[
\min_{x \ge 0} c^\top x \quad \text{s.t. } Ax = b.
\]

Take  
\(f(x) = c^\top x + \delta_{\{x\ge0\}}(x)\),  
\(g(z)=\delta_{\{z=b\}}(z)\).

Then
\[
f^*(y) = \delta_{\{y \le c\}}(y),
\qquad
g^*(y) = b^\top y.
\]

Dual:
\[
\max_y \; b^\top y \quad \text{s.t. } A^\top y \le c,
\]
which is the standard LP dual.

 

### (b) Quadratic + Set Constraint

Primal:
\[
\min_x \tfrac{1}{2}\|x\|_2^2 + \delta_C(x).
\]

Then
\[
f^*(y)=\tfrac{1}{2}\|y\|_2^2, \qquad g^*(y)=\sigma_C(y),
\]
so the dual is
\[
\max_y -\tfrac{1}{2}\|y\|_2^2 - \sigma_C(y).
\]
Optimality gives \(x^*=y^*\), the projection condition in Euclidean geometry.

 

## D.8 Practical Significance

| Area | How Fenchel Duality Appears |
|------|------------------------------|
| Optimization theory | Derives general dual problems beyond inequality constraints. |
| Algorithm design | Basis for primal–dual and splitting methods (ADMM, Chambolle–Pock, Mirror Descent). |
| Geometry | Dual problem finds the “best supporting hyperplane” to the primal epigraph. |
| Machine Learning | Loss–regularizer pairs (hinge ↔ clipped loss, logistic ↔ log-sum-exp) often form conjugate pairs. |
| Proximal operators | Linked via Moreau identity:  \(\mathrm{prox}_{f^*}(y) = y - \mathrm{prox}_f(y)\). |

 

## D.9 Conceptual Unification

Convex conjugates and Fenchel duality tie together nearly every idea in this book:

- From geometry: support functions, projections, subgradients (Appendices B–C).  
- From analysis: inequalities like Fenchel’s and Jensen’s (Appendix A).  
- From optimization: Lagrange duality, KKT, and strong duality (Chapters 7–8).  
- From computation: proximal, ADMM, and mirror-descent algorithms (Chapters 9–10).

Together, they show that convex optimization is self-dual: every convex structure has an equally convex mirror image.

 
## D.10 Summary and Takeaways

- The convex conjugate \(f^*\) expresses \(f\) through its linear support planes.  
- The Fenchel–Young inequality connects primal variables and dual slopes.  
- Fenchel duality constructs a systematic dual problem using these conjugates.  
- Under mild conditions, strong duality holds, and subgradients link primal and dual optima.  
- These ideas underpin most modern optimization algorithms and geometric interpretations of convexity.

---

Further Reading

- Rockafellar, R. T. (1970). *Convex Analysis*. Princeton UP.  
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*, Chs. 3 & 5.  
- Bauschke, H. H., & Combettes, P. L. (2017). *Convex Analysis and Monotone Operator Theory*.  
- Hiriart-Urruty, J.-B., & Lemaréchal, C. (2001). *Fundamentals of Convex Analysis*.  
