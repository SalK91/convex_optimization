# Chapter 8: Lagrange Multipliers and the KKT Framework

We now have the ingredients for understanding optimality in convex optimization:

- convex functions define well-behaved objectives,
- convex sets describe feasible regions,
- gradients and subgradients encode descent directions.

This chapter unifies these ideas. We begin with unconstrained minimization and then incorporate equality and inequality constraints. The resulting system of conditions—the Karush–Kuhn–Tucker (KKT) conditions—is the central optimality framework for constrained convex optimization.

In constrained problems, the gradient of the objective cannot vanish freely. Instead, it must be balanced by “forces’’ coming from the constraints. Lagrange multipliers measure these forces, and the KKT conditions express this balance algebraically and geometrically.


## 8.1 Unconstrained Convex Minimization

Consider the problem
$$
\min_x f(x),
$$
where $f$ is convex and differentiable.

Gradient descent iteratively updates
$$
x^{(k+1)} = x^{(k)} - \alpha_k \nabla f(x^{(k)}),
$$
with step size $\alpha_k > 0$.

Intuition:

- Moving opposite the gradient decreases $f$.
- If the gradient is Lipschitz continuous and the step size is small enough ($\alpha_k \le 1/L$), then gradient descent converges to a global minimizer.
- If $f$ is *strongly convex*, the minimizer is unique and convergence is faster (linear rate with an appropriate step size).

In machine learning, this is the foundation of back-propagation and weight training: each update follows the negative gradient of the loss.


## 8.2 Equality-Constrained Problems and Lagrange Multipliers

Now consider minimizing $f$ subject to equality constraints:
$$
\begin{array}{ll}
\text{minimize} & f(x) \\
\text{subject to} & h_j(x) = 0,\quad j = 1,\dots,p.
\end{array}
$$

Define the Lagrangian
$$
L(x, \lambda) = f(x) + \sum_{j=1}^p \lambda_j h_j(x),
$$
where $\lambda = (\lambda_1,\dots,\lambda_p)$ are the Lagrange multipliers.

Under differentiability and regularity assumptions, a point $x^*$ is optimal only if:

1. Primal feasibility
   $$
   h_j(x^*) = 0,\quad \forall j.
   $$

2. Stationarity
   $$
   \nabla f(x^*) + \sum_{j=1}^p \lambda_j^* \nabla h_j(x^*) = 0.
   $$

Geometric meaning:

- The feasible set $ \{x : h_j(x)=0\} $ is typically a smooth manifold.
- At an optimum, the gradient of the objective must be orthogonal to all feasible directions.
- The multipliers $\lambda_j^*$ weight the constraint normals to exactly cancel the objective’s gradient.

In other words, the objective tries to decrease, the constraints push back, and at the optimum these forces balance.

 
## 8.3 Inequality Constraints and the KKT Conditions

Now consider the general convex problem:
$$
\begin{array}{ll}
\text{minimize} & f(x) \\
\text{subject to} 
 & g_i(x) \le 0,\quad i=1,\dots,m, \\
 & h_j(x) = 0,\quad j=1,\dots,p.
\end{array}
$$

Form the Lagrangian
$$
L(x,\lambda,\mu) 
= f(x) 
+ \sum_{j=1}^p \lambda_j h_j(x)
+ \sum_{i=1}^m \mu_i g_i(x),
$$
with:

- $ \lambda_j \in \mathbb{R} $ (equality multipliers),
- $ \mu_i \ge 0 $ (inequality multipliers).

A point $x^*$ with multipliers $(\lambda^*,\mu^*)$ satisfies the KKT conditions:

### 1. Primal feasibility
$$
g_i(x^*) \le 0,\quad \forall i,
\qquad
h_j(x^*) = 0,\quad \forall j.
$$

### 2. Dual feasibility
$$
\mu_i^* \ge 0,\quad \forall i.
$$

### 3. Stationarity
$$
\nabla f(x^*) 
+ \sum_{j=1}^p \lambda_j^* \nabla h_j(x^*)
+ \sum_{i=1}^m \mu_i^* \nabla g_i(x^*)
= 0.
$$

### 4. Complementary slackness
$$
\mu_i^*\, g_i(x^*) = 0, \quad i=1,\dots,m.
$$

Complementary slackness expresses a clear dichotomy:

- If constraint $g_i(x) \le 0$ is inactive (strictly $<0$), then it applies no force: $\mu_i^* = 0$.
- If a constraint is active at the boundary, it may exert a force: $\mu_i^* > 0$, and then $g_i(x^*) = 0$.

Only active constraints can push back against the objective.

## 8.4 Slater’s Condition — Guaranteeing Strong Duality

The KKT conditions always provide *necessary* conditions for optimality. For them to also be *sufficient* (and to guarantee zero duality gap), the problem must satisfy a regularity condition.

For convex problems with convex $g_i$ and affine $h_j$, Slater’s condition holds if there exists a strictly feasible point:
$$
\exists\, x^{\text{slater}}:
\quad h_j(x^{\text{slater}})=0,\ \forall j,
\qquad
g_i(x^{\text{slater}}) < 0,\ \forall i.
$$

Interpretation:

- The feasible region contains an interior point.
- The constraints are not “tight” everywhere.
- The geometry is rich enough for supporting hyperplanes to behave nicely.

When Slater’s condition holds:

1. Strong duality holds:  
   $$
   p^* = d^*.
   $$

2. The dual optimum is attained.

3. The KKT conditions are both necessary and sufficient for optimality.

### Duality gap

For a primal problem with optimum $p^*$ and its dual with optimum $d^*$, the duality gap is
$$
p^* - d^* \ge 0.
$$

- A strictly positive gap indicates structural degeneracy or failure of constraint qualification.
- Slater’s condition ensures the gap is zero.

This link between geometry (interior feasibility) and algebra (zero gap) is fundamental.

---

## 8.5 Geometric and Physical Interpretation

The KKT conditions describe an equilibrium of forces:

- The objective gradient pushes the point in the direction of steepest decrease.
- Active constraints push back through normal vectors scaled by multipliers.
- At optimality, these forces exactly cancel.

Physically:

- Lagrange multipliers are “reaction forces’’ keeping a system on the constraint surface.
- In economics, they are “shadow prices’’ indicating how much the objective would improve if a constraint were relaxed.
- Geometrically, the stationarity condition means the objective and the active constraints share a supporting hyperplane at the optimum.

KKT theory unifies all earlier ideas—convexity, gradients/subgradients, feasible regions, tangent and normal cones—into one clean, general optimality framework.


