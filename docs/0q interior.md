# Motivation

We begin with the **linear programming (LP)** problem:

$$
\min_x \; c^\top x \quad \text{subject to} \quad A x \le b
$$

Equivalent **standard form**:

$$
\min_x \; c^\top x \quad \text{subject to} \quad A x = b, \quad x \ge 0
$$

### Why is LP so powerful?

- LP appears in **resource allocation**, **scheduling**, **routing**, and many **combinatorial optimization problems**.
- If an optimal solution exists, it is always located at a **vertex (extreme point)** of the polytope defined by $A x \le b$.
- The feasible region $\mathcal{P} = \{ x \in \mathbb{R}^n \mid A x \le b \}$ is a **polyhedron**, which typically has **many corners**.


## Extreme Points and Combinatorial Explosion

Let:

$$
\mathcal{P} = \{x \mid A x \le b\}
$$

- With $m$ inequality constraints in $\mathbb{R}^n$, the number of extreme points can be **exponential in $m$**.
- Simple example: **hypercube** $[0,1]^n$ — it has $2^n$ vertices.
- Thus, **corner-based search** (like **Simplex**) may require **visiting many corners** in the worst case.


## Classical Methods: Simplex and Projected Gradient

### 1. Projected Gradient Descent for LP

Minimize a quadratic surrogate:

$$
\min_x \|x - x_k\|^2 \quad \text{subject to} \quad A x \le b
$$

Main challenge: **projection onto polytope** is expensive.

### 2. Simplex Method

- Moves **from corner to corner**.
- Efficient in practice, but **not polynomial-time guaranteed**.
- **Geometrically:** slides along edges of polytope — stays on the **boundary**.


## Motivation for Interior Point Methods

- **Simplex walks on the boundary**, potentially traversing exponentially many vertices.
- **Gradient projection struggles** with feasibility projection.
- **Interior Point Methods take a different path**:
  - They **stay strictly inside** the polytope.
  - They follow a **central trajectory** rather than bouncing between corners.
  - They use **Newton’s method** on a **barrier-regularized objective**, giving both **feasibility** and **fast convergence**.

n

# Interior Point Methods — Barrier and Newton Fusion

We now revisit the constrained problem:

$$
\min_x \; f(x) \quad \text{subject to } A x \le b
$$

Introduce the **log-barrier** for each constraint:

$$
B(x) = -\sum_{i=1}^m \log(b_i - a_i^\top x)
$$

Construct the **barrier objective**:

$$
\Phi_t(x) = t f(x) + B(x), \quad t > 0
$$

> Interpretation: As we increase $t$, we **push harder toward the true optimum**, while the barrier **keeps us in the interior**.


## Newton Method Applied to Barrier Objective

Compute:

$$
\nabla \Phi_t(x) = t \nabla f(x) + \sum_{i=1}^m \frac{1}{b_i - a_i^\top x} a_i
$$

$$
\nabla^2 \Phi_t(x) = t \nabla^2 f(x) + \sum_{i=1}^m \frac{1}{(b_i - a_i^\top x)^2} a_i a_i^\top
$$

Newton Direction:

$$
d = - \left[\nabla^2 \Phi_t(x)\right]^{-1} \nabla \Phi_t(x)
$$

Update with damping to keep strict feasibility:

$$
x_{k+1} = x_k + \lambda_k d \quad \text{such that } A x_{k+1} < b
$$

 
## Interior Point Algorithm Flow

```
Given strictly feasible x₀ and initial t:
repeat:
    Solve the barrier problem: minimize Φₜ(x) using damped Newton
    Increase parameter t ← μ t  (μ > 1)
until m / t < ε   # duality gap condition
