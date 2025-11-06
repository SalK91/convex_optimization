# Chapter 14: Optimization Algorithms for Inequality-Constrained Problems

In many real-world optimization problems, not all solutions are allowed — we often have both equality and inequality constraints that define a feasible region. In this chapter, we study algorithms that handle inequality constraints efficiently, focusing on the logarithmic barrier method, which is foundational for interior-point methods.



## 14.1 Inequality-Constrained Minimization

We consider the general inequality- and equality-constrained optimization problem:

$$
\begin{aligned}
\text{minimize} \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \leq 0, \quad i = 1, \dots, m, \\
& A x = b.
\end{aligned}
$$

We assume:

- Each $f_i$ is convex and twice continuously differentiable.  
- $A \in \mathbb{R}^{p \times n}$ has full rank, $\text{rank}(A) = p$.  
- The optimal value $p^*$ is finite and attained.  
- The problem is strictly feasible, meaning there exists a point $\bar{x}$ satisfying:

$$
\bar{x} \in \text{dom } f_0, \quad f_i(\bar{x}) < 0, \; i=1,\dots,m, \quad A \bar{x} = b.
$$

Under these assumptions, strong duality holds and the dual optimum is attained.

 

### Examples

Many common optimization problems fall under this framework:

- LP (Linear Program): $f_0(x) = c^T x$, $f_i(x) = a_i^T x - b_i$  
- QP (Quadratic Program): $f_0(x) = \tfrac{1}{2} x^T P x + q^T x$  
- QCQP (Quadratically Constrained Quadratic Program)  
- GP (Geometric Program)  
- Entropy maximization with linear inequality constraints:

$$
\begin{aligned}
\text{minimize} \quad & \sum_{i=1}^n x_i \log x_i \\
\text{subject to} \quad & F x \leq g, \\
& A x = b,
\end{aligned}
$$

where $\text{dom } f_0 = \mathbb{R}_{++}^n$.

 

## 14.2 Reformulation via Indicator Functions

We can reformulate the constrained problem as an unconstrained one using an *indicator function* for the feasible set.

Define the indicator of the nonpositive orthant:

$$
I_-(u) =
\begin{cases}
0, & u \leq 0, \\
+\infty, & u > 0.
\end{cases}
$$

Then the constrained problem is equivalent to:

$$
\begin{aligned}
\text{minimize} \quad & f_0(x) + \sum_{i=1}^m I_-(f_i(x)) \\
\text{subject to} \quad & A x = b.
\end{aligned}
$$

However, this formulation is not smooth — $I_-(u)$ is discontinuous.  
To apply smooth optimization methods, we approximate $I_-$ using a logarithmic barrier.

 
## 14.3 Logarithmic Barrier Approximation

We replace $I_-(u)$ by a smooth function:

$$
\Phi(u) = -\frac{1}{t} \log(-u),
$$

where $t > 0$ is a barrier parameter controlling the accuracy of the approximation.  
The resulting logarithmic barrier problem is:

$$
\begin{aligned}
\text{minimize} \quad & f_0(x) - \frac{1}{t} \sum_{i=1}^m \log(-f_i(x)) \\
\text{subject to} \quad & A x = b.
\end{aligned}
$$

- For small $t$, the barrier term is significant — it keeps iterates strictly inside the feasible region.  
- As $t \to \infty$, the approximation becomes exact, and the solution approaches the true constrained optimum.

Thus, the original constrained problem is transformed into a sequence of smooth equality-constrained problems.

 
## 14.4 Logarithmic Barrier Function

Define the logarithmic barrier function:

$$
\phi(x) = -\sum_{i=1}^m \log(-f_i(x)),
$$

with domain

$$
\text{dom } \phi = \{ x \mid f_i(x) < 0, \; i = 1, \dots, m \}.
$$

### Properties

- $\phi(x)$ is convex, since it is a composition of convex and decreasing functions.  
- It is twice continuously differentiable, with:

$$
\nabla \phi(x) = \sum_{i=1}^m \frac{1}{-f_i(x)} \nabla f_i(x),
$$

and

$$
\nabla^2 \phi(x) =
\sum_{i=1}^m \frac{1}{f_i(x)^2} \nabla f_i(x) \nabla f_i(x)^T
+ \sum_{i=1}^m \frac{1}{-f_i(x)} \nabla^2 f_i(x).
$$

 
## 14.5 Central Path

For each $t > 0$, define $x^*(t)$ as the minimizer of the barrier problem:

$$
\begin{aligned}
\text{minimize} \quad & t f_0(x) + \phi(x) \\
\text{subject to} \quad & A x = b.
\end{aligned}
$$

The set $\{ x^*(t) \mid t > 0 \}$ is called the central path.

As $t \to \infty$, $x^*(t)$ approaches the solution $x^*$ of the original constrained problem.

### Example: Central Path for a Linear Program

For an LP:

$$
\begin{aligned}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & a_i^T x \leq b_i, \quad i = 1, \dots, m,
\end{aligned}
$$

the central path consists of points $x^*(t)$ where hyperplanes $c^T x = c^T x^*(t)$ are tangent to the level sets of the barrier $\phi(x)$.

 
## 14.6 Interpretation via KKT Conditions

At any point on the central path $(x^*(t), \lambda^*(t), v^*(t))$, the following hold:

1. Primal feasibility:  
   $f_i(x^*(t)) \leq 0, \; i = 1, \dots, m$, and $A x^*(t) = b$.

2. Dual feasibility:  
   $\lambda_i^*(t) \geq 0$.

3. Approximate complementary slackness:  
   $-\lambda_i^*(t) f_i(x^*(t)) = \frac{1}{t}$, for $i = 1, \dots, m$.

4. Stationarity (gradient condition):
   $$
   \nabla f_0(x^*(t)) + \sum_{i=1}^m \lambda_i^*(t) \nabla f_i(x^*(t)) + A^T v^*(t) = 0.
   $$

The only difference between these and the exact KKT conditions is the relaxation of complementary slackness —  
instead of $\lambda_i f_i(x) = 0$, we have $\lambda_i f_i(x) = -1/t$.

As $t \to \infty$, the approximate KKT conditions converge to the true KKT conditions.

---

## 14.7 Force Field Interpretation

To build geometric intuition, consider the centering problem (no equality constraints):

$$
\text{minimize} \quad t f_0(x) - \sum_{i=1}^m \log(-f_i(x)).
$$

- The term $f_0(x)$ represents a “potential energy” whose gradient is the external force:
  $$
  F_0(x) = -\nabla f_0(x).
  $$

- Each constraint contributes a repulsive force preventing violation:
  $$
  F_i(x) = \frac{1}{f_i(x)} \nabla f_i(x).
  $$

At equilibrium (the central point $x^*(t)$), forces balance perfectly:

$$
F_0(x^*(t)) + \sum_{i=1}^m F_i(x^*(t)) = 0.
$$

The solution thus represents a balance between the pull of minimizing $f_0$ and the repulsion from the boundaries $f_i(x) = 0$.

---

## 14.8 Barrier Method Algorithm

The barrier method solves a sequence of equality-constrained barrier subproblems, gradually increasing $t$ to approach the true optimum.

---

Given: strictly feasible $x$, initial barrier parameter $t := t^{(0)} > 0$, multiplier $\mu > 1$, tolerance $\varepsilon > 0$.

Repeat:
1. Centering Step:  
   Compute $x^*(t)$ by minimizing  
   $$
   f_0(x) - \frac{1}{t} \sum_{i=1}^m \log(-f_i(x))
   $$
   subject to $A x = b$.  
   (This is typically done with Newton’s method.)

2. Update:  
   $x := x^*(t)$.

3. Stopping Criterion:  
   Quit if $\frac{m}{t} < \varepsilon$.

4. Increase Barrier Parameter:  
   $t := \mu t$.

---

### Notes on Practical Implementation

- Each centering step (inner loop) is an equality-constrained Newton method.  
- The outer loop gradually increases $t$, making the barrier steeper.  
- A larger $\mu$ means fewer outer iterations but more Newton steps per inner problem.  
  Typical values: $\mu = 10$ or $\mu = 20$.

The method terminates when:

$$
f_0(x) - p^* \leq \varepsilon,
$$

which follows directly from the bound $f_0(x^*(t)) - p^* \leq \frac{m}{t}$.

 

### Convergence and Intuition

- Early iterations focus on feasibility and centrality — staying deep within the feasible region.  
- Later iterations approach the true optimum, as the barrier term becomes negligible.  
- The iterates $x^*(t)$ always remain strictly feasible, hence the method name: *interior-point*.
