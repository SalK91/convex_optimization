# Chapter 13: Optimization Algorithms for Equality-Constrained Problems

In many optimization problems, we are not free to choose any $x \in \mathbb{R}^n$; instead, we must satisfy a set of equality constraints of the form:

$$
A x = b,
$$

where $A \in \mathbb{R}^{p \times n}$ and $b \in \mathbb{R}^p$.  
Such problems arise naturally in engineering design, resource allocation, and systems governed by conservation laws.



## 13.1 Equality-Constrained Minimization

We consider the general equality-constrained smooth optimization problem:

$$
\begin{aligned}
\text{minimize} \quad & f(x) \\
\text{subject to} \quad & A x = b,
\end{aligned}
$$

where:

- $f: \mathbb{R}^n \to \mathbb{R}$ is convex and twice continuously differentiable,  
- $A$ has full row rank, i.e., $\text{rank}(A) = p$,  
- and the optimum value $p^*$ is finite and attained.


### Optimality Conditions

The point $x^*$ is optimal if and only if there exists a Lagrange multiplier vector $v^*$ such that:

$$
\nabla f(x^*) + A^T v^* = 0, \qquad A x^* = b.
$$

These are the Karush–Kuhn–Tucker (KKT) conditions for equality-constrained optimization.

Intuitively:

- $\nabla f(x^*)$ represents the gradient (direction of steepest ascent) of the objective.
- $A^T v^*$ represents a correction term ensuring that we stay within the feasible set $A x = b$.
- At the optimum, the projected gradient along feasible directions is zero.



## 13.2 Equality-Constrained Quadratic Minimization

A particularly important case is when $f$ is quadratic:

$$
f(x) = \tfrac{1}{2} x^T P x + q^T x + r,
$$

where $P \in \mathbb{S}_+^n$ (symmetric positive semidefinite).  
Then $\nabla f(x) = P x + q$, and the optimality conditions become the linear system:

$$
\begin{bmatrix}
P & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
x^* \\ v^*
\end{bmatrix}
=
\begin{bmatrix}
- q \\ b
\end{bmatrix}.
$$

This is known as the KKT system for the quadratic program.



### The KKT Matrix

The block matrix

$$
K = 
\begin{bmatrix}
P & A^T \\
A & 0
\end{bmatrix}
$$

is called the KKT matrix.  
It is nonsingular (and thus the problem has a unique solution) if and only if:

$$
A x = 0, \, x \neq 0 \quad \Rightarrow \quad x^T P x > 0,
$$

i.e., $P$ is positive definite on the nullspace of $A$.  
An equivalent condition is that:

$$
P + A^T A \succ 0.
$$



## 13.3 Eliminating Equality Constraints

Sometimes it is convenient to eliminate the equality constraints by parameterizing the feasible set.

The feasible set $\{x \mid A x = b\}$ can be expressed as:

$$
x = F z + \hat{x}, \quad z \in \mathbb{R}^{n-p},
$$

where:

- $\hat{x}$ is a particular solution of $A x = b$,
- $F \in \mathbb{R}^{n \times (n-p)}$ is a matrix whose columns form a basis for the nullspace of $A$ (so $A F = 0$).

Substituting this into the objective gives an unconstrained problem:

$$
\min_z \; f(F z + \hat{x}).
$$

Once we solve for $z^*$, we recover:

$$
x^* = F z^* + \hat{x}, \quad v^* = - (A A^T)^{-1} A \nabla f(x^*).
$$



### Example: Optimal Resource Allocation

Suppose we allocate resources $x_i \in \mathbb{R}$ to $n$ agents, with cost functions $f_i(x_i)$ for each agent $i$.

We have a total resource constraint:

$$
x_1 + x_2 + \dots + x_n = b.
$$

The problem is:

$$
\begin{aligned}
\text{minimize} \quad & f_1(x_1) + \cdots + f_n(x_n) \\
\text{subject to} \quad & x_1 + \cdots + x_n = b.
\end{aligned}
$$

To eliminate the constraint, we express $x_n$ as:

$$
x_n = b - (x_1 + \cdots + x_{n-1}),
$$

and define:

$$
\hat{x} = b e_n, \quad
F = 
\begin{bmatrix}
I_{n-1} \\
- \mathbf{1}^T
\end{bmatrix} \in \mathbb{R}^{n \times (n-1)}.
$$

The reduced problem is:

$$
\min_{x_1, \ldots, x_{n-1}} f_1(x_1) + \cdots + f_{n-1}(x_{n-1}) + f_n(b - x_1 - \cdots - x_{n-1}).
$$



## 13.4 Newton’s Method for Equality-Constrained Problems

We now extend Newton’s method to handle equality constraints.

At a feasible point $x$ (i.e., $A x = b$), the Newton step $\Delta x_{\text{nt}}$ is obtained by solving:

$$
\begin{bmatrix}
\nabla^2 f(x) & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
v \\ w
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(x) \\ 0
\end{bmatrix}.
$$

The Newton step is $\Delta x_{\text{nt}} = v$.



### Interpretation

The step $\Delta x_{\text{nt}}$ solves the second-order approximation of the equality-constrained problem:

$$
\begin{aligned}
\text{minimize} \quad & \hat{f}(x + v)
= f(x) + \nabla f(x)^T v + \tfrac{1}{2} v^T \nabla^2 f(x) v \\
\text{subject to} \quad & A(x + v) = b.
\end{aligned}
$$

The equations follow from linearizing the KKT conditions:

$$
\nabla f(x + v) + A^T w \approx \nabla f(x) + \nabla^2 f(x) v + A^T w = 0, \quad
A(x + v) = b.
$$



## 13.5 Newton Decrement and Stopping Criterion

The Newton decrement measures proximity to the optimum:

$$
\lambda(x) =
(\Delta x_{\text{nt}}^T \nabla^2 f(x) \Delta x_{\text{nt}})^{1/2}
=
(-\nabla f(x)^T \Delta x_{\text{nt}})^{1/2}.
$$

It provides a local estimate of suboptimality:

$$
f(x) - f^* \approx \frac{\lambda(x)^2}{2}.
$$

The directional derivative along the Newton direction is:

$$
\left.\frac{d}{dt} f(x + t \Delta x_{\text{nt}})\right|_{t=0} = -\lambda(x)^2.
$$

A common stopping condition is $\lambda(x)^2 / 2 \leq \varepsilon$.



## 13.6 Newton’s Method with Equality Constraints

A practical algorithm:



Given: feasible starting point $x \in \text{dom } f$ with $A x = b$, and tolerance $\varepsilon > 0$.

Repeat:
1. Compute the Newton step and decrement $\Delta x_{\text{nt}}, \lambda(x)$.  
2. Stopping criterion: if $\lambda^2 / 2 \leq \varepsilon$, quit.  
3. Line search: choose step size $t$ via backtracking line search.  
4. Update: $x := x + t \Delta x_{\text{nt}}$.



This is a feasible descent method: all iterates satisfy $A x = b$ and $f(x^{(k+1)}) < f(x^{(k)})$.  
It is also affine invariant — the algorithm behaves identically under affine transformations of the variables.



## 13.7 Newton’s Method and Elimination

If we have already eliminated the constraints, Newton’s method can be directly applied to the reduced function:

$$
\tilde{f}(z) = f(F z + \hat{x}), \quad A \hat{x} = b, \quad A F = 0.
$$

Unconstrained Newton’s method for $\tilde{f}(z)$ generates iterates $z^{(k)}$, and corresponding:

$$
x^{(k)} = F z^{(k)} + \hat{x}.
$$

Thus, the equality-constrained Newton iterates correspond exactly to unconstrained Newton iterates on the reduced problem.



## 13.8 Newton Step at Infeasible Points

If $x$ is not feasible (i.e., $A x \neq b$), we can generalize to the infeasible Newton step.

Define the primal-dual residual:

$$
r(y) = 
\begin{bmatrix}
\nabla f(x) + A^T v \\
A x - b
\end{bmatrix}, \quad y = (x, v).
$$

Linearizing $r(y) = 0$ around $y$ gives:

$$
r(y + \Delta y) \approx r(y) + D r(y) \Delta y = 0.
$$

Hence, the Newton step $\Delta y = (\Delta x_{\text{nt}}, \Delta v_{\text{nt}})$ satisfies:

$$
\begin{bmatrix}
\nabla^2 f(x) & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x_{\text{nt}} \\ \Delta v_{\text{nt}}
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(x) + A^T v \\ A x - b
\end{bmatrix}.
$$

This is known as the primal-dual Newton step.



## 13.9 Infeasible-Start Newton Method

A more general algorithm that does not require feasibility at the start.



Given: initial $(x, v)$, tolerance $\varepsilon > 0$, constants $\alpha \in (0, 1/2)$, $\beta \in (0, 1)$.

Repeat:
1. Compute primal-dual Newton step $(\Delta x_{\text{nt}}, \Delta v_{\text{nt}})$.  
2. Backtracking line search on $\|r(y)\|_2$:  
   - set $t = 1$  
   - while $\|r(y + t \Delta y)\|_2 > (1 - \alpha t)\|r(y)\|_2$, set $t := \beta t$.  
3. Update:  
   $x := x + t \Delta x_{\text{nt}}, \quad v := v + t \Delta v_{\text{nt}}$.

Continue until $\|r(y)\|_2 \leq \varepsilon$.



Although this is not a descent method for $f(x)$, it ensures that $\|r(y)\|_2$ decreases at each iteration.

Directional derivative of $\|r(y)\|_2$ along the Newton direction is:

$$
\left.\frac{d}{dt} \|r(y + t \Delta y)\|_2\right|_{t=0} = -\|r(y)\|_2.
$$



## 13.10 Solving KKT Systems Efficiently

Both feasible and infeasible Newton methods require solving KKT systems of the form:

$$
\begin{bmatrix}
H & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
v \\ w
\end{bmatrix}
= -
\begin{bmatrix}
g \\ h
\end{bmatrix},
$$
