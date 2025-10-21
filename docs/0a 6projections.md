# Projections and Orthogonal Decompositions

Projections and orthogonal decompositions are fundamental tools in convex optimization. They allow us to enforce constraints, compute optimality conditions, and analyze the geometry of feasible regions. This chapter develops projections onto subspaces and convex sets, orthogonal decomposition, and their connections to algorithms.

## Projection onto Subspaces

Let $W \subseteq \mathbb{R}^n$ be a subspace with an orthonormal basis $\{q_1, \dots, q_k\}$. The projection of $x \in \mathbb{R}^n$ onto $W$ is

$$
P_W(x) = \sum_{i=1}^k \langle x, q_i \rangle q_i.
$$

Properties:

- $x - P_W(x)$ is orthogonal to $W$: $\langle x - P_W(x), y \rangle = 0$ for all $y \in W$  
- $P_W$ is linear and idempotent: $P_W(P_W(x)) = P_W(x)$  
- $P_W(x)$ minimizes the distance to $W$: $\|x - P_W(x)\| = \min_{y \in W} \|x - y\|$

Geometric intuition: $P_W(x)$ is the “shadow” of $x$ onto the subspace $W$. In optimization, projections are used in projected gradient descent, where iterates are kept within a feasible linear subspace.


## Projection onto Convex Sets

For a closed convex set $C \subseteq \mathbb{R}^n$, the metric projection of $x$ onto $C$ is

$$
P_C(x) = \arg\min_{y \in C} \|x - y\|.
$$

Properties:

- Uniqueness: There exists a unique minimizer $P_C(x)$  
- Firm nonexpansiveness:

$$
\|P_C(x) - P_C(y)\|^2 \le \langle P_C(x) - P_C(y), x - y \rangle \quad \forall x, y \in \mathbb{R}^n
$$

- First-order optimality: $\langle x - P_C(x), y - P_C(x) \rangle \le 0$ for all $y \in C$

Applications in optimization:

- Ensures iterates remain feasible in projected gradient methods  
- Guarantees stability and convergence due to nonexpansiveness  
- Appears in proximal operators when $C$ is a level set of a regularizer


A useful corollary is nonexpansiveness:
$$
\|\operatorname{proj}_C(x) - \operatorname{proj}_C(y)\| \le \|x - y\|.
$$

## Orthogonal Decomposition

Any vector $x \in \mathbb{R}^n$ can be uniquely decomposed into components along a subspace $W$ and its orthogonal complement $W^\perp$:

$$
x = P_W(x) + (x - P_W(x)), \quad P_W(x) \in W, \quad x - P_W(x) \in W^\perp
$$

Properties:

- The decomposition is unique  
- $\|x\|^2 = \|P_W(x)\|^2 + \|x - P_W(x)\|^2$ (Pythagoras’ theorem)  

Applications:

- In constrained optimization, the gradient can be decomposed into components tangent to constraints and normal to them, forming the basis for KKT conditions.  
- In machine learning, projections onto feature subspaces allow dimensionality reduction and orthogonalization of data.
