Projection is the operation of finding the closest point in a given set to a point outside the set. It is a key step in many algorithms (projected gradient, alternating projections, etc.) to enforce constraints. Geometrically, projections are about “dropping perpendiculars” to a subspace or convex set.

**Projection onto a subspace:** Let $W \subseteq \mathbb{R}^n$ be a subspace (e.g. defined by a set of linear equations $Ax=0$ or spanned by some basis vectors). The orthogonal projection of any $x \in \mathbb{R}^n$ onto $W$ is the unique point $P_W(x) \in W$ such that $x - P_W(x)$ is orthogonal to $W$. If ${q_1,\dots,q_k}$ is an orthonormal basis of $W$, then
​
$$
P_W(x) = \sum_{i=1}^k \langle x, q_i \rangle \, q_i
$$


as mentioned earlier. This $P_W(x)$ minimizes the distance $|x - y|2$ over all $y\in W$. The residual $r = x - P_W(x)$ is orthogonal to every direction in $W$. For example, projecting a point in space onto a plane is dropping a perpendicular to the plane. In $\mathbb{R}^n$, $P_W$ is an $n \times n$ matrix (if $W$ is $k$-dimensional, $P_W$ has rank $k$) that satisfies $P_W^2 = P_W$ (idempotent) and $P_W = P_W^T$ (symmetric). In optimization, if we are constrained to $W$, a projected gradient step does $x_{k+1} = P_W(x_k - \alpha \nabla f(x_k))$ to ensure $x_{k+1} \in W$.

**Projection onto a convex set:** More generally, for a closed convex set $C \subset \mathbb{R}^n$, the projection $\operatorname{proj}_C(x)$ is defined as the unique point in $C$ closest to $x$:

$$
P_C(x) = \arg\min_{y \in C} \|x - y\|_2
$$


For convex $C$, this best approximation exists and is unique. While we may not have a simple formula like in the subspace case, projections onto many sets have known formulas or efficient algorithms (e.g. projecting onto a box $[l,u]$ just clips each coordinate between $l$ and $u$). Some properties of convex projections: 

- $P_C(x)$ lies on the boundary of $C$ along the direction of $x$ if $x \notin C$. 

- The first-order optimality condition for the minimization above says $(x - P_C(x))$ is orthogonal to the tangent of $C$ at $P_C(x)$, or equivalently $\langle x - P_C(x), y - P_C(x)\rangle \le 0$ for all $y \in C$. This means the line from $P_C(x)$ to $x$ forms a supporting hyperplane to $C$ at $P_C(x)$. 
- Also, projections are firmly non-expansive: $|P_C(x)-P_C(y)|^2 \le \langle P_C(x)-P_C(y), x-y \rangle \le |x-y|^2$. Intuitively, projecting cannot increase distances and in fact pulls points closer in a very controlled way. This is important for convergence of algorithms like alternating projections and proximal point methods, ensuring stability.

**Examples:**

- Projection onto an affine set $Ax=b$ (assuming it’s consistent) can be derived via normal equations: one finds a correction $\delta x$ in the row space of $A^T$ such that $A(x+\delta x)=b$. The solution is $P_C(x) = x - A^T(AA^T)^{-1}(Ax-b)$ (for full row rank $A$).

- Projection onto the nonnegative orthant ${x: x_i\ge0}$ just sets $x_i^- = \min(x_i,0)$ to zero (i.e. $[x]^+ = \max{x,0}$ componentwise). This is used in nonnegativity constraints.

- Projection onto an $\ell_2$ ball ${|x|_2 \le \alpha}$ scales $x$ down to have length $\alpha$ if $|x|>\alpha$, or does nothing if $|x|\le\alpha$.

- Projection onto an $\ell_1$ ball (for sparsity) is more involved but essentially does soft-thresholding on coordinates to make the sum of absolute values equal $\alpha$.

**Why projections matter in optimization:** Many convex optimization problems involve constraints $x \in C$ where $C$ is convex. If we can compute $P_C(x)$ easily, we can use projection-based algorithms. For instance, projected gradient descent: if we move in the negative gradient direction and then project back to $C$, we guarantee the iterate stays feasible and we still decrease the objective (for small enough step). The property of projections that $(x - P_C(x))$ is orthogonal to the feasible region at $P_C(x)$ connects to KKT conditions: at optimum $\hat{x}$ with $\hat{x} = P_C(x^* - \alpha \nabla f(\hat{x}))$, the vector $-\nabla f(\hat{x})$ must lie in the normal cone of $C$ at $\hat{x}$, meaning the gradient is “balanced” by the constraint boundary — this is exactly the intuition behind Lagrange multipliers. In fact, one of the KKT conditions can be seen as stating that $\hat{x} = P_C(x^* - \alpha \nabla f(x^*))$ for some step $\alpha$, i.e. you cannot find a feasible direction that improves the objective (otherwise the projection of a slight step would move along that direction).

**Orthogonal decomposition:** Any vector $x$ can be uniquely decomposed relative to a subspace $W$ as $x = P_W(x) + r$ with $r \perp W$. Moreover, $|x|^2 = |P_W(x)|^2 + |r|^2$ (Pythagorean theorem). This orthogonal decomposition is a geometric way to understand degrees of freedom. In constrained optimization with constraint $x\in W$, any descent direction $d$ can be split into a part tangent to $W$ (which actually moves within $W$) and a part normal to $W$ (which violates constraints). Feasible directions are those with no normal component. At optimum, the gradient $\nabla f(x^)$ being orthogonal to the feasible region means $\nabla f(x^)$ lies entirely in the normal subspace $W^\perp$ — no component lies along any feasible direction. This is exactly the condition for optimality with equality constraints: $\nabla f(x^)$ is in the row space of $A$ if $Ax^=b$ are active constraints, which leads to $\nabla f(x^*) = A^T \lambda$ for some $\lambda$ (the Lagrange multipliers). Thus, orthogonal decomposition underpins the optimality conditions in constrained problems.

**Projection algorithms:** The simplicity or difficulty of computing $P_C(x)$ often determines if we can solve a problem efficiently. If $C$ is something like a polyhedron given by linear inequalities, $P_C$ might require solving a QP each time. But for many simple sets (boxes, balls, simplices, spectral norm or nuclear norm balls, etc.), we have closed forms. This gives rise to the toolbox of proximal operators in convex optimization, which generalize projections to include objective terms. Proximal gradient methods rely on computing $\operatorname{prox}{\gamma g}(x) = \arg\min_y {g(y) + \frac{1}{2\gamma}|y-x|^2}$, which for indicator functions of set $C$ yields $\operatorname{prox}{\delta_C}(x) = P_C(x)$. Thus projection is a special proximal operator (one for constraints).

In conclusion, projections are how we enforce constraints and decompose optimization problems. They appear in the analysis of alternating projection algorithms (for finding a point in $C_1 \cap C_2$ by $x_{k+1}=P_{C_1}(P_{C_2}(x_k))$), in augmented Lagrangian methods (where a proximal term causes an update like a projection), and in many other contexts. Mastering the geometry of projections — that the closest point condition yields orthogonality conditions and that projections do not expand distances — is crucial for understanding how constraint-handling algorithms converge.


