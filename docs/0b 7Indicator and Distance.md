# Indicator & Distance Functions

Indicator and distance functions are fundamental in convex analysis and optimization. They provide **a convenient way to represent constraints** and measure distances to sets:

- Indicator functions encode whether a point is feasible.  
- Distance functions quantify **how far a point is from feasibility**.  
- Both are essential in **projection algorithms, proximal methods, and dual formulations**.

Intuition: Think of a feasible region in space. The **indicator function** is zero inside the region and infinite outside—it tells you “allowed or not allowed.” The **distance function** tells you how far you are from being allowed and naturally guides **projection-based updates**.

 
## Definitions and Formal Statements

### Indicator Function

For a set $C \subseteq \mathbb{R}^n$, the **indicator function** $\delta_C: \mathbb{R}^n \to \{0, +\infty\}$ is defined as:

\[
\delta_C(x) =
\begin{cases}
0 & \text{if } x \in C \\
+\infty & \text{if } x \notin C
\end{cases}
\]

- Encodes **hard constraints** in optimization.  
- Appears naturally in **proximal operators**: $\text{prox}_{\delta_C}(x) = P_C(x)$, the metric projection onto $C$.

 
### Distance Function

The **distance function** to a set $C$ is:

\[
d_C(x) = \inf_{y \in C} \|x - y\|
\]

- Measures how far $x$ is from the set $C$.  
- Related to the **indicator function**: $\delta_C(x) = 0$ if $d_C(x) = 0$, otherwise $+\infty$.  
- Differentiable almost everywhere if $C$ is convex, and its gradient points toward the **closest point in $C$**.

 
## Step-by-Step Analysis / How to Use

1. Identify the feasible set $C$ for constraints.  
2. Use the **indicator function** to include constraints in unconstrained optimization formulations:  
\[
\min_x f(x) + \delta_C(x)
\]  
3. Use the **distance function** to measure infeasibility or guide projections.  
4. Apply in algorithms:  
   - **Proximal operators:** $\text{prox}_{\delta_C}(x) = P_C(x)$  
   - **Projected gradient methods:** $x_{k+1} = P_C(x_k - \alpha \nabla f(x_k))$  
   - **Penalty methods:** $f(x) + \frac{\rho}{2} d_C(x)^2$ approximates constraints softly.

 
## Examples

### Example 1: Nonnegative Orthant

Let $C = \{ x \in \mathbb{R}^n : x \ge 0 \}$:

- Indicator function: $\delta_C(x) = 0$ if $x_i \ge 0$ for all $i$, $+\infty$ otherwise.  
- Distance function: $d_C(x) = \|x - P_C(x)\|_2$, where $P_C(x) = \max(x, 0)$ componentwise.  
- Proximal operator: $\text{prox}_{\delta_C}(x) = P_C(x)$ projects negative entries to zero.

### Example 2: Euclidean Ball

Let $C = \{x : \|x\|_2 \le r\}$:

- Distance: $d_C(x) = \max(0, \|x\|_2 - r)$  
- Projection: $P_C(x) = x$ if $\|x\|_2 \le r$, otherwise $P_C(x) = r \frac{x}{\|x\|_2}$

 
## Applications / Implications

- **Constraint Handling:** Indicator functions allow transforming constrained problems into **unconstrained forms** suitable for proximal algorithms.  
- **Proximal Algorithms:** Projection onto $C$ is equivalent to applying the proximal operator of $\delta_C$.  
- **Distance Minimization:** Distance functions quantify infeasibility and guide **penalty or barrier methods**.  
- **Duality & Conjugates:** Indicator functions are conjugate to **support functions**, connecting geometry and duality.

 
## Summary / Key Takeaways

- Indicator functions encode **hard constraints**; distance functions measure **infeasibility**.  
- Proximal operators of indicator functions are **metric projections**.  
- Distance functions and projections are central in **projection-based optimization algorithms**.  
- They link **geometric intuition** with **algebraic optimization tools**, forming the basis for constraints, duality, and proximal methods.
