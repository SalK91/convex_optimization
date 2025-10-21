# Convex Conjugates

Convex conjugates, also known as **Fenchel conjugates**, are a fundamental tool in convex analysis and optimization. They provide a **dual representation of functions** and are central to:

- Deriving dual problems in convex optimization  
- Understanding subgradients and supporting hyperplanes  
- Designing efficient algorithms such as proximal and primal-dual methods  

Intuition: the convex conjugate of a function $f$ captures the **maximum linear overestimate** of $f$. It tells us, for any direction $y$, what is the steepest slope that does not underestimate $f$.


## Definitions and Formal Statements

Let $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ be a proper convex function. The **convex conjugate** $f^*: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is defined as:

\[
f^*(y) = \sup_{x \in \mathbb{R}^n} \left( \langle y, x \rangle - f(x) \right)
\]

- $y$ is the **dual variable**, representing a slope or linear functional.  
- Geometric intuition: for each $y$, $f^*(y)$ is the **height of the tightest linear overestimate** of $f$ along $y$.

### Key Properties

- $f^*$ is always **convex**, even if $f$ is not strictly convex.  
- The **biconjugate** $f^{**}$ equals $f$ if $f$ is proper, convex, and lower semicontinuous:  
\[
f^{**} = f
\]
- **Fenchel-Young inequality:** For all $x, y \in \mathbb{R}^n$:  
\[
\langle y, x \rangle \le f(x) + f^*(y)
\]
Equality holds if and only if $y \in \partial f(x)$ (subgradient condition).

---

## Step-by-Step Analysis / How to Use

1. Identify the convex function $f(x)$.  
2. For a given dual vector $y$, compute  
\[
f^*(y) = \sup_x \left( \langle y, x \rangle - f(x) \right)
\]  
3. Use the properties:  
   - For **norm functions**, conjugates give dual norms.  
   - For **indicator functions**, conjugates give support functions.  
   - Fenchel conjugates are used to **construct dual problems** in convex optimization.  
4. Apply in optimization algorithms:  
   - Proximal operators of $f^*$ relate to those of $f$ (Moreau identity)  
   - Subgradients of $f^*$ are linked to primal solutions

---

## Examples

### Example 1: Quadratic Function

Let $f(x) = \frac{1}{2} \|x\|_2^2$. Then

\[
f^*(y) = \sup_x \left( \langle y, x \rangle - \frac{1}{2} \|x\|_2^2 \right)
\]

- Differentiate: $\nabla_x (\langle y, x \rangle - \frac{1}{2} \|x\|_2^2) = y - x = 0 \implies x = y$  
- Substitute back:  
\[
f^*(y) = \langle y, y \rangle - \frac{1}{2} \|y\|_2^2 = \frac{1}{2} \|y\|_2^2
\]

**Observation:** quadratic is self-conjugate.

---

### Example 2: $\ell_1$ Norm

Let $f(x) = \|x\|_1$. Then

\[
f^*(y) = \sup_{\|x\|_1 \le \infty} \left( \langle y, x \rangle - \|x\|_1 \right)
\]

- The supremum is finite only if $\|y\|_\infty \le 1$, otherwise $+\infty$.  
- Therefore:  
\[
f^*(y) = \delta_{\{\|y\|_\infty \le 1\}}(y)
\]  
where $\delta$ is the **indicator function**.

---

### Example 3: Indicator Function

Let $f(x) = \delta_C(x)$, the indicator of a convex set $C$. Then

\[
f^*(y) = \sup_{x \in C} \langle y, x \rangle = \sigma_C(y)
\]

- Observation: **support functions are conjugates of indicator functions**.

---

## Applications / Implications

- **Dual Optimization Problems:** Fenchel duals are derived by taking conjugates of primal objectives and constraints.  
- **Subgradient Methods:** Fenchel-Young inequality provides a direct link between primal and dual subgradients.  
- **Proximal Algorithms:** Proximal operators of conjugates allow efficient updates in primal-dual splitting methods.  
- **Norm Duality:** Conjugates of norm functions yield dual norms, connecting directly to constrained optimization theory.

---

## Summary / Key Takeaways

- Convex conjugates provide a **dual perspective** of convex functions, capturing maximal linear overestimates.  
- Biconjugation recovers the original function if it is convex and lower semicontinuous.  
- Fenchel-Young inequality links primal and dual variables via subgradients.  
- Indicator functions and norm functions have conjugates that reveal **support functions** and **dual norms**.  
- Conjugates are foundational in **dual optimization, proximal algorithms, and geometric analysis**.
