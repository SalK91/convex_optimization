# A.5 Epigraphs and Convexity

Epigraphs are geometric tools that help us define and visualize convex functions. They serve as a bridge between convex **functions** and convex **sets**, allowing us to apply set-based reasoning (see [A.1 Convex Sets]) to analyze functions (see [A.4 Convex Functions]).

---

## 1. What Is an Epigraph?

Let $f : \mathbb{R}^n \to \mathbb{R}$. The **epigraph** of $f$ is the set:

$$
\operatorname{epi}(f) = \{ (x, t) \in \mathbb{R}^n \times \mathbb{R} \;\mid\; f(x) \le t \}.
$$

- Each point $(x, t)$ lies **on or above the graph** of $f$.
- The epigraph lives in $\mathbb{R}^{n+1}$.
- It includes all **superlevel points** with respect to the function.

---

## 2. Intuition and Visualization

Think of the graph of $f$ as a landscape. Then:

- The **graph** is the exact curve or surface traced out by $f$.
- The **epigraph** is the **solid region above** the graph — like the sky above a hill.

Example:  
If $f(x) = x^2$, then $\operatorname{epi}(f)$ is the set of points above the parabola in 2D.

> The epigraph captures **all points that are “safe” from below** the function — a perspective crucial for defining convexity.

---

## 3. Convexity from Epigraphs

A function $f$ is **convex** if and only if its epigraph is a **convex set**.

This means:
- If $(x_1, t_1)$ and $(x_2, t_2)$ lie in $\operatorname{epi}(f)$, then for any $\theta \in [0,1]$, the convex combination
  $$
  (\theta x_1 + (1 - \theta) x_2, \; \theta t_1 + (1 - \theta) t_2)
  $$
  also lies in $\operatorname{epi}(f)$.

This condition geometrically encodes **Jensen’s inequality** and convexity.

For background on convex sets and linear combinations, see [A.1 Convex Sets] and [A.2 Affine Sets and Hyperplanes].

---

## 4. Examples

### Example 1: Convex Function

Let $f(x) = x^2$.

- Epigraph: the set of points above the parabola.
- This set is convex.
- $\Rightarrow$ $f$ is convex (see [A.4]).

### Example 2: Nonconvex Function

Let $f(x) = -x^2$.

- Epigraph: the region above the upside-down parabola.
- This set is **not convex** (you can draw line segments that dip below the curve).
- $\Rightarrow$ $f$ is not convex.

---

## 5. Epigraph Form in Optimization

Epigraphs allow us to **reformulate optimization problems** in a form that emphasizes convexity.

Given:
$$
\min_{x} f(x)
$$

We can write the **epigraph form** as:
$$
\min_{x, t} \; t \quad \text{subject to } f(x) \le t
$$

This has advantages:
- The feasible region becomes $\operatorname{epi}(f)$.
- Makes it easier to check convexity of constraints.
- Many solvers (e.g., CVXPY) require problems to be in epigraph-compatible form.

This idea is a building block in **disciplined convex programming** — see [E.1 Modeling Convex Optimization Problems].

---

## 6. Why Epigraphs Matter

Epigraphs unify:
- **Function analysis** (shape of $f$)
- **Geometric reasoning** (convex sets in higher dimensions)
- **Modeling flexibility** (lift problems to introduce new variables)
- **Duality theory** (used to characterize convex conjugates in [D.4 Dual Norms and Fenchel Conjugates])

They also make nonsmooth functions (like norms or max operations) easier to analyze and optimize — see [A.6 Norms and Balls] and [A.7 Subgradients].

---

## ✅ Summary and Takeaways

- The **epigraph** of a function is the region above its graph.
- A function is **convex if and only if its epigraph is convex**.
- Many optimization problems can be lifted into **epigraph form** to clarify structure and aid computation.
- Epigraphs provide an elegant bridge between **function analysis** and **convex geometry**.

