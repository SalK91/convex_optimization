# 📘 Geometric Programming (GP) 
Geometric Programming (GP) is a flexible, widely used optimization class (communications, circuit design, resource allocation, control, ML model fitting). In its natural variable form it looks **nonconvex**, but — crucially — there is a canonical change of variables and a monotone transformation that converts a GP into a **convex optimization problem**.


## Definitions: monomials, posynomials, and the standard GP

Let $x=(x_1,\dots,x_n)$ with $x_i>0$.

- A **monomial** (in GP terminology) is a function of the form
$$
m(x) = c\, x_1^{a_1} x_2^{a_2}\cdots x_n^{a_n},
$$
where $c>0$ and the exponents $a_i\in\mathbb{R}$ (real exponents allowed).  
> Note: in GP literature "monomial" means *positive coefficient times a power product* (not to be confused with polynomial monomial which has nonnegative integer powers).

- A **posynomial** is a sum of monomials:
$$
p(x) = \sum_{k=1}^K c_k\, x_1^{a_{1k}} x_2^{a_{2k}} \cdots x_n^{a_{nk}},
\qquad c_k>0.
$$

- **Standard (inequality) form** of a geometric program:
$$
\begin{aligned}
\min_{x>0}\quad & p_0(x) \\
\text{s.t.}\quad & p_i(x) \le 1,\quad i=1,\dots,m,\\
& m_j(x) = 1,\quad j=1,\dots,q,
\end{aligned}
$$
where each $p_i$ is a posynomial and each $m_j$ is a monomial. (Any GP with other RHS values can be normalized to this form by dividing.)


## ❗ Why GP (in the original $x$ variables) is **not** convex

- A monomial $m(x)=c x^a$ (single variable) is convex on $x>0$ only for certain exponent ranges (e.g. $a\le 0$ or $a\ge 1$). For $0<a<1$ it is **concave**; for general real $a$ it can be neither globally convex nor concave over $x>0$.  
  Example: $f(x)=x^{1/2}$ ($0<a<1$) is concave on $(0,\infty)$ (second derivative $=\tfrac{1}{4}x^{-3/2}>0$? — check sign; in fact $f''(x)= -\tfrac{1}{4} x^{-3/2}<0$ showing concavity).

- A **posynomial** is a **sum** of monomials. Sums of nonconvex (or concave) terms are generally nonconvex. There is no general convexity guarantee for posynomials in the original variables $x$.

- Therefore the objective $p_0(x)$ and constraints $p_i(x)\le 1$ are **not convex functions/constraints in $x$**, so the GP is not a convex program in the $x$-space.

**Concrete counterexample (1D):** take $p(x)=x^{1/2}+x^{-1}$. The term $x^{1/2}$ is concave on $(0,\infty)$, $x^{-1}$ is convex, and the sum is neither convex nor concave. One can find points $x_1,x_2$ and $\theta\in(0,1)$ that violate the convexity inequality.

---

## ✅ How to make GP convex: the log-change of variables and log transformation

Key facts that enable convexification:

- **Monomials become exponentials of affine functions in log-variables.**  
  Define $y_i = \log x_i$ (so $x_i = e^{y_i}$) and write $y=(y_1,\dots,y_n)$. For a monomial
  $$
  m(x) = c \prod_{i=1}^n x_i^{a_i},
  $$
  we have
  $$
  m(e^y) = c \exp\!\big( a^T y \big),
  \qquad\text{and}\qquad
  \log m(e^y) = \log c + a^T y,
  $$
  which is **affine** in $y$.

- **Posynomials become sums of exponentials of affine functions**. For a posynomial
  $$
  p(x) = \sum_{k=1}^K c_k \exp(a_k^T y),\qquad a_k\in\mathbb{R}^n,
  $$
  where $a_k$ is the exponent-vector for the $k$th monomial and $y=\log x$.

- **Taking the log of a posynomial yields a log-sum-exp function**, i.e.
  $$
  \log p(e^y) = \log\!\Big(\sum_{k=1}^K c_k e^{a_k^T y}\Big)
  = \operatorname{LSE}\big( a_1^T y + \log c_1,\; \dots,\; a_K^T y + \log c_K \big),
  $$
  where $\operatorname{LSE}(z_1,\dots,z_K)=\log\!\sum_{k} e^{z_k}$.

- **The log-sum-exp function is convex.** Hence constraints of the form $p_i(x)\le 1$ become
  $$
  \log p_i(e^y) \le 0,
  $$
  i.e. a convex constraint in $y$ because $\log p_i(e^y)$ is convex.

- Since $\log(\cdot)$ is monotone, minimizing $p_0(x)$ is equivalent to minimizing $\log p_0(x)$. Therefore one may transform the GP to the equivalent convex program in $y$:

  $$\boxed{%
  \begin{aligned}
  \min_{y\in\mathbb{R}^n}\quad & \log\!\Big(\sum_{k=1}^{K_0} c_{0k} e^{a_{0k}^T y}\Big) \\
  \text{s.t.}\quad & \log\!\Big(\sum_{k=1}^{K_i} c_{ik} e^{a_{ik}^T y}\Big) \le 0,\quad i=1,\dots,m,\\
  & a_{j}^T y + \log c_j = 0,\quad \text{(for each monomial equality } m_j(x)=1).
  \end{aligned}}$$

This $y$-problem is convex: *log-sum-exp* objective/constraints are convex; monomial equalities are affine in $y$.

---

## 🔍 Why log-sum-exp is convex (brief proof via Hessian)

Let $g(y)=\log\!\sum_{k=1}^K e^{u_k(y)}$ with $u_k(y)=a_k^T y + b_k$ (affine). Define
$$
s(y)=\sum_{k=1}^K e^{u_k(y)},\qquad p_k(y)=\frac{e^{u_k(y)}}{s(y)},\; \sum_k p_k=1.
$$
- Gradient:
$$
\nabla g(y) = \sum_{k=1}^K p_k(y)\, a_k.
$$
- Hessian:
$$
\nabla^2 g(y) = \sum_{k=1}^K p_k\, a_k a_k^T - \Big(\sum_{k=1}^K p_k a_k\Big)\Big(\sum_{k=1}^K p_k a_k\Big)^T
= \sum_{k=1}^K p_k (a_k-\bar a)(a_k-\bar a)^T \succeq 0,
$$
where $\bar a=\sum_k p_k a_k$. The Hessian is a weighted covariance matrix of the vectors $a_k$ (weights $p_k\ge0$), hence PSD. Thus $g$ is convex.

---

## ✳️ Monomials as affine constraints in $y$

A monomial equality $c x^{a} = 1$ becomes
$$
\log c + a^T y = 0,
$$
an affine equality in $y$. So monomial equality constraints become linear equalities after the log change.

---

## 🔁 Equivalence and solving workflow

1. **Start** with GP in $x>0$: minimize $p_0(x)$ subject to posynomial constraints and monomial equalities.  
2. **Change variables:** $y=\log x$ (domain becomes all $\mathbb{R}^n$).  
3. **Apply log to posynomials** (objective + inequality LHS). Because $\log$ is monotone increasing, inequalities maintain direction.  
4. **Solve the convex problem** in $y$ (log-sum-exp objective, convex constraints). Use interior-point or other convex solvers.  
5. **Recover $x^\star = e^{y^\star}$**.

Because $x\mapsto \log x$ is a bijection for $x>0$, solutions correspond exactly.

---

## 🔧 Worked-out simple example (2 variables)

**Original GP (standard form):**
$$
\begin{aligned}
\min_{x_1,x_2>0}\quad & p_0(x)=3 x_1^{-1} + 2 x_1 x_2,\\
\text{s.t.}\quad & p_1(x)=0.5 x_1^{-1} + 1\cdot x_2 \le 1.
\end{aligned}
$$

**Change variables:** $y_1=\log x_1,\; y_2=\log x_2$.

- Transform terms:
  - $3 x_1^{-1} = 3 e^{-y_1}$ with $\log$ term $\log 3 - y_1$.
  - $2 x_1 x_2 = 2 e^{y_1+y_2}$ with $\log$ term $\log 2 + y_1 + y_2$.
  - Constraint posynomial: $0.5 e^{-y_1} + e^{y_2}$.

**Convex form (in $y$):**
$$
\begin{aligned}
\min_{y\in\mathbb{R}^2}\quad & \log\!\big( 3 e^{-y_1} + 2 e^{y_1+y_2} \big) \\
\text{s.t.}\quad & \log\!\big( 0.5 e^{-y_1} + e^{y_2} \big) \le 0.
\end{aligned}
$$

Both objective and constraint are log-sum-exp functions (convex). Solve for $y^\star$ with a convex solver; then $x^\star = e^{y^\star}$.

---

## ⚙️ Numerical & implementation remarks

- **Domain requirement:** GP requires $x_i>0$. The log transform only works on the positive orthant. If some variables can be zero, model reformulation (introducing small positive lower bounds) may be necessary.

- **Normalization:** Standard GPs use constraints $p_i(x)\le 1$. If you have $p_i(x) \le t$, divide by $t$ to normalize.

- **Numerical stability:** Use the stable log-sum-exp implementation:
  $$
  \log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m},\qquad m=\max_k z_k,
  $$
  to avoid overflow/underflow.

- **Solvers:** After convexification the problem can be passed to generic convex solvers (CVX, CVXOPT, MOSEK, SCS, ECOS). Many solvers accept the log-sum-exp cone directly. Interior-point methods are effective on moderate-size GPs.

- **Interpretation:** The convexified problem is not an LP; it is a convex program with log-sum-exp terms (equivalently representable using exponential/relative entropy cones or by second-order cone approximations in some cases).

---

## 🚫 Limitations and extensions

- **Signomials:** If the problem contains negative coefficients (e.g. sums of monomials with arbitrary signs), it is a *signomial* program and **the log transform does not yield a convex problem**. Signomial programs are generally nonconvex and require local optimization or sequential convex approximations (e.g., successive convex approximation / condensation, branch-and-bound heuristics).

- **Robust GP:** Uncertainty in coefficients $c_k$ or exponents $a_k$ can sometimes be handled by robustification. If uncertainty is modeled multiplicatively (log-convex uncertainty sets), the robust counterpart often remains convex in $y$; specifics depend on the uncertainty set shape (box → additional terms, ellipsoidal → conic forms). Robust GP is a rich topic — it frequently yields convex reformulations after log-transform if the uncertainty preserves log-convexity.

- **Sparsity/exponents:** Exponents $a_{ik}$ need not be integers — positive real exponents are allowed. Sparsity in exponent matrix ($a_{ik}$) often leads to efficient implementations.

---

## ✅ Takeaway / Summary (concise)

- **GP in $x$ is typically nonconvex** because posynomials are not convex in $x$.  
- **Key transformation:** $y=\log x$, and take $\log$ of posynomials → posynomial $\mapsto$ log-sum-exp of affine functions in $y$.  
- **Result:** The GP becomes a *convex* optimization problem in $y$: log-sum-exp objective/constraints (convex), and monomial equalities become linear equalities.  
- **Solve in $y$**, then exponentiate to get $x$.  
- **Caveats:** requires $x>0$; signomials (negative coefficients) remain nonconvex and need other techniques.

---

If you want, I can:
- provide a **numerical, step-by-step example** solving the worked GP above (with numbers) and show the solver output and $x^\star$;  
- show how **robust** multiplicative uncertainty in coefficients is handled in log domain; or  
- give a short code snippet (CVX/MOSEK or CVXPY) that constructs and solves the convexified GP.

Which of those would you like next?
