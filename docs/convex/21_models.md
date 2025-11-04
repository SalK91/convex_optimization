# Chapter 14: Modelling Patterns and Algorithm Selection

Real-world modelling starts not with algorithms but with data, assumptions, and design goals.  We choose a loss function from statistical assumptions (e.g. noise model, likelihood) and a complexity penalty or constraints from design preferences (simplicity, robustness, etc.).  The resulting convex (or nonconvex) optimization problem often *tells* us which solver class to use.  In practice, solving machine learning problems looks like: modeling → recognize structure → pick solver.  Familiar ML models (linear regression, logistic regression, etc.) can be viewed as convex programs.  Below we survey common patterns (convex and some nonconvex) and the recommended algorithms/tricks for each.

## 11.1 Regularized estimation and the accuracy–simplicity tradeoff

Many learning tasks use a regularized risk minimization form:
\[
\min_x \; \underbrace{\text{loss}(x)}_{\text{data-fit}} \;+\; \lambda\;\underbrace{\text{penalty}(x)}_{\text{complexity}}.
\]
Here the loss measures fit to data (often from a likelihood) and the penalty (regularizer) enforces simplicity or structure.  Increasing $\lambda$ trades accuracy for simplicity (e.g. model sparsity or shrinkage).

- Ridge regression (ℓ₂ penalty):  
  \[
  \min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2.
  \]  
  This arises from Gaussian noise (squared-error loss) plus a quadratic prior on $x$.  It is a smooth, strongly convex quadratic problem (Hessian $A^TA + \lambda I \succ 0$).  One can solve it via Newton’s method or closed‐form normal equations, or for large problems via (accelerated) gradient descent or conjugate gradient.  Strong convexity means fast, reliable convergence with second-order or accelerated first-order methods.

- LASSO / Sparse regression (ℓ₁ penalty):  
  \[
  \min_x \tfrac12\|Ax - b\|_2^2 + \lambda \|x\|_1.
  \]  
  The $\ell_1$ penalty encourages many $x_i=0$ (sparsity) for interpretability.  The problem is convex but nonsmooth (since $|\cdot|$ is nondifferentiable at 0).  A standard solver is proximal gradient: take a gradient step on the smooth squared loss, then apply the proximal (soft-thresholding) step for $\ell_1$, which sets small entries to zero.  Coordinate descent is another popular solver – updating one coordinate at a time with a closed-form soft-thresholding step.  Proximal methods and coordinate descent scale to very high dimensions.  

- Elastic net (mixed ℓ₁+ℓ₂):  
  \[
  \min_x \|Ax - b\|_2^2 + \lambda_1\|x\|_1 + \lambda_2\|x\|_2^2.
  \]  
  This combines the sparsity of LASSO with the stability of ridge regression.  It is still convex and (for $\lambda_2>0$) strongly convex[^4].  One can still use proximal gradient (prox operator splits into soft-threshold and shrink) or coordinate descent.  Because of the ℓ₂ term, the objective is smooth and unique solution.

- Group lasso, nuclear norm, etc.: Similar composite objectives arise when enforcing block-sparsity or low-rank structure.  Each adds a convex penalty (block $\ell_{2,1}$ norms, nuclear norm) to the loss.  These remain convex, often separable or prox-friendly.  Proximal methods (using known proximal maps for each norm) or ADMM can handle these.

Algorithmic pointers for 11.1:  

- *Smooth+ℓ₂ (strongly convex)* → Newton / quasi-Newton or (accelerated) gradient descent (Chapter 9).  Closed-form if possible.  
- *Smooth + ℓ₁* → Proximal gradient or coordinate descent (Chapter 9/10).  These exploit separable nonsmoothness.  
- *Mixed penalties (ℓ₁+ℓ₂)* → Still convex; often handle like ℓ₁ case since smooth part dominates curvature.  
- *Large-scale data* → Stochastic/mini-batch variants of first-order methods (SGD, SVRG, etc.).  

*Remarks:*  Choose $\lambda$ via cross-validation or hold-out to balance fit vs simplicity.  In high dimensions ($n$ large), coordinate or stochastic methods often outperform direct second-order methods.

## 11.2 Robust regression and outlier resistance

Standard least-squares uses squared loss, which penalizes large errors quadratically. This makes it sensitive to outliers. Robust alternatives replace the loss:

### 11.2.1 Least absolute deviations (ℓ₁ loss)

Formulation:
$$
\min_x \sum_i \lvert a_i^\top x - b_i \rvert.
$$

Interpretation:

- This corresponds to assuming Laplace (double-exponential) noise on the residuals.
- Unlike squared error, it penalizes big residuals *linearly*, not quadratically, so outliers hurt less.

Geometry/structure:
- The objective is convex but nondifferentiable at zero residual (the kink in \(|r|\) at \(r=0\)).

How to solve it:

1. As a linear program (LP).  
   Introduce slack variables \(t_i \ge 0\) and rewrite:

    - constraints:  
     \(-t_i \le a_i^\top x - b_i \le t_i\),
    - objective:  
     \(\min \sum_i t_i\).
   
    This is now a standard LP. You can solve it with:

    - an interior-point LP solver,
    - or simplex.

    These methods give high-accuracy solutions and certificates.

2. First-order methods for large scale.  
   
    For *very* large problems (millions of samples/features), you can apply:
   
    - subgradient methods,
    - proximal methods (using the prox of \(|\cdot|\)).

    These are slower in theory (subgradient is only \(O(1/\sqrt{t})\) convergence), but they scale to huge data where generic LP solvers would struggle.



### 11.2.2 Huber loss

Definition of the Huber penalty for residual \(r\):
$$
\rho_\delta(r) =
\begin{cases}
\frac{1}{2} r^2, & |r| \le \delta, \\
\delta |r| - \frac{1}{2}\delta^2, & |r| > \delta.
\end{cases}
$$

Huber regression solves:
$$
\min_x \sum_i \rho_\delta(a_i^\top x - b_i).
$$

Interpretation:

- For small residuals (\(|r|\le\delta\)): it acts like least-squares (\(\tfrac{1}{2}r^2\)). So inliers are fit tightly.
- For large residuals (\(|r|>\delta\)): it acts like \(\ell_1\) (linear penalty), so outliers get down-weighted.
- Intuition: “be aggressive on normal data, be forgiving on outliers.”

Properties:

- \(\rho_\delta\) is convex.
- It is smooth except for a kink in its second derivative at \(|r|=\delta\).
- Its gradient exists everywhere (the function is once-differentiable).

How to solve it:

1. Iteratively Reweighted Least Squares (IRLS) / quasi-Newton.  
    Because the loss is basically quadratic near the solution, Newton-type methods (including IRLS) work well and converge fast on moderate-size problems.

2. Proximal / first-order methods.  
    You can apply proximal gradient methods, since each term is simple and has a known prox.

3. As a conic program (SOCP).  
    The Huber objective can be written with auxiliary variables and second-order cone constraints.  
    That means you can feed it to an SOCP solver and let an interior-point method handle it efficiently and robustly.  
    This is attractive when you want high accuracy and dual certificates.



### 11.2.3 Worst-case robust regression

Sometimes we don’t just want “fit the data we saw,” but “fit any data within some uncertainty set.” This leads to min–max problems of the form:
$$
\min_x \;\max_{u \in \mathcal{U}} \; \| (A + u)x - b \|_2.
$$

Meaning:

- \(\mathcal{U}\) is an uncertainty set describing how much you distrust the matrix \(A\), the inputs, or the measurements.
- You choose \(x\) that performs well even in the worst allowed perturbation.

Why this is still tractable:

- If \(\mathcal{U}\) is convex (for example, an $\ell_2$ ball or box bounds on each entry), then the inner maximization often has a closed-form expression.

- That inner max usually turns into an extra norm penalty or a conic constraint in the outer problem.
    - Example: if the rows of \(A\) can move within an $\ell_2$ ball of radius \(\epsilon\), the robustified problem often picks up an additional \(\ell_2\) term like \(\gamma \|x\|_2\) in the objective.
    - The final problem is still convex (often a QP or SOCP).

How to solve it:

- If it reduces to an LP / QP / SOCP, you can use an interior-point (conic) solver to get a high-quality solution and dual certificate.

- If the structure is separable and high-dimensional, you can sometimes solve the dual or a proximal/ADMM splitting of the robust problem using first-order methods.

 


## 11.3 Maximum likelihood and loss design

Choosing a loss often comes from a probabilistic noise model. The negative log-likelihood (NLL) of an assumed distribution gives a convex loss for many common cases:

- Gaussian (normal) noise

    Model:
    $$
    b = A x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I).
    $$

    The negative log-likelihood (NLL) is proportional to:
    $$
    |A x - b|_2^2.
    $$

    This recovers the classic least-squares loss (as in linear regression).  
    It is smooth and convex (strongly convex if $A^T A$ is full rank).

    Algorithms:

    - Closed-form via $(A^T A + \lambda I)^{-1} A^T b$ (for ridge regression),
    
    - Iterative methods: conjugate gradient, gradient descent (Chapter 9),
    
    - Or Newton / quasi-Newton methods (Chapter 9) using the constant Hessian $A^T A$.

- Laplace (double-exponential) noise
                        
    If $\varepsilon_i \sim \text{Laplace}(0, b)$ i.i.d., the NLL is proportional to:
    $$
    \sum_i |a_i^T x - b_i|.
    $$

    This is exactly the ℓ₁ regression (least absolute deviations).  
    It can be solved as an LP or with robust optimization solvers (interior-point),  
    or with first-order nonsmooth methods (subgradient/proximal) for large-scale problems.

- Logistic model (binary classification)

    For $y_i \in \{0,1\}$, model:
    $$
    \Pr(y_i = 1 \mid a_i, x) = \sigma(a_i^T x),
    \quad \text{where } \sigma(z) = \frac{1}{1 + e^{-z}}.
    $$

    The negative log-likelihood (logistic loss) is:
    $$
    \sum_i \left[ -y_i (a_i^T x) + \log(1 + e^{a_i^T x}) \right].
    $$

    This loss is convex and smooth in $x$.  
    No closed-form solution exists.

    Algorithms:

    - With ℓ₂ regularization: smooth and (if $\lambda>0$) strongly convex → use accelerated gradient or quasi-Newton (e.g. L-BFGS).
    - With ℓ₁ regularization (sparse logistic): composite convex → use proximal gradient (soft-thresholding) or coordinate descent.

- Softmax / Multinomial logistic (multiclass)

    For $K$ classes with one-hot labels $y_i \in \{e_1, \dots, e_K\}$, the softmax model gives NLL:
    $$
    -\sum_i \sum_{k=1}^K y_{ik}(a_i^T x_k)
    + \log\!\left(\sum_{j=1}^K e^{a_i^T x_j}\right).
    $$

    This loss is convex in the weight vectors $\{x_k\}$ and generalizes binary logistic to multiclass.

    Algorithms:

    - Gradient-based solvers (L-BFGS, Newton with block Hessian) for moderate size.
    - Stochastic gradient (SGD, Adam) for large datasets.

- Generalized linear models (GLMs)

    In GLMs, $y_i$ given $x$ has an exponential-family distribution (Poisson, binomial, etc.) with mean related to $a_i^T x$.  
    The NLL is convex in $x$ for canonical links (e.g. log-link for Poisson, logit for binomial).

    Examples:

    - Poisson regression for counts: convex NLL, solved by IRLS or gradient.
    - Probit models: convex but require iterative solvers.

## 11.4 Structured constraints in engineering and design

Optimization problems often include explicit convex constraints from physical or resource limits: e.g. variable bounds, norm limits, budget constraints. The solver choice depends on how easily we can handle projections or barriers for $\mathcal{X}$:

- Simple (projection-friendly) constraints


    Examples:

    - Box constraints: $l \le x \le u$  
        → Projection: clip each entry to $[\ell_i, u_i]$.

    - ℓ₂-ball: $\|x\|_2 \le R$  
        → Projection: rescale $x$ if $\|x\|_2 > R$.

    - Simplex: $\{x \ge 0, \sum_i x_i = 1\}$  
        → Projection: sort and threshold coordinates (simple $O(n \log n)$ algorithm).

- General convex constraints (non-projection-friendly)
If constraints are complex (e.g. second-order cones, semidefinite, or many coupled inequalities), projections are hard. Two strategies:

    1. Barrier / penalty and interior-point methods : Add a log-barrier or penalty and solve with an interior-point solver (Chapter 9). This handles general convex constraints well and returns dual variables (Lagrange multipliers) as a bonus.

    2. Conic formulation + solver: Write the problem as an LP/QP/SOCP/SDP and use specialized solvers (like MOSEK, Gurobi) that exploit sparse structure. If only first-order methods are feasible for huge problems, one can apply dual decomposition or ADMM by splitting constraints (Chapter 10), but convergence will be slower.

Algorithmic pointers for 11.4:

- Projection-friendly constraints → Projected (stochastic) gradient or proximal methods (fast, maintain feasibility).
- Complex constraints (cones, PSD, many linear) → Use interior-point/conic solvers (Chapter 9) for moderate size. Alternatively, use operator-splitting (ADMM) if parallel/distributed solution is needed (Chapter 10).
- LP/QP special cases → Use simplex or specialized LP/QP solvers (Section 11.5).

Remarks: Encoding design requirements (actuator limits, stability margins, probability budgets) as convex constraints lets us leverage efficient convex solvers. Feasible set geometry dictates the method: easy projection → projective methods; otherwise → interior-point or operator-splitting.

## 11.5 Linear and conic programming: the canonical models

Many practical problems reduce to linear programming (LP) or its convex extensions.  
LP and related conic forms are the workhorses of operations research, control, and engineering optimization.

-  Linear programs: standard form

    $$
    \min_x \; c^T x 
    \quad \text{s.t.} \quad A x = b, \; x \ge 0.
    $$
    Both objective and constraints are affine, so the optimum lies at a vertex of the polyhedron. Simplex method traverses vertices and is often very fast in practice. Interior-point methods approach the optimum through the interior and have polynomial-time guarantees. For moderate LPs, interior-point is robust and accurate; for very large LPs (sparse, structured), first-order methods or decomposition may be needed.
- Quadratic, SOCP, SDP:
    Convex quadratic programs (QP), second-order cone programs (SOCP), and semidefinite programs (SDP) generalize LP. For example, many robust or regularized problems (elastic net, robust regression, classification with norm constraints) can be cast as QPs or SOCPs. All these are solvable by interior-point (Chapter 9) very efficiently. Interior-point solvers (like MOSEK, Gurobi, etc.) are widely used off-the-shelf for these problem classes.

- Practical patterns:

    1. Resource allocation/flow (LP): linear costs and constraints.
    2. Minimax/regret problems: e.g. $\min_{x}\max_{i}|a_i^T x - b_i|$ → LP (as in Chebyshev regression).
    3. Constrained least squares: can be QP or SOCP if constraints are polyhedral or norm-based.

Algorithmic pointers for 11.5:
- Moderate LP/QP/SOCP: Use interior-point (robust, yields dual prices) or simplex (fast in practice, warm-startable).
- Large-scale LP/QP: Exploit sparsity; use decomposition (Benders, ADMM) if structure allows; use iterative methods (primal-dual hybrid gradient, etc.) for extreme scale.
- Reformulate into standard form: Recognize when your problem is an LP/QP/SOCP/SDP to use mature solvers. (E.g. ℓ∞ regression → LP, ℓ2 regression with ℓ2 constraint → SOCP.)

## 11.6 Risk, safety margins, and robust design

Modern design often includes risk measures or robustness. Two common patterns:

-  Chance constraints / risk-adjusted objectives
    E.g. require that $Pr(\text{loss}(x,\xi) > \tau) \le \delta$. A convex surrogate is to include mean and a multiple of the standard deviation:
    $$
    \min_x \; \mathbb{E}[\ell(x, \xi)] + \kappa \sqrt{\mathrm{Var}[\ell(x, \xi)]}.
    $$
    Algebra often leads to second-order cone constraints (e.g. forcing $\mathbb{E}\pm \kappa\sqrt{\mathrm{Var}}$ below a threshold). Such problems are SOCPs. Interior-point solvers handle them well (polynomial-time, high accuracy).

- Worst-case (robust) optimization:
    Specify an uncertainty set $\mathcal{U}$ for data (e.g. $u$ in a norm-ball) and minimize the worst-case cost $\max_{u\in\mathcal{U}}\ell(x,u)$. Many losses $\ell$ and convex $\mathcal{U}$ yield a convex max-term (a support function or norm). The result is often a conic constraint (for ℓ₂ norms, an SOCP; for PSD, an SDP). Solve with interior-point (if problem size permits) or with specialized proximal/ADMM methods (splitting the max-term).

Algorithmic pointers for 11.6:

 - Risk/SOCP models: Interior-point (Chapter 9) is the standard approach.
 - Robust max-min problems: Convert inner max to a convex constraint (norm or cone). Then use interior-point if the reformulation is conic. If the reformulation is a nonsmooth penalty, use proximal or dual subgradient methods.
 - Distributed or iterative solutions: If $\mathcal{U}$ or loss separable, ADMM can distribute the computation (Chapter 10).


## 11.7 Cheat sheet: If your problem looks like this, use that

This summary gives concrete patterns of models and recommended solvers/tricks:

- (A) Smooth least-squares + ℓ₂:

    - Model: $|Ax-b|_2^2 + \lambda|x|_2^2$. 
    - Solve: Gradient descent, accelerated gradient, conjugate gradient, or Newton/quasi-Newton. (Strongly convex quadratic ⇒ fast second-order methods.)

- (B) Sparse regression (ℓ₁):
   
    - Model: $\tfrac12|Ax-b|_2^2 + \lambda|x|_1$. 
    - Solve: Proximal gradient (soft-thresholding) or coordinate descent. (Composite smooth+nonsmooth separable.)

- (C) Robust regression (outliers):
    
    - Models: $\sum|a_i^T x - b_i|$, Huber loss, etc. 
    - Solve: Interior-point (LP/SOCP form) for high accuracy; subgradient/proximal (Chapter 9/10) for large data. (Convex but nondifferentiable or conic.)

- (D) Logistic / log-loss (classification):
    
    - Model: $\sum[-y_i(w^Ta_i)+\log(1+e^{w^Ta_i})] + \lambda R(w)$ with $R(w)=|w|_2^2$ or $|w|_1$. 
    - Solve:
        - If $R=\ell_2$: use Newton/gradient (smooth, strongly convex).
        - If $R=\ell_1$: use proximal gradient or coordinate descent. (Convex; logistic loss is smooth; ℓ₁ adds nonsmoothness.)

- (E) Constraints (hard limits):
    
    - Model: $\min f(x)$ s.t. $x\in\mathcal{X}$ with $\mathcal{X}$ simple. 
    - Solve: Projected (stochastic) gradient or proximal methods if projection $\Pi_{\mathcal{X}}$ is cheap (e.g. box, ball, simplex). If $\mathcal{X}$ is complex (second-order or SDP), use interior-point.

- (F) Separable structure:
    
    - Model: $\min_{x,z} f(x)+g(z)$ s.t. $Ax+Bz=c$. 
    - Solve: ADMM (Chapter 10) – it decouples updates in $x$ and $z$; suits distributed or block-structured data.

- (G) LP/QP/SOCP/SDP:
    
    - Model: linear/quadratic objective with linear/conic constraints. 
    - Solve: Simplex or interior-point (for moderate sizes). For very large sparse LPs exploit problem structure: warm-starts, decomposition methods (dual/block), or first-order methods (PDHG/ADMM).

- (H) Nonconvex patterns:
    
    - Examples: Deep neural networks (nonconvex weights), matrix factorization (bilinear), K-means clustering, mixture models. 
    
    - Solve: There is no single global solver – typically use stochastic gradient (SGD/Adam), alternating minimization (e.g. alternating least squares for matrix factorization), or EM for mixtures. Caveat: Convergence to global optimum is not guaranteed; solutions depend on initialization and may get stuck in local minima. Use regularization, multiple restarts, and heuristics (batch normalization, momentum) as needed.

- (I) Logistic (multi-class softmax):
    
     - Model: One weight vector per class, convex softmax loss (see Section 11.3). 
     
     - Solve: Similar to binary case – Newton/gradient with L2, or proximal/coordinate with ℓ₁.

- (J) Poisson and count models:
    - Model: Negative log-likelihood for Poisson (convex, see Section 11.3). 
    - Solve: Newton (IRLS) or gradient-based; interior-point can be used after conic reformulation.

 
Rule of thumb: Identify whether your objective is smooth vs nonsmooth, strongly convex vs just convex, separable vs coupled, constrained vs unconstrained. Then pick from:

 - Smooth & strongly convex → (quasi-)Newton or accelerated gradient.
 - Smooth + ℓ₁ → Proximal gradient/coordinate.
 - Nonsmooth separable → Proximal or coordinate.
 - Easy projection constraint → Projected gradient.
 - Hard constraints or conic structure → Interior-point.
 - Large-scale separable → Stochastic gradient/ADMM.

Convexity guarantees global optimum. When nonconvex (deep nets, latent variables, etc.), we rely on heuristics: SGD, random restarts, and often settle for local minima or approximate solutions.
