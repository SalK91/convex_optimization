Convex optimization problems form the backbone of modern optimization theory due to their well-behaved geometry and tractable properties. By definition, an optimization problem is convex if (a) its objective function is convex and (b) its feasible set is convex. In practical terms, this means any weighted average (or convex combination) of two feasible points remains feasible (a hallmark of convex sets – see Section A, Chapter 4: Affine and Convex Geometry). These conditions are crucial because they guarantee that every local optimum is a global optimum. In other words, you never get trapped in a "bad" local minimum when the problem is convex. This global optimality property makes convex problems far easier to solve reliably than general nonlinear problems.

**Convex Optimization Problem – Formal Definition:** A standard form optimization problem is convex if it can be written as: minimize $f_0(x)$ subject to $f_i(x) \le 0$ (for $i=1,\dots,m$) and $h_j(x) = 0$ (for $j=1,\dots,p$), where $f_0$ is a convex function, each inequality constraint function $f_i$ is convex, and each equality constraint $h_j(x)$ is an affine function (linear function plus a constant). Equivalently, it is the problem of minimizing a convex function over a convex set. The feasible region in a convex problem is the intersection of a convex domain and constraint sets, which is itself convex.

Convexity of the objective and constraints can be tested using both geometric intuition and analytic tools. In this section, we'll walk through how to verify the convexity of an optimization problem step by step. We start by checking the objective function (is it convex?), then each constraint (does it define a convex region?), and finally we outline a checklist and even a simple flowchart of questions to systematically determine convexity. We will also highlight common patterns that break convexity (so you can spot non-convex structures easily), and work through several intuitive examples (like least squares, LASSO, and a quadratic program) to solidify the concepts. Throughout, we'll lean on geometric language (think of “bowl-shaped” functions and “flat” constraints) and simple derivations rather than heavy proofs – our goal is building understanding and intuition for graduate students and ML practitioners.

### Convexity of the Objective Function

The objective function $f_0(x)$ is the function we seek to minimize (we'll assume a minimization problem for concreteness; a maximization of a concave function is equivalent since maximizing a concave $g$ is same as minimizing $f_0(x) = -g(x)$, which is convex
en.wikipedia.org
). To check if $f_0(x)$ is convex, we have several tools at our disposal, ranging from visual (geometric) tests to analytical conditions:

- **Geometric Definition (Epigraph Test):** Recall that a function $f:\mathbb{R}^n\to\mathbb{R}$ is convex if and only if its epigraph – the set $\mathrm{epi}(f) = {(x,t): f(x) \le t}$ – is a convex set. Geometrically, this means that if you take any two points on the graph of $f$, the line segment (or “chord”) connecting them lies above the graph everywhere between those points. Algebraically, for any two points $x,y$ in the domain and any $\theta\in[0,1]$, a convex $f$ satisfies the inequality: 
    $$
    f(\theta x + (1 - \theta) y)
    \le
    \theta f(x) + (1 - \theta) f(y)
    $$
    
    This is the precise definition of convexity (see Section A, Chapter 4). In contrast, a non-convex function will "curve up and down" – its graph might dip below some chord (a familiar example is $\sin x$, which is neither convex nor concave over its full domain).

- **First-Order Test (Tangent Underestimation):** If $f(x)$ is differentiable, a very useful characterization of convexity is that its tangent plane at any point lies below the graph of $f$ everywhere. Formally, $f$ is convex if and only if for all points $x,y$ in its domain,
    $$
    f(y) \ge f(x) + \nabla f(x)^\top (y - x)
    $$
    This inequality says that the first-order Taylor approximation of $f$ at $x$ underestimates the function at any other point $y$ (no tangent ever goes above the curve). This is intuitive: a convex function has no “hidden dips” below its tangents. In geometric terms, you can stand anywhere on the surface and the local linear approximation forms a supporting plane underneath the surface. If $f$ is not differentiable, we can use the more general notion of a subgradient $g \in \partial f(x)$. A vector $g$ is a subgradient of $f$ at $x$ if $f(y) \ge f(x) + g^T (y - x)$ for all $y$. For convex functions, at least one subgradient exists at every point (even at a kink), and the inequality still holds. For example, the absolute value function $f(x)=|x|$ is convex but not differentiable at 0; its subgradients at 0 are any $g\in[-1,1]$, which gives an entire family of supporting lines $f(y) \ge |0| + g,(y-0)$ hovering below the $|x|$ V-shape.

- **Second-Order Test (Hessian Criterion):** If $f(x)$ is twice differentiable, convexity can be checked via the Hessian matrix (the matrix of second partial derivatives). The Hessian $\nabla^2 f(x)$ must be positive semidefinite (PSD) at every point $x$ in the domain for $f$ to be convex. In $\mathbb{R}$ (one dimension), this reduces to the simple condition $f''(x) \ge 0$ for all $x$. In higher dimensions, positive semidefiniteness means $z^T (\nabla^2 f(x)),z \ge 0$ for all vectors $z$ (intuitively, $f$ curves "upward" in every direction). This second-order test is often the most straightforward for smooth functions: for instance, if $f(x) = \tfrac{1}{2}x^T Q x + c^T x$ is a quadratic function, then $\nabla^2 f(x) = Q$, so $f$ is convex exactly when $Q \succeq 0$ (PSD matrix). As a quick example, $f(x)=x^2$ has $f''(x)=2>0$ so it's convex (a bowl shape), whereas $f(x)=\cos x$ has $f''(x)=-\cos x$ which is negative around $x=0$, revealing its non-convex curvature.

- **Recognizing Known Convex Functions (By Type):** Over time, you will build a library of common convex functions and patterns. Many functions are known to be convex by their form. For example: any linear or affine function ($f(x)=a^T x + b$) is convex (and concave) because it just produces a flat plane. Quadratic functions $f(x)=x^T Q x + c^T x + d$ are convex if $Q$ is PSD, as noted. Norms like $||x||_2$ or $||x||1$ are convex (the $L_1$ norm is essentially a sum of absolute values, forming a pointed "diamond" cone). Exponential functions $e^{ax}$ are convex for any real $a$. Even-powered monomials $x^{2}, x^4, x^6,\dots$ are convex on $\mathbb{R}$. More generally, $x^p$ is convex on $\mathbb{R}{++}$ (positive reals) for $p\ge 1$ or $p\le 0$. Negative entropy $x\log x$ is convex on $x>0$. The log-sum-exp function (important in machine learning) $f(x)=\log(e^{x_1}+\cdots+e^{x_n})$ is convex. Even the maximum of a set of convex functions is convex (e.g. $f(x)=\max{f_1(x),\dots,f_k(x)}$ is convex) because its epigraph is the intersection of halfspaces from each $f_i$. If your objective $f_0(x)$ can be expressed as a sum of convex functions, a maximum of convex functions, or an affine transformation of a convex function, then $f_0$ is convex. (Convexity is closed under these operations: e.g. nonnegative weighted sums of convex functions remain convex, and composing a convex function with an affine mapping keeps it convex.)

In practice, a good strategy is to decompose the objective into known building blocks. If each piece is convex and they are combined by operations that preserve convexity, then the whole objective is convex. For example, if $f_0(x) = g(h(x))$ and you know $g$ is convex non-decreasing and $h(x)$ is convex, then $f_0$ is convex (one common case: $g(t)=\log t$ which is increasing concave, so $-g(t)=-\log t$ is convex non-decreasing; thus $- \log(h(x))$ is convex if $h(x)$ is convex and positive). On the other hand, if you detect any component that is non-convex (say a term like $\sin x$ or $x^T Q x$ with an indefinite $Q$, or a product of variables like $x_i x_j$), that is a red flag – the objective might be non-convex unless that term can be transformed or bounded within a convex structure.

### Convexity of the Constraints (Feasible Set)

The second part of the convexity check is to examine the constraints, which determine the feasible set. Even if the objective is convex, a non-convex feasible region will make the overall problem non-convex (since you're effectively minimizing a convex function over a non-convex set, which can introduce local minima). To ensure the feasible set is convex, each constraint must individually define a convex set, and all constraints together (their intersection) must therefore also yield a convex set.

Here are the typical types of constraints and how to check their convexity:

- **Convex Inequality Constraints:** These are of the form
    $$
    f_i(x) \le 0
    $$
    
    where $f_i(x)$ is a convex function. Such a constraint means we are taking a sublevel set of a convex function, ${x \mid f_i(x) \le 0}$. By definition, any sublevel set of a convex function is a convex set. Intuitively, if $f_i$ is convex, the region where $f_i(x)$ is below some threshold looks like a "bowl" or a filled-in convex region. For example, $f_i(x) = ||x||_2 - 1 \le 0$ describes the set ${x: ||x||_2 \le 1}$, which is a solid Euclidean ball – a convex set. Likewise, linear inequalities like $a^T x \le b$ are convex constraints (they define half-spaces). Rule of thumb: If each $f_i(x)\le 0$ is convex in $x$, then the feasible set defined by all such inequalities is convex (since it's an intersection of convex sets).

    > note: Sometimes constraints are given in an equivalent form like $g(x)\ge 0$. You can always rewrite $g(x)\ge 0$ as $-g(x) \le 0$. So if you encounter $g(x)\ge 0$, check if $g(x)$ is concave (since $-g$ would then be convex). For instance, $g(x) = \text{log}(x)$ is concave on $x>0$, so the constraint $\log(x) \ge 3$ is equivalent to $-,\log(x) \le -3$, and $-,\log(x)$ is convex; hence this constraint defines a convex feasible set (here $x \ge e^3$).

- **Affine Equality Constraints:** These are constraints of the form
    $$
    h_j(x) = 0
    $$
    
    where $h_j(x)$ is an affine function, meaning $h_j(x) = a_j^T x + b_j$ for some constant vector $a_j$ and scalar $b_j$. Affine equalities are convex constraints because an affine set ${x \mid a^T x + b = 0}$ is actually a flat hyperplane (or a translate of a subspace), which is a convex set (recall: any line or plane is convex since the line segment between any two points on a line/plane stays on that line/plane; see Section A, Chapter 4). For example, $x_1 + 2x_2 = 5$ defines a line in $\mathbb{R}^2$, which is convex. Important: If an equality constraint is non-affine (e.g. $x_1 x_2 = 10$ or $x_1^2 + x_2^2 = 1$), the feasible set will typically be non-convex. For instance, $x_1 x_2 = 10$ is a hyperbola – a curved set that is not convex; $x_1^2 + x_2^2 = 1$ is the unit circle (just the boundary of a ball), which is not a convex set by itself (any line between two points on the circle goes through the inside which is not included in the feasible set). Thus, any nonlinear equality generally signals non-convexity (except in trivial cases like something that reduces to an affine constraint on a higher-dimensional space).

- **Convex Set Membership Constraints:** Sometimes constraints are given in the form “$x$ belongs to a set $C$”, denoted $x \in C$. In order for the problem to remain convex, the set $C$ must be convex. Many common constraint sets in optimization are indeed convex: for example, polyhedra defined by linear inequalities ($\{\, x \mid A x \le b \,\}$) are convex; norm balls like $\{\, x \mid \|x\|_p \le \alpha \,\}$ are convex sets (for any $p \ge 1$); the set of probability distributions $\{\, x \mid x_i \ge 0, \ \sum_i x_i = 1 \,\}$ is convex; the set of positive semidefinite matrices (in semidefinite programming) is convex. However, if $C$ is something like a finite set or has a discrete structure (e.g. "$x_i \in \{0,1\}$ for some component"), then $C$ is non-convex. Discrete constraints (integer or binary decisions) break convexity because the feasible region becomes a set of isolated points or separate chunks, not a single nicely connected region. As another example, the set $C = \{\, x : 1 \le \|x\|_2 \le 2 \,\}$ – an annulus (ring) between two circles – is not convex because it excludes the interior donut hole (a line between a point on the inner circle and a point on the outer circle would pass through the hole which is infeasible).

In summary, to have a convex feasible set, every constraint should individually carve out a convex region. All inequality constraints should be convex functions (producing convex sublevel sets), all equalities should be affine (flat), and any set-membership conditions should refer to convex sets. Since the intersection of any collection of convex sets is convex, these conditions ensure the overall feasible set is convex. If any one constraint is non-convex, the feasible region (being an intersection including a non-convex piece) will be non-convex – and hence the whole problem is not convex.

### Common Non-Convex Structures to Watch For

Having a mental checklist of "usual suspects" that break convexity is extremely useful. Often, by scanning the form of the objective and constraints, you can spot patterns that are inherently non-convex. Here are some common non-convex structures:

- **Bilinear or Multilinear Terms:** If the objective or a constraint involves a product of decision variables (e.g. a term like $x_i \cdot y_j$ or $x_1 x_2$), this is generally non-convex. For example, the function $f(x_1,x_2) = x_1 x_2$ is not convex on $\mathbb{R}^2$ (its Hessian is indefinite), and the constraint $x_1 x_2 \le c$ typically yields a hyperbolic region which is not convex. Bilinear terms often arise in problems like optimal power flow, portfolio optimization with products, or geometry problems – and they usually indicate the problem is hard (non-convex) unless you can reformulate them in convex form (sometimes via change of variables or relaxations).

- **Indefinite Quadratics:** A quadratic function $x^T Q x + c^T x$ is convex only if $Q$ is PSD. If $Q$ has even one negative eigenvalue (making it indefinite or negative definite), the function is not convex – it “curves downward” in at least one direction. For instance, $f(x_1,x_2) = x_1^2 - x_2^2$ (here $Q=\mathrm{diag}(1,-1)$) is a saddle-shaped function, not convex. So if you see a quadratic objective or constraint, check the matrix: a negative sign on a squared term or a “difference of squares” usually signals non-convexity (unless you can somehow constrain that term’s effect away).

- **Nonlinear Equality Constraints:** As mentioned, anything like $h(x)=0$ where $h$ is nonlinear (especially products, trigonometric equations, polynomial equations beyond first degree) is likely non-convex. A classic example is a fixed product or fixed norm constraint: $x_1 x_2 = 1$ or $||x||_2 = 1$. These carve out a curved manifold (a hyperbola or a sphere surface) without thickness – not convex. When you have such constraints, the feasible region often ends up disconnected or curved in a way that violates convexity. (One exception: something like $x^2+y^2=0$ is convex, but only because it implies $x=0,y=0$ – a trivial single-point set, which is convex. In general, non-affine equalities that allow multiple points are trouble.)

- **Ratios and Fractional Forms:** Objective terms like $\frac{p(x)}{q(x)}$ (where $p$ and $q$ are functions of $x$) or constraints like $\frac{f(x)}{g(x)} \le c$ are typically non-convex (these are quotient or fractional programs). A simple example: $f(x)=\frac{1}{x}$ on $x>0$ is convex, but if you have something like $\frac{x_1}{x_2}$ it’s neither convex nor concave on a broad domain. Many ratio problems can sometimes be convexified by clever transformations (e.g. transforming variables if $x_2$ is positive), but at face value, be cautious with fractional terms.

- **Discrete Variables or Logic:** If your problem involves integer variables (e.g. $x_i \in {0,1}$ or $x_i$ must be an integer) or logical constraints (if-then conditions, either-or constraints), then the feasible set is not convex. For example, requiring $x$ to be 0 or 1 means the feasible set is just two points, which is not convex (no line segment between 0 and 1 stays in the set except at the endpoints). These kinds of problems fall into combinatorial optimization or mixed-integer programming, which are generally NP-hard. They are solved with very different techniques (branch-and-bound, etc.) compared to convex optimization. There are ways to relax some discrete problems into convex ones (for example, dropping the integrality to allow continuous variables between 0 and 1, or using convex hulls), but the original discrete problem is non-convex.

- **“U-Shaped then Inverted U” Functions:** Any single-variable function that isn’t convex over its whole domain often shows a change in curvature. For example, $\sin x$ alternates between convex and concave regions; a function like $f(x) = x^3$ has $f''(x) = 6x$, which is negative for $x<0$ and positive for $x>0$, so $f(x)$ is not convex on the entire real line (it fails the Hessian test globally). If the objective function or a constraint function “bends” upward in some places and downward in others, it’s not globally convex. Recognizing these shapes (e.g. an objective with multiple local minima valleys separated by hills) is key — convex functions have a single valley (global bowl shape), whereas non-convex ones can have multiple valleys and peaks.

In summary, when scanning a problem, look out for these red flags. Spotting a single non-convex structure is enough to conclude the problem (as stated) is non-convex. Sometimes, such problems can be transformed or approximated by convex problems (this is a big area of research), but that’s beyond our current scope. Here, our goal is identification: know it when you see it.

### Examples: Convex or Not?

Let's solidify these concepts with a few intuitive examples. We will examine some optimization problems and apply the convexity checks. This will illustrate both positive examples (problems that are convex and why) and a negative example (a non-convex problem and how to tell).

1. **Unconstrained Least Squares (Convex):** 

    Problem: $\min_x f(x)$ where $f(x) = |Ax - b|_2^2 = (Ax - b)^T(Ax - b)$. This is the classic least squares problem. 

    Objective: $f(x)$ is a quadratic function. We can expand $f(x) = x^T (A^T A) x - 2b^T A x + |b|_2^2$. Here the Hessian is $Q = 2A^T A$. The matrix $A^T A$ is symmetric positive semidefinite (in fact positive definite if $A$ has full column rank). Thus $Q \succeq 0$, so $f(x)$ is convex. There are no constraints (the domain is all $x$, which is a convex set $\mathbb{R}^n$), so the feasible set is convex. By our criteria, this optimization problem is convex. Indeed, least squares has a unique global minimum given by the normal equations. If we plot $f(x)$ as a function (for example, if $x$ is one-dimensional, $f(x)$ is a simple parabola), it's clearly bowl-shaped and has no secondary minima. This matches our formal check: quadratic with PSD curvature → convex objective.

2. **LASSO Regression (Convex):**
    
     Problem: $\min_{x\in\mathbb{R}^n} ; \frac{1}{2}|Ax - b|_2^2 + \lambda |x|_1$, for some $\lambda > 0$. This is the LASSO optimization used in machine learning for sparse linear models. 
     
     Objective: It is a sum of two terms: $g(x) = \frac{1}{2}|Ax - b|_2^2$ (a convex quadratic, as just discussed) and $h(x) = \lambda |x|_1$ (the $L_1$ norm times a positive scalar, which is convex). The sum of convex functions is convex, so $f(x) = g(x)+h(x)$ is convex. There are no explicit constraints (again $x\in\mathbb{R}^n$), so the feasible set is all of $\mathbb{R}^n$ (convex). Thus, the LASSO problem is convex. Geometrically, the first term $|Ax-b|_2^2$ defines ellipsoidal level sets (nice and smooth), and the second term $|x|_1$ has level sets that are diamonds (corners along axes). The combination yields a convex “bowl with a corner-like shape around the bottom,” but crucially no local minima besides the global one. Many algorithms exploit this convexity (LASSO can be solved efficiently by coordinate descent or proximal gradient methods, leveraging the convex subgradient of $|x|_1$).

3. **Quadratic Program with Constraints (Convex)**:

     Problem: $\displaystyle \min_{x} ;\frac{1}{2} x^T Q x + c^T x \quad \text{s.t.}; A x \le d; Bx = e.$ This is a quadratic program (QP) with linear constraints. 
     
     To check convexity: The objective is convex if $Q \succeq 0$ (positive semidefinite). All terms are quadratic/linear which are easy to verify. Constraints: $A x \le d$ are linear inequality constraints – each of the form $a_i^T x \le d_i$ – which are convex (halfspaces). $B x = e$ are affine equalities – also convex (subspace translated by $e$). So if $Q$ is PSD, everything in this problem is convex: convex objective, convex feasible region. This becomes a convex optimization problem, often called a convex QP. For example, consider a specific QP:

    $$
    \begin{aligned}
    \min_{x_1,\, x_2} \quad & 2x_1^2 + x_2^2 + 3x_1 + 4x_2 \\
    \text{s.t.} \quad & x_1 + x_2 = 1, \\
    & x_1 \ge 0, \\
    & x_2 \ge 0.
    \end{aligned}
    $$


    Here 
    
    $$
    Q = \begin{pmatrix}
    4 & 0 \\
    0 & 2
    \end{pmatrix}
    $$
    
     (since objective $= 2x_1^2 + x_2^2 + \dots$), which is PSD. The equality $x_1+x_2=1$ is affine (a line), and $x_1 \ge 0, x_2 \ge 0$ are halfspaces (actually together $x_1,x_2\ge0$ define the first quadrant, which is convex). So all conditions check out – it's convex. We know from theory and experience that convex QPs can be solved efficiently to global optimality (there are Interior-Point solvers, etc.). Had $Q$ been non-PSD, the objective would be indefinite and the problem not convex – in that case, even though the constraints are still linear, the objective's shape would cause multiple local minima or unbounded directions.

4. **Non-Convex Optimization Example (Bilinear Constraint)**: 

    Problem: $\displaystyle \min_{x_1,x_2} ; x_1^2 + x_2^2 \quad \text{s.t.}; x_1 x_2 \ge 1.$ This is a made-up example to demonstrate a non-convex feasible region. 
    
    Objective: $f(x_1,x_2) = x_1^2 + x_2^2$ is convex (it's a bowl shaped paraboloid). Constraint: $x_1 x_2 \ge 1$ is not a convex constraint. The feasible set ${(x_1,x_2): x_1 x_2 \ge 1}$ consists of two disjoint regions: one where both $x_1$ and $x_2$ are positive (and their product exceeds 1), and one where both are negative (since a negative times a negative is positive, exceeding 1). The feasible region looks like two opposite quadrants cut away from the origin. This set is non-convex (you can take a point from the positive quadrant and one from the negative quadrant, and the line segment between them will cross through the region near the origin where $x_1 x_2 < 1$, which is infeasible). Thus, even though the objective is convex, the non-convex constraint breaks the problem's convexity. In fact, this problem has two separate “valleys” to minimize in (one in the $x_1,x_2 > 0$ region and one in the $x_1,x_2 < 0$ region). Each region has a local minimum (roughly when $x_1 x_2 = 1$ and the mass is evenly distributed, e.g. $x_1=x_2=1$ or $x_1=x_2=-1$), but there is no single global bowl – we have two distinct basins. A solver that doesn’t account for non-convexity might get stuck in one of them. This illustrates why breaking convexity is dangerous: local optimality need not imply global optimality. Indeed, $x_1=x_2=1$ gives $f=2$ and $x_1=x_2=-1$ gives $f=2$, and those are the best feasible points; but if you started at $(10, 0.1)$ which satisfies $10 \cdot 0.1 = 1$, any local descent would stay in the positive quadrant basin.


### Checklist for Verifying Convexity

To wrap up, here is a handy checklist you can use as a step-by-step guide when faced with a new optimization problem. This is essentially a flowchart in words for convexity verification:

1. **Problem Form:** Convert the problem to a clear standard form. Are you minimizing an objective? (If it’s a maximization, consider the equivalent minimization of the negative objective.) Make sure all constraints are written as either “$\le$” inequalities, “$=$” equalities, or set memberships.

2. **Objective Function Convexity:** Is the objective $f_0(x)$ convex? Check using the tools:

- Does $f_0(x)$ match a known convex function type or a sum/composition of convex functions?
- If it's smooth, is the Hessian $\nabla^2 f_0(x) \succeq 0$ for all $x$?
- If non-smooth, can you reason via subgradients or epigraph definition that it’s convex?
- Any suspect terms (e.g. products, non-convex patterns) in the objective? If no issues and all tests indicate convex, proceed. If not convex, you have a non-convex problem (stop here, unless you plan to reformulate it).

3. **Inequality Constraints:** For each constraint of the form $f_i(x) \le 0$, check $f_i(x)$:
- Is $f_i$ convex? If yes, this constraint defines a convex region (the sublevel set). If any $f_i$ is non-convex (e.g. concave or neither), that constraint is non-convex – the feasible set will not be convex. (Remember you can rewrite $g(x)\ge 0$ as $-g(x)\le 0$ and check $-g$ for convexity.)
- Also ensure the inequality is properly directed: a convex function should be $\le 0$, a concave function should be $\ge 0$ to be convex (since $g(x)\ge 0$ with $g$ concave is the same as $-g(x)\le 0$ with $-g$ convex).

4. **Equality Constraints:** For each $h_j(x) = 0$, determine if $h_j$ is affine. If yes, it’s fine (convex constraint). If not (e.g. quadratic or anything nonlinear), the problem is not convex (unless perhaps that equality can be eliminated or transformed in a special way). Non-affine equalities are a deal-breaker for convexity in almost all cases.

5. **Domain Constraints:** If the problem statement includes $x \in C$ for some set $C$, or implicit domain restrictions (like $x_i$ must be integer, or $x > 0$), verify those sets are convex. $x_i \ge 0$ (the nonnegative orthant) is convex. Norm balls, simplices, linear subspaces – all convex. But if $C$ is, say, ${0,1}^n$ (binary vectors), or a finite set of points, or defined by a weird non-convex condition, then the problem is not convex. Ensure there is no hidden discreteness.

6. **Intersections:** After checking all individual constraints, consider the intersection of all feasible conditions. If each constraint is convex, their intersection is convex. Double-check there isn’t an implicit “either-or” structure (which would actually be a union of sets, not an intersection). If it’s a straightforward intersection of convex sets, you’re good.

7. **Conclusion:** If all the above checks pass – objective is convex, all inequalities convex, all equalities affine, domain convex – then congratulations, the problem is convex! You can now be confident that any local optimum you find is globally optimal, and you can leverage the rich theory and algorithms of convex optimization. If any check fails, the problem is not convex. In that case, you might need to try reformulating the problem (sometimes through clever algebra or change of variables) or resort to global optimization techniques if you must solve it as is.

8. **(Optional Advanced Check:)** If you’re still in doubt, one advanced technique is to consider the Lagrange dual of the problem (see later chapters) – convex problems have a well-behaved duality theory. Another is to examine the Fenchel conjugate of the objective: for a convex function, the conjugate is well-defined and useful. These are beyond the scope of this checklist, but they can sometimes provide confirmation. Generally, though, the steps above are sufficient in practice.
in any optimization endeavor