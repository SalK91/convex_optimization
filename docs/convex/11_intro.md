# Chapter 1:  Introduction and Overview

Optimization is at the heart of most machine-learning methods. Whether training a linear model or a deep neural network, learning usually means adjusting parameters to minimize a loss that measures how well the model fits the data. Convex optimization is a particularly important and well-understood part of optimization. When both the objective and the constraints are convex, the problem has helpful properties:

1. No bad local minima: any local minimum is also the global minimum.  
2. Predictable behavior: algorithms like gradient descent have clear and well-studied convergence.  
3. Solutions are easy to verify: convex problems come with simple mathematical conditions that tell us when we have reached the optimum.

These features make convex optimization a reliable tool for building and analyzing machine-learning models. Even though many modern models are nonconvex, a surprising amount of ML still depends on convex ideas. Common loss functions, regularizers, and inner algorithmic steps often rely on convex structure.

This web-book is written for practitioners who have basic familiarity with optimization, especially gradient-based methods, and want to understand how convex optimization principles help guide reliable machine-learning practice.

 
## 1.1 Motivation: Optimization in Machine Learning

Many supervised learning problems can be written in a common form:

$$
\min_{x \in \mathcal{X}} 
\; \frac{1}{N}\sum_{i=1}^{N} \ell(a_i^\top x, b_i) 
+ \lambda R(x),
$$

where

- $\ell(\cdot,\cdot)$ is a loss function that measures how well the model predicts $b_i$ from $a_i$,  
- $R(x)$ is a regularizer that encourages certain structure (such as sparsity or small weights),  
- $\mathcal{X}$ is a set of allowed parameter values, often simple and convex.

Many widely used losses and regularizers are convex. Examples include least squares, logistic loss, hinge loss, Huber loss, the $\ell_1$ norm, and the $\ell_2$ norm. Convexity is what makes these problems tractable and allows them to be solved efficiently at scale using well-behaved optimization algorithms.

 
## 1.2 Convex Sets and Convex Functions — First Intuition

A set $\mathcal{C}$ is convex if, whenever you pick two points in the set, the line segment between them stays entirely inside the set:

$$
\theta x + (1-\theta)y \in \mathcal{C} 
\quad \text{for all } x,y \in \mathcal{C},\; \theta \in [0,1].
$$

Convex functions follow a similar idea. A function $f$ is convex if its graph never dips below the straight line connecting two points on the function:

$$
f(\theta x + (1-\theta)y)
\le
\theta f(x) + (1-\theta) f(y).
$$

Intuitively, convex functions look like bowls: they curve upward and have at most one global minimum. Affine functions are both convex and concave, and quadratics with positive semidefinite Hessians are convex. Many ML loss functions share this shape, which makes them easy to optimize.

 
## 1.3 Why Convex Optimization Remains Central in ML

Although many modern models are nonconvex, convex optimization continues to play a major role in three ways:

1. Convex surrogate losses: Losses such as logistic, hinge, and Huber are convex substitutes for harder objectives like the $0\text{–}1$ loss. They make optimization practical while still leading to models that generalize well.

2. Convex subproblems inside larger algorithms:  Many nonconvex methods solve convex problems as part of their inner loop. Examples include least-squares steps in matrix factorization, proximal updates in regularized learning, and simple convex problems that appear in line-search procedures.

3. Implicit bias in linear models:  In overparameterized linear least-squares problems, gradient descent starting from zero converges to the minimum-norm solution. This phenomenon helps explain generalization and implicit regularization in linear and kernel models.

These roles make convex optimization a key component of modern ML toolkits, even when the main model is nonconvex.

 
## 1.4 From Global Optima to Algorithms

A major advantage of convex optimization is that it eliminates the possibility of non-global local minima. For a differentiable convex function on an open domain:

$$
\nabla f(x^*) = 0 
\quad \Rightarrow \quad
x^* \text{ is a global minimizer}.
$$

This means that simply finding a point where the gradient is zero is enough. For constrained or nondifferentiable problems, optimality is checked using subgradients or KKT conditions:

$$
0 \in \partial f(x^*) + N_{\mathcal{X}}(x^*),
$$

where $N_{\mathcal{X}}(x^*)$ represents the outward directions that are blocked by the constraint set. These conditions are useful because many iterative algorithms aim to drive the gradient or subgradient toward zero.

 
## 1.5 Canonical Convex ML Problems at a Glance

| Problem | Objective | Typical Solver |
|--------|-----------|----------------|
| Least squares | $\|A x - b\|_2^2$ | Gradient descent, conjugate gradient |
| Ridge regression | $\|A x - b\|_2^2 + \lambda\|x\|_2^2$ | Closed form, gradient methods |
| LASSO | $\|A x - b\|_2^2 + \lambda\|x\|_1$ | Proximal gradient (ISTA/FISTA) |
| Logistic regression | $\sum_i \log(1+\exp(-y_i a_i^\top x)) + \lambda\|x\|_2^2$ | Newton, quasi-Newton, SGD |
| SVM (hinge loss) | $\tfrac{1}{2}\|x\|^2 + C\sum_i \max(0,1-y_i a_i^\top x)$ | Subgradient, coordinate methods, SMO |
| Robust regression | $\|A x - b\|_1$ | Linear programming |
| Elastic Net | $\|A x-b\|_2^2 + \lambda_1\|x\|_1 + \lambda_2\|x\|_2^2$ | Coordinate descent |

These problems illustrate how convex models appear throughout ML.

 
## 1.6 Web-Book Roadmap and How to Use It

| Question | Where to Look | Key Idea |
|---------|----------------|----------|
| What makes a function or set convex? | Chapters 2–5 | Geometry and basic properties of convexity |
| How do gradients, subgradients, and KKT conditions define optimality? | Chapters 6–9 | Optimality conditions and duality |
| How are convex problems solved in practice? | Chapters 10–14 | First-order, second-order, and interior-point methods |
| How to choose an algorithm for a given optimization problem? | Chapters 15–17 | Large-scale and structured optimization techniques |

 