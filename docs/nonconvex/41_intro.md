# Chapter 1: Non-Convex Optimization Fundamentals


## 1.1 Why Non-Convexity Matters

Convex optimization ensures a unique global minimum and strong theoretical guarantees.
However, many practical problems in machine learning, deep learning, control, and physics are non-convex:

* Neural network loss surfaces
* Reinforcement learning value functions
* Matrix factorization
* Clustering and combinatorial tasks

These landscapes cannot be handled efficiently with traditional convex methods.

---

## 1.2 Characteristics of Non-Convex Landscapes

| Property              | Description                | Consequence                        |
| --------------------- | -------------------------- | ---------------------------------- |
| Multiple local minima | Many suboptimal valleys    | Gradient descent may get trapped   |
| Saddle points         | Flat or neutral zones      | Slow or no convergence             |
| Discontinuities       | Non-differentiable regions | Gradients undefined                |
| Non-linearity         | Coupled variables          | Non-trivial curvature and topology |

Visualization of 2D loss surfaces often reveals chaotic or fractal-like geometry.

---

## 1.3 Gradient-Based Methods and Their Limits

Even though SGD, Adam, and similar methods dominate deep learning, they:

* Depend heavily on initialization
* May converge to poor local minima or plateaus
* Are sensitive to learning rate and batch size
* Cannot handle discrete or combinatorial variables

This motivates global search strategies that can explore the space more broadly.

---

## 1.4 Toward Global Optimization

Global optimization aims to find near-optimal solutions without convexity assumptions.
Two main families exist:

1. Deterministic global methods — exhaustive, branch-and-bound, interval analysis
2. Stochastic and metaheuristic methods — probabilistic, adaptive, and nature-inspired

The remainder of this book focuses on the latter, due to their flexibility and robustness in black-box settings.
