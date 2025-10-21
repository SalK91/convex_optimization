Convex optimization is a cornerstone of modern applied mathematics, machine learning, and artificial intelligence. At its core, it concerns the problem of identifying the optimal point within a structured space, such as the weights of a neural network or the coefficients of a regression model, under the crucial guarantee that every local optimum is also a global one. This property sets convex optimisation apart: it offers reliability, interpretability, and provable convergence, qualities that are increasingly essential in the design of trustworthy machine learning systems.

For practitioners, convex optimisation is not merely a theoretical curiosity. It quietly powers widely-used methods including Lasso, Ridge Regression, Logistic Regression, Support Vector Machines, Principal Component Analysis, and numerous convex relaxations of probabilistic models. To understand convex optimisation is to gain the ability to:

* Identify whether a problem admits a convex formulation and is therefore tractable.
* Select or design algorithms suited to a given machine learning objective.
* Anticipate convergence behaviour rather than merely observe it empirically.
* Engineer models and regularisation schemes with explicit guarantees.

This text is structured around three tightly connected themes:

1. **Foundations of convex optimisation:** The geometry of convex sets, properties of convex functions, and the structure of optimisation landscapes that enable global guarantees.
2. **Algorithms for convex problems:** Gradient-based methods, proximal operators, duality principles, and the interplay between first-order and second-order techniques.
3. **Convex models in machine learning:** Sparse regression, support vector machines, principal component analysis, and convex relaxations of more complex probabilistic models.

The aim is not only to present the theory but to develop the reader’s ability to recognise convex structure in practice and to apply it with confidence. In doing so, convex optimisation becomes not just a mathematical topic but a methodological lens through which modern machine learning can be understood and designed with rigour.
