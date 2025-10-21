# Linear Operators and Operator Norms

Linear operators and their norms quantify how matrices or linear maps amplify vectors. Understanding these concepts is crucial for step size selection, conditioning, low-rank approximations, and stability of optimization algorithms. This chapter introduces operator norms, special cases, and the singular value decomposition, highlighting their role in convex optimization and machine learning.

## Operator Norms

Let $A: \mathbb{R}^n \to \mathbb{R}^m$ be a linear operator (matrix). The operator norm induced by vector norms $\|\cdot\|_p$ and $\|\cdot\|_q$ is defined as

$$
\|A\|_{p \to q} = \sup_{x \neq 0} \frac{\|Ax\|_q}{\|x\|_p} = \sup_{\|x\|_p \le 1} \|Ax\|_q.
$$

Intuition: $\|A\|_{p \to q}$ measures the maximum **amplification factor** of a vector under $A$ from $\ell_p$ space to $\ell_q$ space.

Special cases:

- $\|A\|_{1 \to 1} = \max_{j} \sum_{i} |A_{ij}|$ (maximum absolute column sum)  
- $\|A\|_{\infty \to \infty} = \max_{i} \sum_{j} |A_{ij}|$ (maximum absolute row sum)  
- $\|A\|_{2 \to 2} = \sigma_{\max}(A)$, the largest singular value of $A$  


## Singular Value Decomposition (SVD)

Any matrix $A \in \mathbb{R}^{m \times n}$ admits a singular value decomposition:

$$
A = U \Sigma V^\top,
$$

where:

- $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal matrices  
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with non-negative entries $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$, called **singular values**

Interpretation:

- $V^\top$ rotates coordinates in the input space  
- $\Sigma$ scales each coordinate along its principal direction  
- $U$ rotates coordinates in the output space

Geometric intuition: The action of $A$ on the unit sphere in $\mathbb{R}^n$ produces an **ellipsoid** in $\mathbb{R}^m$, with axes given by singular vectors and lengths given by singular values.

## Applications in Optimization and Machine Learning

- **Conditioning:** The ratio $\kappa(A) = \sigma_{\max}(A) / \sigma_{\min}(A)$ determines sensitivity of linear systems $Ax = b$ and least-squares problems to perturbations. Poor conditioning slows convergence.  
- **Low-rank approximations:** The best rank-$k$ approximation of $A$ (in Frobenius or spectral norm) is obtained by truncating its SVD. This is widely used in PCA and collaborative filtering.  
- **Preconditioning:** Linear transformations can be preconditioned using SVD to accelerate gradient-based methods.  
- **Step amplification:** For gradient descent on quadratic objectives $f(x) = \frac{1}{2}x^\top A x - b^\top x$, the largest singular value $\sigma_{\max}(A)$ determines the safe step size $\alpha \le 2 / \sigma_{\max}(A)^2$.

## Numerical Considerations

- Computing full SVD is expensive for large matrices; truncated SVD or randomized SVD is preferred in high dimensions.  
- Operator norms help identify **directions of largest amplification** and potential instability.  
- Column scaling or whitening of matrices improves conditioning and convergence of iterative optimization algorithms.

## Summary and Optimization Connections

- Operator norms quantify maximum amplification of vectors under linear transformations.  
- Singular values provide geometric insight into the shape of linear maps and determine condition numbers.  
- These tools are essential for step size selection, preconditioning, and low-rank modeling in convex optimization and ML.
