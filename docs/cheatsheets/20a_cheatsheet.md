## First-Order Methods
| Method                       | Problem Type                            | Assumptions                            | Core Update Rule                                                                                    | Applications                                               |
| ---------------------------- | --------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Gradient Descent (GD)        | Unconstrained smooth (convex/nonconvex) | Differentiable, L-smooth (Lipschitz ∇) | $x_{k+1} = x_k - \eta \nabla f(x_k)$                                                                | Logistic regression, deep neural networks                  |
| Nesterov’s Accelerated GD    | Smooth convex (fast convergence)        | Convex, L-smooth                       | $y_k = x_k + \frac{k-1}{k+2}(x_k - x_{k-1})$, $x_{k+1} = y_k - \eta \nabla f(y_k)$ (momentum-based) | Same as GD (e.g. convex regression, accelerating training) |
| (Polyak) Heavy-ball Momentum | Unconstrained smooth (accelerated)      | f convex, differentiable, β ∈ (0,1)    | $x_{k+1} = x_k - \eta \nabla f(x_k) + \beta(x_k - x_{k-1})$ (momentum)                              | Deep learning (momentum SGD), convex optimization          |

## Second-Order Methods
| Method              | Problem Type                  | Assumptions                               | Core Update Rule                                                                         | Applications                                    |
| ------------------- | ----------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Newton’s Method     | Unconstrained smooth (convex) | Twice-differentiable, Hessian PD          | $x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1}\nabla f(x_k)$                                    | Logistic regression (IRLS), convex optimization |
| BFGS (Quasi-Newton) | Unconstrained smooth (convex) | f twice-differentiable (no exact Hessian) | Solve $B_k p_k = -\nabla f(x_k)$ then $x_{k+1}=x_k+p_k$; update $B_k$ by the secant rule | Large-scale learning (L-BFGS for ML models)     |

## Proximal & Projected Methods
| Method                   | Problem Type                          | Assumptions                                   | Core Update Rule                                                          | Applications                               |
| ------------------------ | ------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------ |
| Proximal Gradient (ISTA) | Convex composite (smooth + nonsmooth) | $f(x)=g(x)+h(x)$ where $g$ smooth, $h$ convex | $x_{k+1} = \mathrm{prox}_{\alpha h}(x_k - \alpha \nabla g(x_k))$          | LASSO, elastic net, regularized regression |
| Fast Proximal (FISTA)    | Convex composite (accelerated)        | same as ISTA, with momentum                   | Like ISTA but with Nesterov momentum (e.g. update $y_k$ as above)         | Same as ISTA (faster convergence)          |
| Projected Gradient (PG)  | Convex constrained (onto set $C$)     | $f$ L-smooth; $C$ convex (efficient proj)     | $x_{k+1} = \Pi_C!\bigl(x_k - \eta \nabla f(x_k)\bigr)$ (project onto $C$) | Constrained problems (e.g. box/QP)         |

## Interior-Point & Barrier Methods
| Method         | Problem Type                       | Assumptions                              | Core Update Rule                                                                                 | Applications                       |
| -------------- | ---------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------- |
| Interior-Point | Convex with inequality constraints | Convex problem, self-concordant barriers | Solve Newton step on barrier-augmented objective $\phi(x)=f(x)+\frac{1}{t}\sum_i -\log(-g_i(x))$ | Linear/Quadratic programming, SDPs |

## Stochastic & Mini-Batch Methods
| Method                    | Problem Type                       | Assumptions                                            | Core Update Rule                                                                                                                                              | Applications                   |
| ------------------------- | ---------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Stochastic Gradient (SGD) | Stochastic (often nonconvex)       | Unbiased gradient samples, decaying step-size          | $x_{k+1} = x_k - \eta ,\nabla f_{i_k}(x_k)$ (use random sample or mini-batch)                                                                                 | Online learning, deep networks |
| Adaptive SGD (Adam)       | Stochastic nonconvex (NN training) | Bounded variances; (β<sub>1</sub>,β<sub>2</sub>∈[0,1)) | $m_k=β_1 m_{k-1}+(1-β_1)g_k$, $v_k=β_2 v_{k-1}+(1-β_2)g_k^2$; $\hat m_k,\hat v_k$ bias-corrected, <br>$x_{k+1}=x_k - \eta,\frac{\hat m_k}{\sqrt{\hat v_k}+ε}$ | Deep learning (CNNs, RNNs)     |


## ADMM (Alternating Direction Method of Multipliers)
| Method | Problem Type                         | Assumptions                                  | Core Update Rule                                                                                                                                                                                | Applications                                  |
| ------ | ------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| ADMM   | Split convex with linear constraints | $f,g$ convex; equality constraints $Ax+Bz=c$ | $x^{k+1}=\arg\min_x\Bigl{f(x)+\frac{\rho}{2}|Ax+Bz^k-c+w^k|^2\Bigr}$,  <br>$z^{k+1}=\arg\min_z\Bigl{g(z)+\frac{\rho}{2}|Ax^{k+1}+Bz-c+w^k|^2\Bigr}$,  <br>$w^{k+1}=w^k + (Ax^{k+1}+Bz^{k+1}-c)$ | Consensus optimization, distributed ML, LASSO |


## Majorization–Minimization (MM)
| Method                    | Problem Type                  | Assumptions    | Core Update Rule                | Applications              |                   |                          |              |                                                                  |
| ------------------------- | ----------------------------- | -------------- | ------------------------------- | ------------------------- | ----------------- | ------------------------ | ------------ | ---------------------------------------------------------------- |
| Majorization–Minimization | General nonconvex (or convex) | Surrogate $g(x | x_k)$ majorizes $f(x)$ at $x_k$ | $x_{k+1} = \arg\min_x,g(x | x_k)$, where $g(x | x_k)\ge f(x)$ and $g(x_k | x_k)=f(x_k)$ | EM algorithm (mixture models), IRLS for ℓ<sub>p</sub> regression |
