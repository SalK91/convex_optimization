# Statistical Estimation and Maximum Likelihood

 
## 1. Maximum Likelihood Estimation (MLE)

Suppose we have a family of probability densities

$$
p_x(y), \quad x \in \mathcal{X},
$$

where $x$ (often written as $\theta$ in statistics) is the parameter to be estimated.  

- $p_x(y) = 0$ for invalid parameter values $x$.  
- The function $p_x(y)$, viewed as a function of $x$ with $y$ fixed, is called the **likelihood function**.  
- The **log-likelihood** is defined as  

$$
\ell(x) = \log p_x(y).
$$  

- The **maximum likelihood estimate (MLE)** is  

$$
\hat{x}_{\text{MLE}} \in \arg\max_{x \in \mathcal{X}} \; p_x(y) 
= \arg\max_{x \in \mathcal{X}} \; \ell(x).
$$  

 
### Convexity Perspective

- If $\ell(x)$ is **concave in $x$** for each fixed $y$, then the MLE problem is a **convex optimization problem**.  
- Important distinction: this requires concavity in $x$, not in $y$.  
  - Example: $p_x(y)$ may be a **log-concave density in $y$** (common in statistics),  
  - but this does **not imply** that $\ell(x)$ is concave in $x$.  

Thus, convexity of the MLE depends on the parameterization of the distribution family.  

 
## 2. Linear Measurements with IID Noise

Consider the linear measurement model:

$$
y_i = a_i^\top x + v_i, \quad i = 1, \ldots, m,
$$

where  

- $x \in \mathbb{R}^n$ is the **unknown parameter vector**,  
- $a_i \in \mathbb{R}^n$ are known measurement vectors,  
- $v_i$ are **i.i.d. noise variables** with density $p(z)$,  
- $y \in \mathbb{R}^m$ is the vector of observed measurements.  

 
### Likelihood Function

Since the noise terms are independent:

$$
p_x(y) = \prod_{i=1}^m p\!\left(y_i - a_i^\top x\right).
$$

Taking logs:

$$
\ell(x) = \log p_x(y) 
= \sum_{i=1}^m \log p\!\left(y_i - a_i^\top x\right).
$$

 
### MLE Problem

The MLE is any solution to:

$$
\hat{x}_{\text{MLE}} \in \arg\max_{x \in \mathbb{R}^n} \; \sum_{i=1}^m \log p\!\left(y_i - a_i^\top x\right).
$$

 
### Convexity Note

- If $p(z)$ is **log-concave in $z$**, then $\log p(y_i - a_i^\top x)$ is concave in $x$.  
- Therefore, under log-concave noise distributions (e.g. Gaussian, Laplace, logistic), the MLE problem is a **concave maximization problem**, hence equivalent to a **convex optimization problem** after sign change:

$$
\min_x \; -\ell(x).
$$
