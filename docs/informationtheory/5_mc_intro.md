Many problems in machine learning require computing expectations, marginal likelihoods, or posterior distributions of the form

$$
p(z|x) = \frac{p(x,z)}{p(x)}, \qquad 
p(x) = \int p(x,z)\,dz.
$$

For most realistic models, the integral in the denominator is intractable. Modern machine learning therefore relies on several approximation strategies, each with different assumptions, strengths, and limitations. These approaches form a probability toolbox for inference.

This section introduces four major families of methods:

1. complete enumeration  
2. Laplace approximation  
3. Monte Carlo methods  
4. variational methods  

Subsequent chapters expand on these ideas, beginning with a deeper discussion of Monte Carlo sampling.

 
## 1. Complete Enumeration

Complete enumeration computes the integral exactly by summing or integrating over all possible latent configurations:

$$
p(x) = \sum_z p(x,z)
\quad \text{or} \quad
p(x) = \int p(x,z)\,dz.
$$

This is feasible only when:

- the latent variable is low dimensional  
- the domain is small or discrete  
- the joint distribution has a simple closed form  

Although conceptually straightforward, complete enumeration becomes impossible as dimensionality increases. It serves mainly as a theoretical reference point.

 
## 2. Laplace Approximation

The Laplace method approximates an intractable posterior by a Gaussian distribution centered at its mode.

Given a posterior

$$
p(z|x) \propto p(x,z),
$$

the Laplace approximation fits a Gaussian distribution

$$
q(z|x) \approx \mathcal{N}(z_{\text{MAP}}, H^{-1}),
$$

where:

- $z_{\text{MAP}}$ is the mode of $p(z|x)$  
- $H$ is the Hessian of $-\log p(z|x)$ at the mode  

This method assumes the posterior is approximately unimodal and locally Gaussian. It is fast and easy to compute, but may be inaccurate when the posterior is skewed or multimodal.

 
## 3. Monte Carlo Methods

Monte Carlo methods approximate integrals using random samples. The central idea is:

$$
\mathbb{E}_{p(z|x)}[f(z)] 
\approx \frac{1}{N}\sum_{i=1}^N f(z_i),
\qquad z_i \sim p(z|x).
$$

Monte Carlo estimators do not require closed-form integrals and scale well to high dimensions. They are widely used in Bayesian inference, reinforcement learning, generative modeling, and probabilistic programming.

Sampling strategies fall into two groups:

- independent sampling  
- Markov chain–based sampling (MCMC)  

The next chapter explains Monte Carlo and sampling methods in detail.

 
## 4. Variational Methods

Variational methods replace an intractable posterior with a tractable family of approximations. Instead of sampling directly from $p(z|x)$, we introduce a distribution $q(z|x)$ and optimize it to be close to the true posterior. The objective is to minimize

$$
D_{\text{KL}}(q(z|x)\|p(z|x)).
$$

Because $p(z|x)$ is unknown, variational inference rewrites this quantity using the Evidence Lower Bound (ELBO):

$$
\log p(x)
=
\mathcal{L}(x) + D_{\text{KL}}(q(z|x)\|p(z|x)).
$$

Maximizing the ELBO yields a tractable approximation to Bayesian inference. Variational methods power VAEs, Bayesian neural networks, diffusion models, and many modern probabilistic approaches.

 
## Summary

Approximate inference methods can be understood as four major strategies:

- complete enumeration: exact but rarely feasible  
- Laplace approximation: fast Gaussian approximation near the mode  
- Monte Carlo methods: sampling-based numerical estimation  
- variational methods: optimization-based posterior approximation  

Monte Carlo sampling is the most flexible approach and serves as the backbone of Bayesian computation. The next chapter develops Monte Carlo and sampling techniques in detail.
