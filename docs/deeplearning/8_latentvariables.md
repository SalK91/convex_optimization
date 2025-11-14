## 1. Generative Modelling

### 1.1 What Are Generative Models?
- Probabilistic models of high-dimensional data.
- Describe how observations are generated from underlying processes.
- Key focus: modelling dependencies between dimensions and capturing the full data distribution.

### 1.2 Why They Matter
Generative models can:
- Estimate data density (detect outliers, anomalies).  
- Enable compression (encode → decode).  
- Map between domains (e.g., translation, text-to-speech).  
- Support model-based RL (predict future states).  
- Learn representations from raw data.  
- Improve understanding of data structure.

 

## 1.3 Types of Generative Models in Deep Learning

### (a) Autoregressive Models
Model joint distribution via chain rule:
$$
p(x) = \prod_{i=1}^D p(x_i \mid x_{<i})
$$

Trained with maximum likelihood

Examples:

- RNN/Transformer LMs  
- NADE  
- PixelCNN / WaveNet

Pros:

- Easy training (max. likelihood).
- No sampling during training.

Cons:

- Slow generation (sequential).
- Often capture local structure better than global structure.

 

### (b) Latent Variable Models
Introduce an unobserved latent variable $z$:

- Prior: $p(z)$  
- Likelihood: $p_\theta(x\mid z)$  

Joint:
$$
p_\theta(x, z) = p_\theta(x\mid z)\,p(z)
$$

Pros

- Flexible & interpretable  
- Natural for representation learning  
- Fast generation  

Cons
- Require approximate inference unless specially designed (e.g., invertible models).

 

### (c) Implicit Models (GANs)

- Define a generator $G(z)$ with no explicit likelihood.
- Trained adversarially using a discriminator.

Pros

- Extremely realistic samples  
- Fast sampling  

Cons

- Cannot evaluate $p(x)$  
- Mode collapse  
- Training instability  

## 2. Latent Variable Models & Inference

### 2.1 What is a Latent Variable Model (LVM)?
A latent variable model introduces an unobserved variable $z$ that explains the observed data $x$.

Model components:

- Prior over latent variables:  

  $$
  p(z)
  $$

- Likelihood / decoder mapping latent → observation:  

  $$
  p_\theta(x \mid z)
  $$

Joint distribution:

$$
p_\theta(x, z) = p_\theta(x \mid z)\,p(z)
$$

Marginal likelihood (what we want to maximize when training):

$$
p_\theta(x) = \int p_\theta(x \mid z)\,p(z)\,dz
$$


### 2.2 Intuition: Latents as “Explanations”

- A particular value of $z$ is a hypothesis about hidden causes that produced $x$.

- Generation = sample latent → map it to data:
  $$
  z \sim p(z), \quad x \sim p_\theta(x\mid z)
  $$

Most of the article focuses on the inverse of this:  
recovering $z$ from $x$.

## 2.3 What Is Inference?
Inference means computing the posterior:
$$
p_\theta(z \mid x) = \frac{p_\theta(x\mid z)\,p(z)}{p_\theta(x)}
$$

Why it matters:

- Explains the observation (which latents likely produced it?)
- Needed inside maximum-likelihood training  
  (the gradient depends on the posterior!)


## 2.4 Inference Requires the Marginal Likelihood

To compute the posterior, we need:
$$
p_\theta(x) = \int p_\theta(x \mid z)p(z)\,dz
$$
This integral is often intractable.

Thus exact inference usually fails except in special models (e.g., mixture models, linear-Gaussian).


## 2.5 Example: Mixture of Gaussians
Model:

- Choose cluster $k$  
- Sample $x$ from Gaussian for that cluster

Posterior:
$$
p(k \mid x)
= 
\frac{\pi_k\,\mathcal{N}(x \mid \mu_k, \Sigma_k)}
{\sum_j \pi_j \mathcal{N}(x \mid \mu_j, \Sigma_j)}
$$

This model is tractable because:

- Finite number of discrete states  
- Closed-form posterior

## 2.6 The Need for Inference in Learning

### Maximum Likelihood as the Core Training Principle
Maximum Likelihood Estimation (MLE) is the dominant method for fitting probabilistic models.  We choose parameters $\theta$ that make the observed training data as probable as possible:

$$
\theta^* = \arg\max_\theta \sum_{i} \log p_\theta(x^{(i)})
$$

For latent variable models, the marginal likelihood is:
$$
\log p_\theta(x) = \log \int p_\theta(x, z)\,dz
$$
This integral is rarely tractable, which makes direct maximization difficult.


### Why Optimization Is Hard in Latent Variable Models

- The log-likelihood involves an integral (or sum) over the latent variables $z$.  
- Because this integral usually has no closed form, we must use iterative optimization methods.

Common approaches:

1. Gradient-based optimization (e.g., gradient descent)
2. Expectation-Maximization (EM)

Below we explain why inference (computing the posterior $p_\theta(z \mid x)$) is essential for both.


## 2.6.1 Gradient-Based Learning Requires the Posterior

Using the identity:

$$
\nabla_\theta \log p_\theta(x)
=
\mathbb{E}_{p_\theta(z\mid x)}[\nabla_\theta \log p_\theta(x, z)]
$$

> Differentiate the log-marginal
> 
> $$\nabla_\theta \log p_\theta(x)
> =
> \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}$$
> 
> Using: $p_\theta(x)=\int p_\theta(x,z)\,dz,$
> 
> differentiate under the integral:
> 
> $$\nabla_\theta p_\theta(x)
> = \nabla_\theta \int p_\theta(x,z)\,dz
> = \int \nabla_\theta p_\theta(x,z)\,dz$$
> 
> Combine:
> 
> $$\nabla_\theta \log p_\theta(x)
> =
> \frac{1}{p_\theta(x)} \int \nabla_\theta p_\theta(x,z)\,dz$$
> 
> Apply the log-derivative identity
> 
> The identity:
> 
> $$\nabla_\theta p_\theta(x,z)
> = p_\theta(x,z)\,\nabla_\theta \log p_\theta(x,z)$$
> 
> Substitute:
> 
> $$\nabla_\theta \log p_\theta(x)
> =
> \frac{1}{p_\theta(x)}
> \int p_\theta(x,z)\,\nabla_\theta \log p_\theta(x,z)\,dz$$
> 
> Recognize the posterior
> 
> Bayes’ rule:
> 
> $$p_\theta(z\mid x)
> = \frac{p_\theta(x,z)}{p_\theta(x)}$$
> 
> Substitute into the integral:
> 
> $$\nabla_\theta \log p_\theta(x)
> =
> \int
> \frac{p_\theta(x,z)}{p_\theta(x)}
> \nabla_\theta \log p_\theta(x,z)\,dz$$
> 
> This becomes:
> 
> $$\nabla_\theta \log p_\theta(x)
> =
> \int p_\theta(z\mid x)\,\nabla_\theta \log p_\theta(x,z)\,dz$$
> 
> Write as an expectation
> 
> $$\nabla_\theta \log p_\theta(x)
> =
> \mathbb{E}_{p_\theta(z\mid x)}
> \left[
> \nabla_\theta \log p_\theta(x,z)
> \right]$$
> 
> ---
> 
> ## Final Result
> 
> $$\boxed{
> \nabla_\theta \log p_\theta(x)
> =
> \mathbb{E}_{p_\theta(z\mid x)}[\nabla_\theta \log p_\theta(x, z)]
> }$$
> 
> ---
> 
> ## Interpretation
> 
> **Computing the gradient of the marginal log-likelihood requires computing an expectation under the posterior.**
> 
> This means:
> - gradient descent on latent-variable models requires inference  
> - EM requires inference  
> - variational inference approximates $p(z\mid x)$  
> - VAEs rely on this identity  

This means:

> To compute the gradient of the marginal likelihood, we must take an expectation under the *posterior*  
> $p_\theta(z\mid x)$.

So:

- We cannot compute $\nabla_\theta \log p_\theta(x)$ without knowing the posterior.
- Inference becomes part of every gradient step.
- If inference is intractable → gradient is intractable.

This is why approximate inference (variational inference, MCMC) is essential for deep latent-variable models.


## 2.6.2 Expectation-Maximization (EM) Also Requires Inference

EM is an alternative to gradient descent for maximizing likelihood.

### E-step:  
Compute (or approximate) the posterior:
$$
q(z) \approx p_\theta(z \mid x)
$$
This assigns responsibilities to each latent configuration.

### M-step:  
Update parameters by maximizing the expected complete-data log-likelihood:
$$
\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{q(z)}[\log p_\theta(x, z)]
$$

Thus, the E-step directly requires inference.


## 2.7 Why Exact Inference Is Hard

### Continuous latents:
- Require multidimensional integration over nonlinear likelihoods.

### Discrete latents:
- Require summing over exponentially many configurations.

Only a few cases allow closed-form inference:

- Mixture models  
- Linear Gaussian systems  
- Invertible / flow-based models (covered next)

## 2.8 Two Strategies to Handle Intractability

### 1. Design tractable models

- Invertible models (normalizing flows)
- Autoregressive latent structures  
Pros: exact inference  
Cons: restricted model class

### 2. Approximate inference

- Use approximations to posterior $p(z \mid x)$  
- Variational Inference or MCMC  
Pros: flexible, expressive models  
Cons: introduces approximation error

