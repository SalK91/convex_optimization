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
> To compute the gradient of the marginal likelihood, we must take an expectation under the *posterior*  $p_\theta(z\mid x)$.

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

## 3. Invertible Models & Exact Inference

### 3.1 What Are Invertible Models?
Invertible models (also called normalizing flows) are latent variable models where:

- The latent variable $z$ and data $x$ have the same dimensionality
- There exists an invertible, differentiable mapping  
  $$
  x = f_\theta(z)
  $$
- Because $f_\theta$ is invertible:
  $$
  z = f_\theta^{-1}(x)
  $$

Key property:  
Inference is exact and trivial — simply apply the inverse function.


## 3.2 Generative Process
To generate a sample:

1. Sample $z \sim p(z)$ (usually a simple prior like $\mathcal{N}(0, I)$)
2. Transform via  
   $$
   x = f_\theta(z)
   $$

Thus, the model pushes forward the prior distribution through a sequence of invertible transformations.


## 3.3 Why Are Invertible Models Attractive?
- Exact inference:  
  $$
  p_\theta(z \mid x)
  $$  
  is computed by a single function evaluation (no approximation needed).

- Exact likelihood:  
  Can compute $\log p_\theta(x)$ exactly using the change-of-variables formula.


## 3.4 Change of Variables for Likelihood

Given an invertible mapping $x = f_\theta(z)$:

$$
p_\theta(x) = p(z) \left| \det \left( \frac{\partial f_\theta^{-1}(x)}{\partial x} \right) \right|
$$

Equivalently, using $z = f_\theta^{-1}(x)$:

$$
\log p_\theta(x)
=
\log p(z)
+
\log \left| \det J_{f_\theta^{-1}}(x) \right|
$$

Where:

- $J_{f_\theta^{-1}}$ is the Jacobian matrix of the inverse map  
- The determinant accounts for volume change introduced by transformation


## 3.5 Example: Independent Component Analysis (ICA)

ICA is the simplest invertible model:

- Latent prior:  factorial prior
  $$
  p(z) = \prod_i p(z_i)
  $$
  with non-Gaussian heavy-tailed components
- Linear invertible mixing:  
  $$
  x = A z
  $$

Inference:
$$
z = A^{-1} x
$$

ICA recovers independent sources that explain the observed signal.


## 3.6 Building Complex Invertible Models

Modern flows build $f_\theta$ by composing many simple invertible layers:

$$
f_\theta = f_K \circ f_{K-1} \circ \dots \circ f_1
$$

Composition of invertible functions is invertible.

Building blocks:

- Linear transforms
- Autoregressive flows (IAF, MAF)
- Coupling layers (RealNVP, Glow)
- Residual flows
- Sylvester flows

Design goal:

> Each layer must have a tractable inverse and a tractable Jacobian determinant.


## 3.7 Advantages & Limitations

### Advantages
- Exact inference  
- Exact log-likelihood  
- Fast, parallel sampling  
- Useful as components in larger probabilistic models

### Limitations
- Latent and data dimensions must match  
- Latents must be continuous  
- Observations must be continuous or quantized  
- Very deep flows require large memory  
- Hard to encode strong structure or sparsity  

> Flows are powerful but rigid: they trade flexibility in modeling for tractability in inference.

## Mar

## 4. Variational Inference (VI)

### 4.1 Why Variational Inference?

In many latent variable models, the true posterior
$$
p_\theta(z \mid x)
$$
is intractable because computing
$$
p_\theta(x) = \int p_\theta(x, z)\,dz
$$
is impossible in closed form.

We still need the posterior for:

- Inference (explaining the observation)
- Learning (MLE gradient depends on it)
- EM algorithm E-step


#### Approximate Inference

There are two major classes of approaches to approximate inference:


####  Markov Chain Monte Carlo (MCMC)  
Generate samples from the exact posterior using a Markov chain.

- Very general; exact in the limit of infinite time / computation  
- Computationally expensive  
- Convergence is hard to diagnose  


#### 2. Variational Inference (VI)  
Approximate the posterior with a tractable distribution  
(e.g., fully factorized, mixture, or autoregressive).

- Fairly efficient — inference reduces to optimization of distribution parameters  
- Fast at test time (single forward pass of the inference network)  
- Cannot easily trade computation for accuracy (unlike MCMC)  

MCMC = flexible, asymptotically exact, but slow.  
VI = fast and scalable, but biased due to restricted approximating family.

 
## 4.2 Core Idea of Variational Inference
Turns inference into a optimization problem. Faster compared to MCMC as optimization is faster than sampleing.
Approximate the posterior with a simpler distribution:

$$
q_\phi(z \mid x) \approx p_\theta(z \mid x)
$$

Where:

- $q_\phi$ is the variational posterior
- $\phi$ are variational parameters (learned)

Requirements:

1. We can sample from $q_\phi(z \mid x)$  
2. We can compute $\log q_\phi(z \mid x)$ and its gradient wrt $\phi$  

Common choice: mean-field approximation

$$
q_\phi(z \mid x) = \prod_i q_\phi(z_i \mid x)
$$

 
## 4.3 Training with Variational Inference
Goal: maximize the marginal likelihood

$$
\log p_\theta(x)
$$

Since it's intractable, VI uses a lower bound on this quantity.

### Variational Lower Bound (ELBO)

Using Jensen’s inequality:

$$
\log p_\theta(x)
\ge 
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x, z)]
-
\mathbb{E}_{q_\phi(z \mid x)}[\log q_\phi(z \mid x)]
$$

This is the Evidence Lower Bound (ELBO):

$$
\text{ELBO}(\theta, \phi)
=
\mathbb{E}_{q_\phi}\!\left[\log p_\theta(x, z)\right]
-
\mathbb{E}_{q_\phi}\!\left[\log q_\phi(z \mid x)\right]
$$

We maximize ELBO w.r.t both $\theta$ and $\phi$.

 
## 4.4 KL Interpretation (Variational Gap)

Rewrite ELBO:

$$
\log p_\theta(x)
=
\text{ELBO}(\theta, \phi)
+
D_{\text{KL}}(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x))
$$

Thus:

- Maximizing ELBO wrt $\phi$  
  → minimizes the KL divergence between $q_\phi$ and the true posterior.

- The variational gap is  
  $$
  D_{\text{KL}}(q_\phi(z\mid x) || p_\theta(z\mid x))
  $$

If $q_\phi$ is expressive enough:
$$
q_\phi(z\mid x) = p_\theta(z\mid x)
\quad \Rightarrow \quad
\text{gap} = 0
$$

 
## 4.5 What Happens When Updating Each Parameter Set?

### Updating variational parameters $\phi$:

- Minimizes the variational gap  
- Makes $q_\phi(z \mid x)$ closer to the true posterior  
- Does not affect the model directly

### Updating model parameters $\theta$:

- Increases $\log p_\theta(x)$ (good)
- BUT often also reduces the gap by making the posterior simpler  
  → Risk: posterior collapse / variational pruning

This motivates using expressive variational families (flows, mixtures, autoregressive).


## 4.6 Variational Pruning (Posterior Collapse)
Because VI pushes $p_\theta(z \mid x)$ towards $q_\phi(z\mid x)$, the model may choose to ignore some latent dimensions:

$$
p_\theta(z_i \mid x) = p(z_i)
$$

Meaning the latent variable carries no information about $x$.

Pros:

- Automatically learns effective latent dimensionality

Cons:

- Prevents fully utilizing the latent capacity  
- Common issue in VAEs (particularly with strong decoders)


## 4.7 Choosing the Variational Posterior Family

### Simple: Mean-field Gaussian

- Fast
- Easy to optimize
- But limited expressivity

### More expressive options:

- Mixture posteriors
- Gaussians with full covariance
- Autoregressive posteriors
- Normalizing-flow posteriors

Trade-off: accuracy vs speed.

 
## 4.8 Amortized Variational Inference

Classic VI:

- Each datapoint $x$ has its own variational parameters  
- Requires iterative optimization per datapoint  
- Too slow for deep learning

Amortized VI:

- Use an inference network (encoder)
  $$
  \phi(x) \mapsto \text{parameters of } q_\phi(z\mid x)
  $$
- Fast inference  
- Works with SGD  
- Introduced in Helmholtz Machines  
- Popularized by Variational Autoencoders

 
## 4.9 Variational vs Exact Inference

### Advantages of VI

- Scalable to modern deep models  
- Fast inference  
- Enables flexible model design  

### Disadvantages

- Approximation bias  
- Posterior may be oversimplified  
- Can limit expressiveness of the full model  

 
## 4.10 Summary of Section 4

- Variational inference approximates the true posterior with a tractable distribution.  
- ELBO gives a trainable lower bound on the marginal likelihood.  
- VI converts inference into optimization.  
- Amortized VI enables neural inference (encoders).  
- Variational pruning can arise naturally and must be managed.  

## 5. Gradient Estimation in Variational Inference

### 5.1 Why Do We Need Gradient Estimators?

To train a latent variable model with variational inference, we maximize the ELBO:

$$
\text{ELBO}(\theta, \phi)
=
\mathbb{E}_{q_\phi(z\mid x)}\Big[ \log p_\theta(x, z) - \log q_\phi(z\mid x) \Big]
$$

We need gradients with respect to:

1. Model parameters $\theta$
2. Variational parameters $\phi$

The expectation makes these gradients intractable in closed form, so we estimate them using Monte Carlo samples.

---

## 5.2 Gradients w.r.t. Model Parameters ($\theta$)

This part is easy.

Because $q_\phi(z\mid x)$ does not depend on $\theta$:

$$
\nabla_\theta \text{ELBO}
=
\mathbb{E}_{q_\phi(z\mid x)} \big[ \nabla_\theta \log p_\theta(x, z) \big]
$$

We estimate this using samples:

1. Draw $z \sim q_\phi(z\mid x)$  
2. Compute $\nabla_\theta \log p_\theta(x,z)$  
3. Average across samples

No special techniques required.

---

## 5.3 Gradients w.r.t. Variational Parameters ($\phi$)

This is more difficult.

We want:

$$
\nabla_\phi \mathbb{E}_{q_\phi(z\mid x)}[f(z)]
$$

But $q_\phi(z\mid x)$ depends on $\phi$.  
Two main strategies exist to handle this dependence:

---

# 5.4 Two Families of Gradient Estimators

## 🔷 1. Likelihood-Ratio / REINFORCE Estimator

Uses the identity:

$$
\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)]
=
\mathbb{E}_{q_\phi(z)}[f(z)\,\nabla_\phi \log q_\phi(z)]
$$

This allows gradients for:
- Discrete latent variables  
- Non-differentiable $f(z)$  
- Any distribution where we can compute $\log q_\phi(z)$

Pros
- Very general  
- Works for discrete and continuous latents  

Cons
- High variance  
- Requires variance reduction (baselines, control variates)

This is the same gradient estimator used in policy gradients in RL.

---

## 🔷 2. Reparameterization / Pathwise Estimator

Instead of sampling $z \sim q_\phi(z\mid x)$ directly,
write it as a differentiable transformation of noise:

$$
z = g_\phi(\epsilon, x), \quad \epsilon \sim p(\epsilon)
$$

Then:

$$
\nabla_\phi \mathbb{E}_{q_\phi(z\mid x)}[f(z)]
=
\mathbb{E}_{\epsilon \sim p(\epsilon)}
\big[ \nabla_\phi f(g_\phi(\epsilon, x)) \big]
$$

This pushes the dependence on $\phi$ inside a differentiable function.

### Example: Gaussian posterior
If
$$
q_\phi(z\mid x) = \mathcal{N}(z\mid \mu_\phi(x), \sigma_\phi(x)^2),
$$
then:

$$
z = \mu_\phi(x) + \sigma_\phi(x)\,\epsilon, \quad \epsilon\sim \mathcal{N}(0,1)
$$

Pros
- Low variance  
- Enables stable VAE training  

Cons
- Only works for continuous latent variables  
- Requires differentiable sampling procedure

---

## 5.5 Comparison Table

| Property | REINFORCE | Reparameterization |
|----------|------------|--------------------|
| Works for discrete latent variables | ✅ | ❌ |
| Works for continuous latent variables | ✅ | ✅ |
| Low-variance gradients | ❌ | ✅ |
| Requires differentiable sampling | ❌ | ✅ |
| Used in VAEs | sometimes | always |

---

## 5.6 Practical Notes

- Modern VAEs always use the reparameterization trick.  
- More expressive posteriors (flows, mixtures) require more advanced reparameterization methods (e.g., implicit gradients).  
- Discrete VAEs use:
  - Gumbel-Softmax  
  - NVIL / REINFORCE with baselines  
  - VIMCO  

---

## 5.7 Summary of Section 5

- Gradient estimation is essential for training VI models.  
- $\nabla_\theta$ is easy: just sample from the variational posterior.  
- $\nabla_\phi$ is hard because sampling depends on parameters.  
- Two estimators solve this:
  1. Likelihood-ratio (REINFORCE)  
  2. Reparameterization trick  
- Reparameterization yields low-variance gradients and powers modern VAEs.

## 6. Variational Autoencoders (VAEs)

### 6.1 What Is a VAE?

A VAE is a latent variable generative model with:

- Continuous latent variables $z$
- Neural networks for both:
  - Encoder (variational posterior) $q_\phi(z \mid x)$  
  - Decoder (likelihood) $p_\theta(x \mid z)$
- Training through amortized variational inference  
- Gradients computed using the reparameterization trick

VAEs were introduced in 2014 by Kingma & Welling and Rezende et al., and marked a major breakthrough in tractable, scalable generative modeling.

---

## 6.2 VAE Model Components

### Prior
Usually a factorized standard Gaussian:
$$
p(z) = \mathcal{N}(0, I)
$$

### Likelihood / Decoder
Maps latents to a distribution over observations.

For binary data:
$$
p_\theta(x \mid z) = \text{Bernoulli}(x; f_\theta(z))
$$

For real-valued data:
$$
p_\theta(x \mid z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)
$$

### Variational Posterior / Encoder
$$
q_\phi(z \mid x) = \mathcal{N}(z \mid \mu_\phi(x), \sigma_\phi^2(x))
$$

All of these functions (encoder & decoder) can be implemented with:
- MLPs  
- ConvNets  
- ResNets  
- Transformers  
depending on the domain.

---

## 6.3 Training Objective: The ELBO

VAEs maximize the Evidence Lower Bound (ELBO):

$$
\mathcal{L}(x)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{\text{KL}}\!\Big(q_\phi(z\mid x)\,\|\, p(z)\Big)
$$

Interpretation:

1. Reconstruction Term  
   Measures how well the model predicts $x$ from $z$.  
   Encourages informative latents.

2. KL Regularization Term  
   Encourages $q_\phi(z\mid x)$ to stay close to the prior $p(z)$.  
   Prevents overfitting and encourages smooth latent spaces.

The KL term often has closed-form for Gaussian distributions.

---

## 6.4 Reparameterization Trick (Key to VAEs)

Direct backprop through a sample $z \sim q_\phi(z\mid x)$ is impossible.

Solution: rewrite sampling as a differentiable transformation of noise:

$$
z = \mu_\phi(x) + \sigma_\phi(x)\,\epsilon,
\quad
\epsilon \sim \mathcal{N}(0, I)
$$

This allows gradient flow through $z$ and makes VAE training practical.

---

## 6.5 VAE as a Framework

The term “VAE” now refers to a broad family of models:
- Continuous latent variables
- Amortized inference
- Reparameterization-based gradients
- Trained by maximizing ELBO (or its variants)

Modern VAEs extend the basic version in many ways:
- Multiple latent layers  
- More expressive posteriors (flows, mixtures)  
- More expressive priors (hierarchical, autoregressive)  
- More expressive decoders (ResNets, autoregressive PixelCNN decoders)  
- Iterative inference networks  
- Variance reduction techniques

The VAE framework is flexible and underlies many state-of-the-art generative models.

---

## 6.6 Summary of Section 6

- VAEs are tractable generative models with continuous latent variables.
- They pair:
  - a decoder $p_\theta(x\mid z)$ and  
  - an encoder $q_\phi(z\mid x)$
  using amortized VI.
- Training uses ELBO + reparameterization trick.
- VAEs balance reconstruction quality with regularized latent structure.
- The VAE framework is highly extensible and central to modern deep generative modeling.
