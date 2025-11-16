Variational inference (VI) provides a general framework for approximating difficult probability distributions with simpler, tractable ones. Many modern machine-learning models rely on VI, including variational autoencoders (VAEs), Bayesian neural networks, latent-variable models, and diffusion models. VI offers a scalable alternative to sampling-based inference and converts the inference problem into an optimization problem.

## 1. The Problem of Inference

Many probabilistic models introduce hidden variables to explain observations. Examples include:

- latent variables $z$ in VAEs  
- weight distributions in Bayesian neural networks  
- cluster indicators in mixture models  
- hidden states in topic models and HMMs  

The goal is to compute the posterior distribution

$$
p(z|x) = \frac{p(x,z)}{p(x)}.
$$

The difficulty lies in the marginal likelihood

$$
p(x) = \int p(x,z)\,dz,
$$

which is often intractable in high-dimensional or complex models. Exact Bayesian inference becomes impossible, which motivates approximate methods. Variational inference addresses this challenge.

 
## 2. The Idea of Variational Inference

Variational inference replaces the intractable posterior with a tractable approximation. Instead of trying to compute $p(z|x)$ exactly, VI introduces a family of simpler distributions

$$
q_\phi(z|x) \in \mathcal{Q},
$$

and chooses the member that is closest to the true posterior. Closeness is measured using the KL divergence:

$$
D_{\text{KL}}(q_\phi(z|x)\|p(z|x)).
$$

The goal is:

$$
\phi^* = \arg\min_\phi
D_{\text{KL}}(q_\phi(z|x)\|p(z|x)).
$$

However, the KL depends on $p(z|x)$, which is unknown, making direct minimization impossible. The key insight is that the KL can be rewritten in terms of computable quantities, leading to the Evidence Lower Bound (ELBO).

 
## 3. Deriving the ELBO

We start from the marginal likelihood:

$$
\log p(x) = \log \int p(x,z)\,dz.
$$

We multiply and divide by $q_\phi(z|x)$:

$$
\log p(x)
=
\log \int q_\phi(z|x)\,
\frac{p(x,z)}{q_\phi(z|x)}\,dz.
$$

Applying Jensen’s inequality yields:

$$
\log p(x)
\ge 
\mathbb{E}_{q_\phi(z|x)}
\left[
\log \frac{p(x,z)}{q_\phi(z|x)}
\right].
$$

This expression is the ELBO:

$$
\mathcal{L}(x;\phi,\theta)
=
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x,z)]
-
\mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x)].
$$

A useful identity reveals:

$$
\log p(x)
=
\mathcal{L}(x;\phi,\theta)
+
D_{\text{KL}}(q_\phi(z|x)\|p(z|x)).
$$

Since KL divergence is non-negative:

$$
\mathcal{L}(x;\phi,\theta) \le \log p(x).
$$

Maximizing the ELBO is equivalent to minimizing the KL divergence between $q_\phi(z|x)$ and the true posterior.

 
## 4. Interpreting the ELBO

The ELBO can be decomposed into two terms that have clear interpretations. Writing

$$
p(x,z) = p_\theta(x|z)p(z),
$$

and substituting into the ELBO gives:

$$
\mathcal{L}(x;\phi,\theta)
=
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
-
D_{\text{KL}}(q_\phi(z|x)\|p(z)).
$$

### Reconstruction term

$$
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)].
$$

This term ensures that $z$ captures enough information to generate or reconstruct the observed data. It corresponds to likelihood or reconstruction accuracy.

### Regularization term

$$
D_{\text{KL}}(q_\phi(z|x)\|p(z)).
$$

This term ensures that the posterior approximation does not drift too far from the prior. In VAEs, the prior is usually a Gaussian, making the latent space smooth and structured.

The ELBO therefore expresses a balance:

- the first term rewards informative latent variables  
- the second term penalizes overly complex or irregular latent distributions  

 
## 5. Why VI Uses Reverse KL

Variational inference minimizes

$$
D_{\text{KL}}(q_\phi(z|x)\|p(z|x)),
$$

which is reverse KL. Reverse KL has important behavioral properties:

- it heavily penalizes assigning probability mass where the true posterior is low  
- it allows $q$ to ignore some modes of $p(z|x)$  
- it prefers tight, conservative approximations  

As a result:

- VI tends to be mode seeking  
- it focuses on a single high-density region  
- it can miss multimodal structure of the true posterior  

This behavior explains why VAEs sometimes produce smooth or blurry samples: the latent space favors safe, central modes.

 
## 6. Variational Autoencoders (VAEs)

A VAE applies variational inference to a deep latent-variable model. It introduces:

1. a latent prior  
   $$
   z \sim p(z)
   $$
2. a decoder (generative model)  
   $$
   x \sim p_\theta(x|z)
   $$
3. an encoder (variational posterior)  
   $$
   q_\phi(z|x) \approx p(z|x)
   $$

The encoder and decoder are neural networks, trained jointly by maximizing the ELBO over all data points.

### 6.1 Generative model

Given $z$ sampled from the prior, the decoder produces a distribution over possible $x$:

$$
p_\theta(x|z).
$$

### 6.2 Inference model

The encoder produces the parameters of the approximate posterior:

$$
q_\phi(z|x) = \mathcal{N}(z\mid \mu_\phi(x), \sigma^2_\phi(x)).
$$

This is the distribution used inside the ELBO.

### 6.3 VAE training objective

The objective for each data point is:

$$
\mathcal{L}(x)
=
\mathbb{E}_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{\text{KL}}(q_\phi(z|x)\|p(z)).
$$

The first term encourages correct reconstruction; the second keeps latent codes regularized.

 
## 7. The Reparameterization Trick

The expectation in the ELBO involves sampling from $q_\phi(z|x)$. To differentiate through this sampling step, VAEs use the reparameterization:

$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
$$

This expresses sampling as a deterministic transformation of noise, allowing gradients to flow through the encoder.

This trick is central to making VI scalable and efficient in deep learning.

 
## 8. Consequences of Reverse KL in VAEs

The reverse KL term shapes the behavior of the VAE:

- it encourages smooth, overlapping latent regions  
- it prefers safe latent representations  
- it explains why VAEs sometimes produce blurry or conservative samples  
- it stabilizes training  
- it produces well-structured latent spaces  

Extensions such as the $\beta$-VAE, hierarchical VAEs, and flows inside the encoder allow for more expressive or disentangled representations.

 
## 9. Variational Inference Beyond VAEs

VI provides a general-purpose framework for approximate inference in many settings.

### Bayesian neural networks  
Posterior distributions over weights are approximated by variational distributions:

$$
q(w)\approx p(w|D).
$$

### Diffusion models  
The training objective resembles a variational bound on the data likelihood, using KL divergences between transition kernels.

### Normalizing flows for VI  
Flows can produce more expressive variational posteriors than simple Gaussians.

### Reinforcement learning  
Entropy-regularized RL and soft Q-learning can be interpreted through variational principles.

VI therefore offers a unifying viewpoint across deep generative models, Bayesian inference, and probabilistic deep learning.

---

 
Variational inference replaces an intractable posterior distribution with a tractable approximation and optimizes this approximation by maximizing the ELBO. The ELBO decomposes into a reconstruction term and a KL regularization term, capturing the trade-off between accuracy and complexity. VAEs are an important application of VI, using neural networks to parameterize both the generative model and the approximate posterior. Reverse KL explains the conservative behavior of VI-based models. Variational inference provides a flexible approach for approximate Bayesian inference and underlies many modern generative and representation-learning techniques.

