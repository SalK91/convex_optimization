Bayesian inference provides a principled framework for reasoning about uncertainty in machine learning models. It describes how to update beliefs about hidden variables when new data is observed. Many modern generative models, including VAEs and diffusion models, are based on Bayesian ideas, and variational inference is a direct approximation to Bayesian posterior inference.

This chapter introduces the core concepts of Bayesian inference, why posterior inference is difficult, and how these ideas set the stage for variational inference and the ELBO in the next chapter.


## 1. Bayes’ Rule

Bayes’ theorem relates prior beliefs, likelihoods, and posterior beliefs. For a hidden variable $z$ and an observed variable $x$:

$$
p(z|x) = \frac{p(x|z)\,p(z)}{p(x)}.
$$

Each term has a clear interpretation.

- $p(z)$: prior belief about the unknown variable  
- $p(x|z)$: likelihood of observing $x$ given $z$  
- $p(x)$: marginal likelihood or evidence  
- $p(z|x)$: posterior distribution after observing data  

Bayesian inference is the task of computing $p(z|x)$.

 
## 2. Priors: Encoding Assumptions About Hidden Variables

The prior $p(z)$ expresses what we believe about the latent variable before observing the data. Priors serve several purposes in machine learning.

### 2.1 Regularization  
A prior can prevent overfitting. For example, a Gaussian prior on weights yields $L_2$ regularization.

### 2.2 Structural assumptions  
Priors can encode assumptions such as smoothness, sparsity, or low-dimensional structure.

### 2.3 Uncertainty  
The prior makes explicit that before observing data, we do not know the true value of $z$.

### 2.4 Generative modeling  
In latent-variable models like VAEs, the prior determines the structure of the latent space.

 
## 3. Likelihood: Connecting Latent Variables to Observed Data

The likelihood $p(x|z)$ describes how the data are generated from latent causes. In many generative models:

- $z$ represents latent structure  
- $x$ represents an image, time series, or text  
- $p(x|z)$ is parameterized by a neural network decoder  

The likelihood term encourages the latent variable $z$ to explain the observed data.

 
## 4. The Posterior: What We Really Want to Compute

The goal of Bayesian inference is the posterior:

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}.
$$

The posterior expresses how our belief about $z$ changes after seeing $x$. It incorporates both:

- prior knowledge  
- evidence from data  

Unfortunately, computing this posterior is usually intractable.

 
## 5. Why Exact Inference Is Hard

The denominator in Bayes’ rule is the marginal likelihood:

$$
p(x) = \int p(x,z)\,dz.
$$

This integral is often impossible to evaluate directly because:

- the latent space $z$ can be high-dimensional  
- the joint distribution $p(x,z)$ may involve a complex neural network  
- the integral has no analytic form  

Computing the exact posterior is rarely feasible in modern models. This makes approximate inference essential.

 
## 6. Maximum a Posteriori (MAP) vs Full Bayesian Inference

There are two kinds of Bayesian computation.

### 6.1 MAP estimation  
MAP finds the *single most likely* value of $z$:

$$
z_{\text{MAP}}
= \arg\max_z\, p(z|x).
$$

MAP is similar to maximum likelihood but includes the prior. MAP is easier to compute but does not provide uncertainty.

### 6.2 Full posterior inference  
The full posterior $p(z|x)$ describes a *distribution* over possible values of $z$, reflecting uncertainty. Most Bayesian methods aim for the full posterior, not MAP. However, because it is intractable, we approximate it.

 
## 7. Bayesian Latent-Variable Models

Many generative models are Bayesian latent-variable models with:

1. a prior over latent variables  
   $$
   z \sim p(z)
   $$

2. a conditional likelihood  
   $$
   x \sim p(x|z)
   $$

3. a posterior  
   $$
   p(z|x)
   $$

Examples include:

- VAEs  
- mixture models  
- topic models  
- probabilistic PCA  
- diffusion models (in a specific sense)  

Bayesian inference is the foundation of these models.

 
## 8. The Evidence and Its Importance

The marginal likelihood, also called the evidence:

$$
p(x) = \int p(x,z)\,dz
$$

plays several roles:

- It normalizes the posterior.  
- It evaluates how well a model explains data.  
- It is used in Bayesian model comparison.  
- Its logarithm appears in training objectives for VAEs and diffusion models.

Maximizing evidence corresponds to learning a model that explains the data well.

 
## 9. Bayesian Interpretation of KL Divergence

KL divergence naturally appears when comparing an approximate posterior $q(z|x)$ with the true posterior $p(z|x)$:

$$
D_{\text{KL}}(q(z|x)\|p(z|x)).
$$

Minimizing this KL divergence means making the approximation $q$ as close as possible to the exact posterior.

This forms the basis of variational inference.

---

## 10. Why We Need Variational Inference

Because the true posterior is intractable, we introduce a simpler distribution $q(z|x)$ and optimize it to approximate $p(z|x)$.

We cannot compute:

$$
D_{\text{KL}}(q(z|x)\|p(z|x))
$$

directly, because $p(z|x)$ depends on $p(x)$, which is the intractable integral.

Variational inference resolves this by rewriting $\log p(x)$ and isolating the KL divergence from quantities we can compute. This leads to the Evidence Lower Bound (ELBO), which forms the training objective of VAEs.

This is the topic of the next chapter.

---
Bayesian inference describes how to update beliefs in light of new evidence using Bayes’ rule. The posterior distribution combines the prior and likelihood to capture all information about latent variables. However, direct computation of the posterior is often intractable due to the marginal likelihood integral.

Approximate inference methods are therefore necessary. Variational inference replaces the true posterior with a tractable approximation and optimizes it by minimizing KL divergence. Understanding Bayesian inference is essential for understanding the ELBO, VAEs, Bayesian neural networks, and modern probabilistic deep learning methods.

