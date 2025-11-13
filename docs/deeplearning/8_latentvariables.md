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

Examples:
- RNN/Transformer LMs  
- NADE  
- PixelCNN / WaveNet

Pros
- Easy training (max. likelihood).
- No sampling during training.

Cons
- Slow generation (sequential).
- Often capture local structure better than global structure.

---

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

---

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

---

## 1.4 Progress Over Time
- 2014: Deep autoregressive networks (early DL-based generative models).  
- 2015: DRAW — recurrent model for images.  
- 2019: Hierarchical autoregressive models achieve near-photographic fidelity.

Generative modelling has rapidly improved due to:
- Better architectures  
- Better optimization  
- Larger compute and datasets  

---

## 1.5 Summary
- Generative modelling aims to capture the full data distribution, enabling sampling, representation learning, and cross-domain mappings.
- Three major families: autoregressive, latent-variable, and implicit models.
- Each makes different trade-offs between tractability, expressiveness, and sample quality.
