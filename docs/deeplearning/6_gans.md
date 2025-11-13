## 1. Overview: Generative Models

- Goal: learn a model of the true data distribution $p^*(x)$ from samples.

### Types of Generative Models
1. Explicit likelihood models – define tractable $p_\theta(x)$
   - Max. likelihood: PPCA, Mixture Models, PixelCNN, Wavenet, autoregressive LMs.
   - Approx. likelihood: Boltzmann Machines, Variational Autoencoders (VAE).

2. Implicit models – define *sampling procedure*, not explicit $p_\theta(x)$  
   - Examples: GANs, Moment Matching Networks.


## 1.1 The GAN Idea

- Two-player minimax game:
  - Generator (G): maps noise $z \sim p(z)$ to data space $G(z)$.
  - Discriminator (D): classifies samples as *real* (from $p^*(x)$) or *fake* ($G(z)$).

- Objectives:
  $$
  \min_G \max_D \; \mathbb{E}_{x\sim p^*(x)}[\log D(x)] +
  \mathbb{E}_{z\sim p(z)}[\log(1 - D(G(z)))]
  $$

- Interpretation:
  - $D$ learns to distinguish real from fake.
  - $G$ learns to fool $D$.
  - Training reaches equilibrium when $p_G(x) = p^*(x)$.

 
## 1.2 Alternative View — Teacher–Student Analogy

- Teacher (D): distinguishes real vs fake, providing feedback.
- Student (G): improves by making fake data look real.
- Cooperative interpretation of the adversarial process.

 
## 1.3 GANs as a Game

- Zero-sum, bi-level optimization → strong connection to game theory.
- GAN equilibrium = Nash equilibrium between $G$ and $D$.
- Training alternates between optimizing $D$ and $G$.

 

Key Intuition:  
GANs learn by *competition* between a generator and discriminator rather than direct likelihood maximization.


## 2. GAN Objective as Divergence Minimization

- Generative modeling often aims to minimize a distance or divergence between
  the true data distribution $p^*(x)$ and model distribution $p_G(x)$.

### 2.1 KL and Related Divergences
- Maximum Likelihood Estimation (MLE):
  $$
  \min_\theta D_{\text{KL}}(p^*(x) \| p_\theta(x))
  $$
  → drives $p_\theta$ to assign high probability to observed data.

- But: implicit models (like GANs) don’t have explicit likelihoods, so MLE can’t be used directly.

 
## 2.2 GAN as Jensen–Shannon (JS) Divergence Minimization

- If discriminator $D$ is optimal:
  $$
  D^*(x) = \frac{p^*(x)}{p^*(x) + p_G(x)}
  $$
  Plugging into the GAN loss shows that the generator minimizes:
  $$
  D_{\text{JS}}(p^*(x) \| p_G(x))
  $$
  → GAN ≈ JS divergence minimization.

- However, this relies on an *optimal discriminator* — not true in practice.

 

## 2.3 Limitations of KL / JS Divergences
- If $p_G$ and $p^*$ have non-overlapping support,  
  → no useful gradient signal (zero gradient problem).
- The density ratio $\frac{p^*(x)}{p_G(x)}$ becomes infinite where $p_G=0$.
- Thus, GANs can fail to learn when supports are disjoint.

 

## 2.4 Alternative Distances & Divergences

### (a) Wasserstein Distance (Earth Mover’s)
- Measures minimal “cost” of moving probability mass:
  $$
  W(p^*, p_G) = \inf_{\gamma \in \Pi(p^*, p_G)} \mathbb{E}_{(x,y)\sim \gamma}[\|x - y\|]
  $$
- Provides smooth, non-vanishing gradients even when supports don’t overlap.
- WGAN: enforce 1-Lipschitz $D$ via:
  - weight clipping,
  - gradient penalty (WGAN-GP),
  - spectral normalization.

### (b) MMD (Maximum Mean Discrepancy)
- Compares distributions via embeddings in a Reproducing Kernel Hilbert Space (RKHS):
  $$
  \text{MMD}^2(p, q) = \|\mathbb{E}_p[\phi(x)] - \mathbb{E}_q[\phi(x)]\|^2
  $$
- MMD-GAN: learns kernel features $\phi$ jointly with $D$.

### (c) f-divergences
- General framework using convex functions $f$:
  $$
  D_f(p \| q) = \mathbb{E}_q[f\!\left(\frac{p(x)}{q(x)}\right)]
  $$
- GAN training derived via variational lower bound on $D_f$.

 

## 2.5 Practical View

- GANs are not pure divergence minimizers in practice:
  - $D$ not optimal → approximate divergence.
  - Neural discriminator learns a smooth approximation to density ratio.
  - Provides *useful gradients* even when the true divergence would fail.

 

## 2.6 Summary Table

| Perspective | Example | Key Idea |
|--------------|----------|-----------|
| KL Divergence | MLE, VAEs | Explicit likelihoods |
| JS Divergence | Original GAN | Adversarial training |
| Wasserstein | WGAN | Smooth gradients |
| MMD | MMD-GAN | Kernel mean embedding |
| f-divergence | f-GAN | Variational bound family |

Insight:  
GANs can be viewed as learning a *neural divergence measure* that provides a stable, informative training signal.

## 3. Evaluating GANs

- Evaluating generative models is difficult — no single metric captures all aspects.
- Must assess:
  1. Sample quality (fidelity, realism)
  2. Diversity / generalization
  3. Representation learning (usefulness of learned features)

 
 
## 3.1 Why Not Log-Likelihood?

- GANs are implicit models — no tractable $p(x)$.
- Estimating log-likelihood is expensive and unreliable.
- Hence: use feature-based or classifier-based proxies.

 
## 3.2 Inception Score (IS)

- Uses a pretrained Inception v3 classifier.
- Compares predicted label distributions of generated samples.

Formula:
$$
\text{IS} = \exp\!\left( \mathbb{E}_{x \sim G} [ D_{KL}(p(y|x) \| p(y)) ] \right)
$$

Intuition:
- High-quality images → confident predictions ($p(y|x)$ low entropy).  
- Diverse images → marginal label distribution $p(y)$ high entropy.

Properties:
- Measures *sample quality* and *diversity*.
- Correlates with human judgment.
- Fails to capture intra-class variation or features beyond ImageNet classes.

Higher is better.

 

## 3.3 Fréchet Inception Distance (FID)

- Compares statistics of features (from pretrained Inception network) for real vs fake samples.

Formula:
$$
\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are mean and covariance of real and generated data features.

Properties:
- Sensitive to mode dropping and artifacts.
- Correlates strongly with human evaluation.
- Lower is better.
- Biased for small sample sizes → use KID (Kernel Inception Distance) for correction.

 
## 3.4 Overfitting Check — Nearest Neighbours

- Compute nearest real images to generated samples in pretrained feature space.
- Helps detect memorization (copying training images).

 
## 3.5 Evaluation Depends on Goal

| Goal | Metric Example | Measures |
|------|----------------|-----------|
| Image quality | FID, IS | Fidelity & diversity |
| Representation learning | Linear probe accuracy | Feature usefulness |
| Data generation | Human evaluation | Perceptual quality |
| RL / control | Policy reward | Functional realism |

 
Key Takeaway:  
Use *multiple complementary metrics* — quantitative (IS, FID) + qualitative (visual inspection, diversity).

## 4. The GAN Zoo

> GANs have evolved rapidly — from simple MLPs on MNIST to massive multi-GPU models like BigGAN and StyleGAN.

 

## 4.1 The Original GAN

- First formulation of adversarial training.
- Architecture: simple multilayer perceptrons (MLPs).
- Trained on small images (e.g. 32×32).  
- Ignored spatial structure (flattened pixels).  
- Introduced the minimax objective still used today.

 
## 4.2 Conditional GAN  

- Adds *conditioning information* $y$ (e.g. class label or input image).  
  $$
  \min_G \max_D \mathbb{E}_{x,y}[\log D(x,y)] + \mathbb{E}_{z,y}[\log(1 - D(G(z,y),y))]
  $$

- Enables controlled generation — specify category or domain.  
  Examples:
  - Class-conditional image synthesis (e.g., "generate a dog").  
  - Image-to-image translation (later: Pix2Pix, CycleGAN).

 

## 4.3 Laplacian GAN  

- Generates images progressively, starting from low resolution.  
- Each level adds high-frequency detail via residual (Laplacian) generation.  
- Fully convolutional — can produce arbitrarily large outputs.
- Improves high-res synthesis through multi-scale structure.

 
## 4.4 Deep Convolutional GAN  

- Replaces MLPs with deep convnets for both $G$ and $D$.
- Uses Batch Normalization and ReLU/LeakyReLU for stability.
- Enables smooth interpolation in latent space:
  - $G(z_1)$ → $G(\frac{1}{2}(z_1 + z_2))$ → $G(z_2)$ produces semantically meaningful transitions.
- Latent space exhibits semantic arithmetic (e.g. “man + glasses – woman”).

 
## 4.5 Spectrally Normalized GAN  

- Enforces 1-Lipschitz constraint on $D$ via spectral normalization:
  $$
  \bar{W} = \frac{W}{\sigma_{\max}(W)}
  $$
  where $\sigma_{\max}(W)$ is the largest singular value.

- Stabilizes training and improves generalization.

 
## 4.6 Projection Discriminator  

- Adds class embedding projection inside $D$:  
  $$
  D(x, y) = h(x)^\top v_y + b_y
  $$
  where $v_y$ is the embedding for class $y$.

- Theoretically consistent probabilistic discriminator formulation.  
- Strong empirical results on class-conditional image synthesis.

 
## 4.7 Self-Attention GAN  

- Introduces self-attention layers to capture long-range dependencies.  
- Improves global structure and coherence in generated images.
- Inspired by Transformer attention.



## 4.8 BigGAN 

- Scaled-up GANs with massive compute + large datasets (ImageNet, JFT).  
- Key ingredients:
  - Hinge loss for $D$  
  - Spectral normalization  
  - Self-attention  
  - Projection discriminator  
  - Orthogonal regularization  
  - Skip connections from noise  
  - Shared class embeddings  
- Truncation trick: reduce noise magnitude to increase fidelity (trade-off with diversity).



## 4.9 LOGAN  

- Introduces latent optimization — optimize $z$ via gradient updates to improve adversarial dynamics.  
- Uses natural gradient descent in latent space.  
- Yields higher FID/IS improvements over BigGAN.

 

## 4.10 Progressive GAN  

- Trains from low to high resolution (4×4 → 8×8 → 16×16 …).  
- Each stage adds new layers to $G$ and $D$.  
- Dramatically improves stability and image quality (especially faces).

 

## 4.11 StyleGAN  

- Adds style-based generator architecture:
  - Latent vector $z$ transformed by MLP to intermediate $w$.
  - AdaIN (Adaptive Instance Normalization): modulates style per channel.
  - Injects per-pixel noise for local details.

- Learns disentangled representations — global attributes (style) vs local (texture).

 
## 4.12 Takeaways

- GAN progress driven by:
  - Better architectures (Conv, Attention, Progressive, Style-based)
  - Normalization & regularization
  - Stability techniques
  - Large-scale training

Trend:  
From small MLPs → Conv architectures → Attention-based, scalable, stable models like BigGAN & StyleGAN.

## 5. Representation Learning with GANs

> Beyond generating samples, GANs can learn rich latent representations of data.

 
## 5.1 Motivation
- GANs implicitly learn latent spaces that capture high-level semantics.
- Exploring or constraining this latent space enables unsupervised representation learning.

---

## 5.2 Evidence from DCGAN 

- DCGAN latent vectors encode meaningful directions:
  - Smooth interpolation between points → semantic transformations.
  - Linear arithmetic in latent space (e.g., *smiling woman – woman + man → smiling man*).
- Suggests disentangled feature representations emerge naturally.

---

## 5.3 InfoGAN  

- Extends GAN with information maximization objective:
  - Encourages some latent codes $c$ to be *interpretable* and *disentangled*.

Objective:
$$
\min_G \max_D V(D,G) - \lambda I(c; G(z, c))
$$
where $I(c; G(z, c))$ is mutual information between latent code and generated output.

- Adds an auxiliary network to infer $c$ from $G(z, c)$.
- Learns to associate:
  - Discrete codes → categories (digits, shapes)
  - Continuous codes → attributes (rotation, scale)

---

## 5.4 ALI / BiGAN 

- Adds an encoder $E(x)$ mapping real data to latent space.
- Joint discriminator distinguishes pairs:
  $$
  (x, E(x)) \quad \text{vs.} \quad (G(z), z)
  $$

- At equilibrium:
  - $E$ and $G$ become approximate inverses:
    - $x \approx G(E(x))$
    - $z \approx E(G(z))$

- Enables inference and representation learning simultaneously.

---

## 5.5 BigBiGAN 
- Scales BiGAN to BigGAN architecture.
- Uses large-scale encoders ($E$) with ResNet blocks.
- Learns strong unsupervised representations competitive with self-supervised models.

Observations:
- Reconstructions $G(E(x))$ preserve semantic content, not exact pixels.
- Encoder features yield high ImageNet classification accuracy after linear probing.

---

## 5.6 Summary

| Model | Key Idea | Outcome |
|--------|-----------|----------|
| DCGAN | Implicitly semantic latent space | Interpolations meaningful |
| InfoGAN | Maximize info between codes and outputs | Disentangled features |
| BiGAN / ALI | Add encoder, joint training | Bidirectional mapping |
| BigBiGAN | Large-scale BiGAN | Competitive unsupervised features |

Key Insight:  
GANs not only *generate*, but also *encode* — their latent structure can act as a rich, learned representation space.

## 6. GANs for Other Modalities and Problems

> GANs extend far beyond images — used for translation, audio, video, RL, and even art.

---

## 6.1 Image-to-Image Translation

### (a) Pix2Pix 
- Conditional GAN trained on *paired* datasets $(x, y)$.
- Learns deterministic mapping between domains (e.g., edges → photos).
- Loss combines adversarial term + L1 reconstruction:
  $$
  \mathcal{L}_{\text{Pix2Pix}} = \mathcal{L}_{\text{GAN}}(G,D) + \lambda \|y - G(x)\|_1
  $$

### (b) CycleGAN  
- Unpaired domain translation — no 1:1 correspondence.
- Uses cycle consistency:
  - $x \in A \to G_B(x) \to F_A(G_B(x)) \approx x$
  - Enforces invertibility between domains.
- Enables tasks like *horse ↔ zebra*, *summer ↔ winter*.

 

## 6.2 Audio Synthesis

### (a) WaveGAN  
- Adapts convolutional GANs to 1D waveforms.
- Fully unsupervised raw-audio synthesis.

### (b) MelGAN  
- Conditional GAN trained to generate mel-spectrogram waveforms.
- Used in text-to-speech (GAN-TTS).

### (c) GAN-TTS  
- High-fidelity speech synthesis model.
- Achieves human-like audio quality via adversarial losses.

 
## 6.3 Video Synthesis & Prediction

- GANs extended to spatiotemporal data:
  - TGAN-v2 (Saito & Saito, 2018): multi-layer subsampling for video generation.
  - DVD-GAN (Clark et al., 2019): scalable adversarial model for long, complex videos.
  - TriVD-GAN (Luc et al., 2020): transformation-based video prediction.

 
## 6.4 GANs in Reinforcement Learning (Imitation & Control)

- GAIL (Ho & Ermon, 2016): *Generative Adversarial Imitation Learning*  
  - Discriminator distinguishes expert vs policy trajectories.
  - Generator = policy network optimizing to mimic experts.

 

## 6.5 Creative & Applied Uses

- GauGAN (Park et al., 2019): semantic image synthesis using spatially-adaptive normalization (SPADE).  
- SPIRAL (Ganin et al., 2018): program synthesis from images via adversarial reinforcement learning.  
- Everybody Dance Now (Chan et al., 2019): motion transfer via adversarial video mapping.  
- DANN (Ganin et al., 2016): domain-adversarial training for domain adaptation.  
- Learning to See (Memo Akten, 2017): interactive GAN-based digital art.

 
## 6.6 Summary

| Domain | Example | Key Idea |
|---------|----------|----------|
| Paired image translation | Pix2Pix | Conditional GAN + L1 loss |
| Unpaired translation | CycleGAN | Cycle consistency |
| Audio | MelGAN, WaveGAN | Conditional waveform generation |
| Video | DVD-GAN, TGAN-v2 | Temporal adversarial modeling |
| RL / Imitation | GAIL | Adversarial trajectory matching |
| Art / Creativity | GauGAN, SPIRAL | Adversarial synthesis and style transfer |

Insight:  
Adversarial learning generalizes across domains — GANs serve as a *universal generator–critic framework* for structured data.
