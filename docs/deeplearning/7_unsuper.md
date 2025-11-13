## 1. What is Unsupervised Learning?

### Definition
- Goal: discover structure in data without explicit labels or rewards.  
- Learns a compact, informative representation of input data.

| Learning Type | Goal | Supervision |
|----------------|------|--------------|
| Supervised | Map inputs → labels | Requires labeled data |
| Reinforcement | Learn actions maximizing future reward | Requires reward signal |
| Unsupervised | Find hidden structure | No labels or rewards |

 
### Core Ideas
- Model latent structure or relationships between observations.  
- Examples:
  - Clustering: group similar data points.  
  - Dimensionality reduction: project data to low-dimensional latent space.  
  - Manifold learning / disentangling: uncover independent factors of variation.

 
### Evaluation Challenges
How do we know if unsupervised learning worked?

- Ambiguity of structure: multiple valid clusterings possible.  
  e.g., cluster by *leg count*, *arm number*, or *height* in robot dataset.
- Metrics depend on downstream use:  
  useful representations should improve data efficiency, generalization, or transfer.

 
### Classic Methods
- PCA (Principal Component Analysis): orthogonal basis capturing variance.  
- ICA (Independent Component Analysis): separates statistically independent components.  
- Modern goal: move beyond orthogonality → learn *disentangled* factors.

 
### Summary
Unsupervised learning discovers patterns, dependencies, or latent variables from data itself — forming the foundation for *representation learning*.

## 2. Why is Unsupervised Learning Important?

### 2.1 Historical Context of Representation Learning

| Era | Key Milestone | Approach |
|------|---------------|-----------|
| 1950s–2000s | Arthur Samuel (1959): *Machine Learning* coined | Feature engineering, clustering |
| 2000s | Kernel methods (Hofmann et al., 2008) | Hand-crafted similarity functions |
| 2006 | Hinton & Salakhutdinov: *RBMs & Autoencoders* | Layer-wise unsupervised pretraining |
| 2012 | Krizhevsky et al.: *AlexNet* | End-to-end supervised learning dominates |

- Progress came from more data, deeper models, and better hardware — but not necessarily more efficient learning.

 
### 2.2 Limitations of Purely Supervised Learning
Supervised models are:
- Data inefficient — need millions of labeled samples.
- Brittle — vulnerable to adversarial perturbations.
- Poor at transfer — struggle with new domains or tasks.
- Lack common sense — limited abstraction and reasoning.
 
### 2.3 Evidence of Current Gaps

| Challenge | Example | Reference |
|------------|----------|------------|
| Data efficiency | Learning from few examples | Lake et al. (2017) |
| Robustness | Adversarial examples, brittle decisions | Goodfellow et al. (2015) |
| Generalization | CoinRun, DMLab-30 | Cobbe (2018), DeepMind |
| Transfer | Schema Networks | Kansky et al. (2017) |
| Common sense | Conceptual reasoning | Lake et al. (2015) |
 
### 2.4 Why Unsupervised Learning Matters

- Enables data-efficient adaptation to new tasks.
- Provides robust, generalizable features.
- Promotes transfer learning by separating invariant factors.
- Encourages abstract reasoning and causal understanding.

### 2.5 Towards General AI
Unsupervised learning provides shared representations enabling:
- Rapid multi-task adaptation.  
- Reuse across vision, language, and control.  
- Reduced supervision in real-world learning.
 
Summary:  
Unsupervised representation learning addresses the core limits of current AI — aiming for *data efficiency, robustness, generalization, transfer,* and *common sense*.

## 3. What Makes a Good Representation?

> A representation is an internal model of the world — an abstraction that makes reasoning and prediction efficient.

### 3.1 What is a Representation?

> “A formal system for making explicit certain entities or types of information, together with a specification of how the system does this.”

- Represents *information about the world* in a way useful for computation.  
- Not about a single feature, but the geometry or manifold shape in representational space.

 
### 3.2 Why Representation Form Matters
- Determines which computations are easy.  
- Should make relevant variations simple (e.g., object position) and irrelevant ones invariant (e.g., lighting).


 
### 3.3 Desirable Properties

| Property | Description | Intuition |
|-----------|--------------|------------|
| Untangling | Simplifies complex input manifolds | Enables linear decoding |
| Attention | Allows selective focus on relevant factors | Supports task-specific filtering |
| Clustering | Groups similar experiences together | Facilitates generalization |
| Latent Information | Encodes hidden or inferred causes | Predicts unobserved aspects |
| Compositionality | Builds complex concepts from simple parts | Enables open-ended reasoning |


### 3.4 Information Bottleneck Principle 
- Good representations compress inputs while preserving information about outputs.
  $$
  \max I(Z; Y) - \beta I(Z; X)
  $$
- Encourages minimal sufficient representations — compact yet predictive.

## 4. Evaluating the Merit of a Representation

> The value of a representation lies in how well it supports efficient, generalizable behavior across tasks.

 
### 4.1 The Evaluation Challenge
- No single metric defines a “good” representation.
- The test: How well does it help solve new, diverse, unseen tasks efficiently?

Representations should enable:
- Data efficiency — learn new tasks from few examples.  
- Robustness — resist noise or perturbations.  
- Generalization — perform well on new data.  
- Transfer — reuse knowledge in new settings.  
- Common sense — support reasoning and abstraction.

 
### 4.2 Example: Evaluating Representations via Symmetries

Let:
- $W$ = world space  
- $Z$ = representational space  
- $G = G_x \times G_y \times G_c$ = group of transformations (e.g., position, color)

A good representation $f: W \rightarrow Z$ should satisfy:
$$
f(g \cdot w) = g' \cdot f(w), \quad \forall g \in G, w \in W
$$

That is, transformations in the world (translation, color shift) correspond to predictable transformations in representation space → equivariance.

 
### 4.3 Desirable Evaluation Criteria

| Criterion | Desired Property | Example / Metric |
|------------|------------------|------------------|
| Equivariance | Transformations map consistently | Translation → shift in latent |
| Compositionality | Combine factors to form new concepts | Modular latent factors |
| Metric structure | Smooth distances reflect similarity | $L_2$, cosine |
| Attention | Selectively focus on task-relevant parts | Masking or gating mechanisms |
| Symmetries | Invariance to irrelevant transformations | Rotation, scale invariance |

 
### 4.4 Downstream Evaluation Tasks

| Evaluation Setting | Example Task | Reference |
|--------------------|---------------|------------|
| Perception / Control | Predict object color or position | Gens & Domingos, *Deep Symmetry Networks* (2014) |
| Robustness | Classify images under adversarial noise | Gowal et al., 2019 |
| Sequential Attention | Learn task-focused vision | Zoran et al., 2020 |
| Transfer / RL | Zero-shot navigation (DARLA) | Higgins et al., ICML 2017 |
| Lifelong Learning | Maintain latent structure over domains | Achille et al., NeurIPS 2018 |
| Reasoning / Imagination | Compositional concept inference | Lake et al., *Science* 2015; Higgins et al., *ICLR* 2018 |

 

### 4.5 Why Evaluation Matters
A good representation supports simple mappings to downstream tasks:
- Linear classifiers for vision tasks (e.g., color or position recognition).  
- Efficient policy learning in RL with fewer samples.  
- Abstract reasoning and imagination — *“If rainbow elephants live in big cities, can we expect one in London?”*

 
## 5. Representation Learning Techniques

> Modern unsupervised representation learning spans generative, contrastive, and self-supervised approaches — all aiming to extract structure from data without labels.


### 5.1 Categories of Methods

| Category | Core Idea | Typical Example |
|-----------|------------|-----------------|
| Generative Modeling | Learn $p(x)$ or a model that can *reconstruct* data | VAE, β-VAE, MONet, GQN, GANs |
| Contrastive Learning | Learn by *discriminating* similar vs dissimilar samples | CPC, SimCLR, word2vec |
| Self-Supervised Learning | Design *pretext tasks* that predict missing or reordered parts | BERT, Colorization, Context Prediction |


## 5.2 Generative Modeling

### 5.2.1 Motivation
- Goal: learn the underlying data distribution $p(x)$ to reveal hidden structure and causal factors.  
- Unsupervised generative modeling captures common regularities in data — enabling representation learning, synthesis, and reasoning.  
- Instead of directly memorizing examples, the model learns a probabilistic process that could have *generated* them.

> Generative models explain the data by learning *how it might have arisen.*

 

### 5.2.2 From Maximum Likelihood to Latent Variable Models

#### Maximum Likelihood Principle
The ideal objective for learning a generative model is to maximize the likelihood of the observed data:
$$
\mathbb{E}_{p^*(x)}[\log p_\theta(x)]
$$
where $p^*(x)$ is the true data distribution and $p_\theta(x)$ is the model.

#### Latent Variable Formulation
- Assume data arises from hidden (latent) variables $z$:
  $$
  \log p_\theta(x) = \log \int p_\theta(x|z)\,p(z)\,dz
  $$
- Here:
  - $p(z)$ — prior over latent variables (e.g., $\mathcal{N}(0, I)$)  
  - $p_\theta(x|z)$ — likelihood or *decoder* mapping latent codes to data

This defines a latent variable model:  
the data-generating process maps from a *low-dimensional latent space* to the observed space.

 
### 5.2.3 Inference in Latent Variable Models

Goal: infer the posterior
$$
p(z|x) = \frac{p_\theta(x|z)p(z)}{p_\theta(x)}
$$
to identify which latent factors $z$ most likely generated observation $x$.

- Intuition:  
  Recover the underlying causes that explain the data — along with uncertainty estimates.
- Problem:  
  Computing $p(z|x)$ is often intractable, since $p_\theta(x)$ involves integrating over all $z$.  
  → We must approximate inference using neural networks.

Thus, generative models combine:

- Generation: $z \rightarrow x$ (decode latent causes into data)
- Inference: $x \rightarrow z$ (encode data into latent causes)

 
### 5.2.4 Variational Autoencoders (VAEs)

To make inference tractable, VAEs introduce an approximate posterior $q_\phi(z|x)$ and optimize a variational bound on the likelihood:

#### Evidence Lower Bound (ELBO)
$$
\log p_\theta(x)
\ge 
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
- D_{KL}[q_\phi(z|x)\,||\,p(z)]
$$

#### Terms
1. Reconstruction term  
   Encourages the model to faithfully reproduce the input from its latent code.

2. KL divergence term  
   Regularizes the latent posterior to match the prior — ensuring smoothness and preventing overfitting.

 

#### Neural Implementation
- Encoder $q_\phi(z|x)$: approximates inference (maps data → latent code).  
- Decoder $p_\theta(x|z)$: generates data from the latent space (latent → data).  
- Both are parameterized by deep neural networks.

Reparameterization trick (Kingma & Welling, 2014):  
$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
enables backpropagation through stochastic latent sampling.

 
#### Why VAEs Matter
- Provide continuous, structured latent spaces capturing generative factors.  
- Support smooth interpolation and semantic manipulation.  
- Foundation for disentangled and interpretable representation learning (e.g., β-VAE).  
- Bridge probabilistic modeling with deep learning.

> VAEs turn probabilistic inference into a scalable neural optimization problem — the cornerstone of modern generative representation learning.



### 5.2.4 β-VAE
- Adds weight β to KL term:
  $$
  \mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta D_{KL}[q(z|x)||p(z)]
  $$
- Encourages disentangled latent factors (position, shape, rotation, color).
- Provides interpretable, semantically meaningful representations.

- DARLA (Higgins et al., 2017): β-VAE for reinforcement learning → improved transfer and sim2real generalization.


### 5.2.5 Sequential and Layered Models

ConvDRAW (Gregor et al., 2016)  
- Sequential VAE with recurrent refinement.  
- Models temporal and spatial dependencies.

MONet (Burgess et al., 2019)  
- Attention-based scene decomposition.  
- Each latent corresponds to one object → compositional representations.  
- Enables object-centric reasoning and RL transfer.

GQN (Eslami et al., 2018)  
- *Generative Query Networks*: learn neural scene representations.  
- Given partial observations, predict unseen viewpoints (3D reasoning).

VQ-VAE (van den Oord et al., 2017)  
- Learns discrete latent variables via vector quantization.  
- Enables hierarchical or symbolic structure.  
- Useful for speech, images, and video.
-

### 5.2.6 GANs (Goodfellow et al., 2014)
- Implicit generative models — learn by adversarial game:
  - Generator creates samples.
  - Discriminator provides learning signal (no reconstruction loss).
- BigBiGAN (Donahue et al., 2019):
  - Adds encoder for inference.
  - Learns rich, high-level representations → SOTA semi-supervised performance on ImageNet.



### 5.2.7 Large-Scale Generative Models
- GPT (Radford et al., 2019):  
  - Large transformer trained via language modeling.
  - Learns general representations useful for multiple downstream tasks (few-shot transfer).



## 5.3 Contrastive Learning

### Core Idea
- No need to model $p(x)$ explicitly.
- Learn representations that maximize mutual information between related samples.

### 5.3.1 word2vec (Mikolov et al., 2013)
- Predict context words given a target word.  
- Contrastive objective: classify positive (true context) vs negative (random) samples.
- Learns semantic embeddings; supports few-shot translation.


### 5.3.2 Contrastive Predictive Coding (CPC, van den Oord et al., 2018)
- Maximize mutual information between current representation and future observations.  
- Trains a classifier to distinguish real future samples from negatives.  
- Learns features useful across modalities (vision, speech).

- Data-efficient Image Recognition (Hénaff et al., 2019):  
  contrastive features outperform pixel-level training in low-data regimes.

### 5.3.3 SimCLR (Chen et al., 2020)
- Simple, scalable contrastive framework:
  - Generate two augmented views of the same image.
  - Maximize agreement via contrastive loss (NT-Xent).
- Achieves state-of-the-art performance on ImageNet with linear evaluation.
- Demonstrates that contrastive signals + strong augmentations suffice for representation learning.



## 5.4 Self-Supervised Learning

### Idea
- Design *pretext tasks* that use natural structure in data as supervision.  
- Representations are deterministic and transferable to new tasks.


### 5.4.1 Examples
| Task | Description | Reference |
|-------|--------------|------------|
| Colorization | Predict color from grayscale image | Zhang et al., 2016 |
| Context Prediction | Predict position of image patches | Doersch et al., 2015 |
| Sequence Sorting | Predict correct frame order in videos | Lee et al., 2017 |
| BERT (Devlin et al., 2019) | Masked language modeling + next sentence prediction | Revolutionized NLP |


### 5.4.2 Key Benefits
- Requires no labels — just structure in data.  
- Produces general features useful for:
  - Semi-supervised classification  
  - Transfer learning  
  - Downstream reasoning tasks


### 5.5 Design Principles

| Consideration | Desired Property |
|----------------|------------------|
| Modality | Align architecture with data type (image, text, audio) |
| Task Design | Choose pretext that aligns with useful features |
| Consistency | Maintain temporal/spatial coherence |
| Discrete + Continuous Latents | Enable symbolic and continuous reasoning |
| Adaptivity | Representations should evolve with experience |


Summary:  
Unsupervised representation learning uses *three complementary lenses*:
- Generative → model what the world looks like.  
- Contrastive → learn what is similar or different.  
- Self-supervised → create pseudo-tasks that reveal structure.  
Together, they aim for data-efficient, transferable, and interpretable representations.
