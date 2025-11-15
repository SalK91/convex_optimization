## Chapter 5 — Representation Learning, Mutual Information, and the Information Bottleneck

Representation learning seeks transformations of data that make tasks such as prediction, compression, and reasoning easier. A representation $Z$ is typically obtained by applying an encoder to an input $X$. Information theory provides a natural way to formalize what makes a representation useful by analyzing the mutual information between $Z$, the input $X$, and the target $Y$.

This chapter introduces mutual information as a measure of shared structure, explains the Information Bottleneck framework, and connects these ideas to deep learning methods such as contrastive learning, VAEs, and self-supervised learning.

---

## 1. What Is a Representation?

A representation is a transformed form of input data:

$$
Z = f_\theta(X),
$$

where $f_\theta$ is usually a neural network. A good representation should satisfy two goals:

1. It should retain information that is relevant for predicting $Y$.  
2. It should discard noise or irrelevant aspects of $X$.

Information theory allows us to express these goals using mutual information.

---

## 2. Mutual Information: Connecting Two Variables

Mutual information (MI) measures how much knowledge of one variable reduces uncertainty about another:

$$
I(X;Y) = H(X) - H(X|Y).
$$

It can also be written as a KL divergence:

$$
I(X;Y) = D_{\text{KL}}(p(x,y)\|p(x)p(y)).
$$

MI is zero when $X$ and $Y$ are independent and increases as $Y$ becomes more predictable from $X$.

In representation learning, we are often interested in the two quantities:

$$
I(Z;Y), \qquad I(Z;X).
$$

These measure how informative the representation $Z$ is regarding the target $Y$ and how much irrelevant detail from $X$ is still present.

---

## 3. The Role of MI in Representation Learning

A representation $Z$ is desirable when:

- $I(Z;Y)$ is large  
  (the representation captures features relevant to prediction)

- $I(Z;X)$ is small  
  (the representation removes noise and redundancy)

This idea appears in supervised learning, contrastive methods, and generative modeling.

Some examples:

- In supervised learning, we want features that preserve label information.  
- In contrastive learning, we want features that preserve the information common across augmented views.  
- In generative models, latent variables should retain structure that explains the data while avoiding unnecessary detail.

---

## 4. The Information Bottleneck Principle

The Information Bottleneck (IB) formalizes the trade-off between informativeness and compression. The goal is:

$$
\max I(Z;Y)
\quad \text{s.t.} \quad 
I(Z;X) \text{ is small}.
$$

This can be written as the Lagrangian:

$$
\mathcal{L}_{\text{IB}}
=
I(Z;Y) - \beta I(Z;X).
$$

The parameter $\beta$ controls how aggressively the representation is compressed.

- Large $\beta$ leads to simpler, more compressed representations.  
- Small $\beta$ allows more expressive, detailed representations.

IB provides a theoretical explanation for the behavior of learned features in deep neural networks.

---

## 5. Deep Learning and the Information Bottleneck

IB theory suggests several statements about deep networks:

1. Early layers preserve much of the information in $X$.  
2. Later layers tend to compress $X$ while emphasizing information predictive of $Y$.  
3. Networks may first memorize and later compress during training.  
4. Generalization is linked to discarding unnecessary information.

Although the exact dynamics remain debated, the overall perspective helps interpret the evolution of features during training.

---

## 6. Variational Information Bottleneck (VIB)

Mutual information terms are often difficult to compute directly. The Variational Information Bottleneck approximates them using variational distributions.

We treat the representation as a random variable drawn from $q(z|x)$ and estimate MI with tractable terms. The VIB objective is:

$$
\mathcal{L}_{\text{VIB}}
=
\mathbb{E}_{p(x,y)}\!
\left[
\mathbb{E}_{q(z|x)}\![\log p(y|z)]
\right]
-
\beta D_{\text{KL}}(q(z|x)\|p(z)).
$$

This resembles the VAE objective, but $p(y|z)$ replaces the reconstruction term. The first term encourages predictive features, while the KL term compresses the representation.

VIB therefore provides a practical implementation of the Information Bottleneck.

---

## 7. Mutual Information and Contrastive Learning

Contrastive learning uses mutual information to learn representations without labels. The idea is:

- Generate two augmented views of the same input: $(x_1, x_2)$.  
- Encode them as $(z_1, z_2)$.  
- Encourage $z_1$ and $z_2$ to be similar.  
- Encourage representations of different inputs to be dissimilar.

This encourages $Z$ to retain the information that is preserved under augmentation, while ignoring irrelevant aspects of the input.

Many methods follow this structure:

- SimCLR  
- MoCo  
- BYOL  
- InfoNCE  
- CPC  
- Deep InfoMax  

The InfoNCE objective is:

$$
\mathcal{L}_{\text{NCE}}
=
-\mathbb{E}\left[
\log
\frac{\exp(\text{sim}(z_i,z_j)/\tau)}
{\sum_k \exp(\text{sim}(z_i,z_k)/\tau)}
\right],
$$

where $(i,j)$ is a positive pair. InfoNCE is a variational lower bound on $I(Z_1;Z_2)$.

---

## 8. MI in Generative Modeling

Mutual information also appears in generative models:

### 8.1 VAEs  
The KL term controls the structure and redundancy of $Z$, and the decoder ensures $I(Z;X)$ stays large enough for accurate reconstruction.

### 8.2 InfoGAN  
This model maximizes:

$$
I(c; G(z,c)),
$$

encouraging the generator to learn interpretable latent factors.

### 8.3 Normalizing flows  
Flows maintain $I(X;Z) = H(X)$ because they are invertible; they do not compress the input.

### 8.4 Diffusion models  
Diffusion models gradually reduce noise and can be interpreted using information-theoretic ideas related to KL divergence and score matching.

---

## 9. MI and Disentanglement

Disentangled representations aim to separate independent generative factors such as orientation or color. The $\beta$-VAE objective:

$$
\mathcal{L}
=
\mathbb{E}[\log p(x|z)]
-
\beta D_{\text{KL}}(q(z|x)\|p(z))
$$

encourages disentanglement by increasing compression in the latent space. A larger $\beta$ pushes different dimensions of $Z$ to encode more independent aspects of the data.

---

## 10. Estimating MI in High Dimensions

Mutual information is difficult to compute exactly in high dimensions. Neural estimation relies on variational bounds such as:

- InfoNCE  
- NWJ bound  
- Donsker–Varadhan bound  
- MINE estimator  
- f-divergence lower bounds  

These allow MI to be used in representation learning even when the true quantities are intractable.

---

## 11. Summary of Chapter 5

Mutual information provides a principled measure of what makes a useful representation: it should retain information relevant for prediction and discard irrelevant detail. The Information Bottleneck formalizes this trade-off and motivates practical methods such as the Variational Information Bottleneck.

Contrastive learning methods maximize MI between augmented views, enabling self-supervised representation learning. Generative models such as VAEs, GANs, flows, and diffusion models each manipulate mutual information in different ways, leading to distinct behaviors and capabilities.

Information theory therefore provides a unified lens through which to understand representation learning in modern deep networks.

