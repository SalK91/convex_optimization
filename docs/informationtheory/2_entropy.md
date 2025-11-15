# Chapter 2 — Entropy, Self-Information, Cross-Entropy & Information Measures
 

## 1. Self-Information (Surprisal)

Self-information quantifies the *surprise* of observing an event.

For an event with probability \(p(x)\):

\[
I(x) = - \log_2 p(x)
\]

Why the log?

- Additivity of independent events  
- Probability → information monotonicity  
- Log base 2 → units in bits  
- Log-likelihoods become additive → ML becomes convex (in many models)

Interpretations:

- Unlikely events carry more information  
- Certain events carry zero information  
- Foundation of cross-entropy and negative log-likelihood  

In ML:  

- The loss used in classification is simply the surprisal of the correct class.

 
## 2. Entropy — Expected Uncertainty

Entropy is the expected self-information:

\[
H(X) = -\sum_x p(x) \log p(x)
\]

Entropy measures:

- Uncertainty  
- Randomness  
- Compressibility  
- Difficulty of prediction  

### Key properties:
- \(H(X) = 0\) if a variable is deterministic  
- Maximum when distribution is uniform  
- Upper bound on achievable compression (Shannon)

### ML Interpretation:
- High entropy labels → noisy dataset → harder learning
- Activation entropy reflects network expressiveness
- Entropy of output distribution measures model confidence
- Entropy regularization improves exploration in RL

 
## 3. Differential Entropy (Continuous Entropy)

For continuous variables:

\[
h(X) = -\int p(x) \log p(x)\,dx
\]

Important differences:

- Can be negative
- Not invariant under reparameterization
- Not comparable between different coordinate systems

### Why it matters in ML:
- VAEs use continuous latent variables \(z\)
- Flows and diffusion models use continuous densities
- Score-based models estimate gradients of log-densities, not densities directly

Differential entropy is not the same thing as Shannon entropy — a common source of confusion.

 
## 4. Joint, Conditional, and Total Entropy

### Joint entropy:
\[
H(X,Y) = -\sum_{x,y} p(x,y)\log p(x,y)
\]

### Conditional entropy:
\[
H(Y|X) = -\sum_{x,y} p(x,y)\log p(y|x)
\]

Interpretation:

- Average residual uncertainty in \(Y\) after observing \(X\)

### Chain rule of entropy:
\[
H(X,Y) = H(X) + H(Y|X)
\]

This rule is foundational for:

- Autoregressive modeling  
- Sequence modeling  
- Transformers (predictive factorization)  
- Bayesian networks  

 
## 5. Cross-Entropy — Coding \(p\) Using \(q\)

Cross-entropy is the expected surprise under model \(q\):

\[
H(p, q) = -\sum_x p(x)\log q(x)
\]

### Crucial identity:
\[
H(p, q) = H(p) + D_{\text{KL}}(p\|q)
\]

Meaning:

- True entropy + penalty for using the wrong distribution
- Cross-entropy ≥ entropy

### ML Interpretation:
Cross-entropy = Negative Log Likelihood:

\[
\mathcal{L} = - \log q(y_{\text{true}})
\]

This powers:

- Softmax classifiers  
- Logistic regression  
- Transformers (next-token prediction)  
- Language models (autoregressive LM)  
- Image segmentation (pixel-wise CE)  

Minimizing cross-entropy is equivalent to making model probabilities match the data distribution.

 
## 6. Perplexity — Entropy in Language Modeling

Perplexity is:

\[
\text{PPL} = 2^{H}
\]

Interpretation:

- The “effective vocabulary size” the model thinks it must guess from
- Lower perplexity = better language model

Transformers and LLMs are explicitly evaluated using this entropy-derived metric.

 
## 7. Mutual Information — Information Shared Between Variables

\[
I(X;Y) = D_{\text{KL}}(p(x,y)\|p(x)p(y))
\]

MI measures:

- How much knowing \(X\) tells us about \(Y\)
- Reduction in entropy of one variable after observing the other

### Equivalent forms:

\[
I(X;Y) = H(X) - H(X|Y)
\]

\[
I(X;Y) = H(X) + H(Y) - H(X,Y)
\]

MI links entropy and KL divergence into a unified measure of dependence.

### Why MI is critical in ML:
- Representation learning (maximize MI with labels)
- Contrastive learning (InfoNCE is a lower bound to MI)
- InfoGAN (maximize MI between latent code and output)
- Feature selection (choose features with highest MI to labels)
- Stochastic encoders control MI with constraints


## 8. The Data Processing Inequality (DPI)

If:

\[
X \rightarrow Z \rightarrow Y
\]

is a Markov chain, then:

\[
I(X;Y) \le I(X;Z)
\]

Meaning:

- Processing or compressing data cannot add information
- Neural networks cannot create information about the input  
  — they can only discard or transform it

ML relevance:

- Explains why deeper layers become more task-specialized  
- Supports the Information Bottleneck theory in deep learning  
- Ensures that any learned representation is bounded by input information  
- Helps analyze generalization and compression in deep nets

 
## 9. Entropy in Neural Networks

Entropy plays multiple roles in deep learning:

### Output entropy
Low entropy → confident predictions  
High entropy → uncertainty

### Entropy of hidden representations
- Early layers: reduce entropy (denoising)  
- Deep layers: compress irrelevant information  
- Good representations retain low entropy but high MI with labels

### Entropy regularization in RL
\[
J(\pi) += \beta H(\pi(\cdot|s))
\]
encourages exploration.

### Dropout increases entropy  
forcing models to encode more robust representations.

