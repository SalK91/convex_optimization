# Chapter 1 — Introduction to Information Theory (for Machine Learning)

Information theory provides a mathematical foundation for uncertainty, compression, communication, and learning. In modern ML and DL, information theory underlies:

- loss functions (cross-entropy, NLL)
- representation learning and contrastive learning
- variational inference and VAEs
- generative modeling (GANs, flows, diffusion)
- reinforcement learning (entropy bonuses, policy KL constraints)
- model capacity, generalization, and bottlenecks

This chapter introduces the core motivations and conceptual tools.

 

## 1. Why Information Theory Matters for ML

Information theory answers questions fundamental to ML:

- How much uncertainty does a model reduce?
- How do we quantify the difference between two probability distributions?
- How do we measure dependence between variables?
- What is the maximum information a neural network layer can transmit?
- How do we formalize compression and generalization?

In ML, information theory is not abstract mathematics —  
it provides the *language* for describing learning itself:

> Learning = finding distributions that compress data optimally  while preserving information relevant for prediction.

This viewpoint unifies:

- Maximum likelihood  
- Variational inference  
- Contrastive learning  
- GAN objectives  
- Representation learning  
- Reinforcement learning signal shaping  

 
## 2. The Communication View (Shannon’s Formulation)

A classical communication system consists of:

1. Source:    Generates data (symbols, images, text, states).

2. Encoder: Transforms data into a compressed or structured representation (ML analogy: neural encoders, feature extraction, token embedding).

3. Channel: Communication medium; may be noisy or bandwidth-limited  (ML analogy: stochastic layers, dropout, variational noise).

4. Decoder: Reconstructs the data (ML analogy: neural decoders, autoregressive models).

5. Receiver:   Obtains the final predictions or reconstructions.

Information theory studies:

- Limits of efficient communication  
- Optimal encoding and representation  
- Tradeoffs between compression and fidelity  
- Effect of noise on learnability

 
## 3. The Uncertainty View (Shannon–Bayesian Perspective)

Information theory also quantifies *uncertainty*:

- More uncertainty → more information needed  
- Less uncertainty → easier prediction and compression  

Key idea: Information is the reduction of uncertainty.

In ML:

- Entropy measures label uncertainty  
- Cross-entropy measures model fit  
- KL divergence measures mismatch  
- Mutual information measures representation quality  
- ELBO measures how well a generative model explains data  

Thus, learning and compression are mathematically the same problem.


## 4. Machine Learning as Communication

Modern ML pipelines resemble a communication system:

### Data → Encoder → Latent Representation → Decoder → Output

Examples:

- Autoencoders / VAEs: compress \(x\) into \(z\), then reconstruct
- Transformers: compress sequences into features, decode predictions
- Contrastive models (SimCLR, CPC): maximize MI between views of data
- GANs: learn generator distributions close to data distribution
- RL agents: compress sensory input into state representations

Thus, the principles governing communication capacity, coding, and noise apply directly to network design.

  
### 5. Roadmap for This Web-book

This web-book is structured to build information theory specifically for ML:

1. Entropy & Self-Information  
   Foundations of uncertainty, coding length, and compression.

2. Cross-Entropy & Negative Log-Likelihood  
   Core ML loss; the bridge between probability and training objectives.

3. KL Divergence & f-Divergences  
   Quantifying model mismatch, VI, GAN divergences.

4. Jensen–Shannon & Wasserstein Distances  
   GAN stability, geometric learning, distribution metrics.

5. Mutual Information & Estimation Bounds  
   Representation learning, contrastive learning, InfoNCE.

6. Variational Inference & ELBO  
   VAEs, Bayesian deep learning, posterior approximations.

7. Information Bottleneck & Representation Theory  
   Why deep networks compress, and how representations generalize.

8. Summary & Concept Map  
   Unifying view of entropy → KL → MI → VI → representation learning.
 