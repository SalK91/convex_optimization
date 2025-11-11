# Deep Learning for Natural Language Processing

- Natural language is context-dependent, compositional, and ambiguous.
- Deep neural networks (DNNs) handle parallel, distributed, and interactive computation — ideal for modeling contextual relationships.
- Early symbolic NLP struggled with discrete word tokens and rigid grammar rules; deep models learn continuous representations that encode meaning and similarity.

###  Key Challenges of Language

Human language presents a unique set of challenges for computational models.  
Unlike artificial symbol systems, linguistic meaning is contextual, compositional, and dynamic, requiring models to infer relationships that go far beyond surface form.

- Words are not discrete symbols.  
  The same word can have several related senses depending on context — for example:  
  `face₁` (human face), `face₂` (clock face), `face₃` (to confront), and `face₄` (a person or presence).  
  Treating these as independent dictionary entries loses the shared semantic structure between them.  
  A more effective representation encodes meaning as distributed patterns in a continuous vector space, where related senses occupy nearby regions.

- Need for distributed representations.  
  Because meanings overlap and interact, we represent words not as atomic tokens but as vectors of features (syntactic, semantic, pragmatic).  
  This allows similarity, analogy, and composition to emerge geometrically — for instance, `king - man + woman ≈ queen`.

- Disambiguation depends on context.  
  The meaning of a word or phrase is determined by its linguistic surroundings.  
  For example, in “The man who ate the pepper sneezed,” the subject of *sneezed* is determined by a non-adjacent clause (*the man*), demonstrating how interpretation depends on sentence structure and longer-range dependencies.

- Non-local dependencies.  
  Natural language contains relationships between words that may be far apart in sequence.  
  Classical RNNs capture these dependencies only through sequential recurrence, which limits parallel computation and struggles with long-range information.  
  Transformers, through self-attention, handle these dependencies efficiently and in parallel by allowing each token to directly attend to every other token in the sequence.

- Compositionality.  
  The meaning of larger expressions arises from the meanings of their parts and how they are combined.  
  However, this combination is not purely linear.  
  For example, `carnivorous plant` is not simply the sum of *carnivore* and *plant* — its interpretation depends on how the features interact (*a plant that eats insects*).  
  Deep neural models capture this by learning nonlinear composition functions that reflect semantic interactions rather than mere addition.

In summary, natural language understanding requires models that can represent overlapping meanings, integrate long-range contextual information, and compose new meanings dynamically.  
Transformers achieve this by combining distributed representations with global attention mechanisms, providing a unified solution to these fundamental linguistic challenges.

 
## The Transformer Architecture

- Sequence models (RNNs, LSTMs) process tokens sequentially — limiting parallelism and long-range context.
- Transformers replace recurrence with self-attention, allowing the model to relate all words to all others simultaneously.

###  Core Mechanism: Self-Attention

Given token embeddings $ e_i \in \mathbb{R}^d $:

$$
q_i = e_i W^Q, \quad k_i = e_i W^K, \quad v_i = e_i W^V
$$

Attention weights:

$$
\alpha_{ij} = \mathrm{softmax}_j \left( \frac{q_i k_j^\top}{\sqrt{d}} \right)
$$

Output:

$$
z_i = \sum_j \alpha_{ij} v_j
$$

Each token’s new representation $ z_i $ is a contextual blend of all others.  
Captures semantic and syntactic relations without explicit recurrence.

### Multi-Head Attention

Use multiple projections $(W^Q_h, W^K_h, W^V_h)$ → multiple “heads.”  
Each head focuses on different relations (e.g. subject–verb, modifier–noun).  
Outputs are concatenated and projected back to dimension $d$:

$$
\text{MHA}(E) = [Z_1; Z_2; \dots; Z_H] W^O
$$

### Position Encoding

Since attention is permutation-invariant, Transformers add position information:

$$
\text{PE}_{(pos,2i)} = \sin(pos / 10000^{2i/d}), \quad
\text{PE}_{(pos,2i+1)} = \cos(pos / 10000^{2i/d})
$$

→ These sinusoidal signals are added to embeddings to encode word order.

### Full Transformer Block

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
+ Skip Connection
  ↓
Layer Normalization
  ↓
Feedforward Network (ReLU)
  ↓
+ Skip Connection
  ↓
Layer Normalization
  ↓
Output
```

Skip connections enable gradient flow and top-down influence.  
Stacking $N$ blocks yields hierarchical contextualization of meaning.

### Intuition

- Self-attention handles non-local relations.
- Multi-head captures multiple semantic dimensions simultaneously.
- Stacked layers build abstraction — from word-level to phrase- and discourse-level features.


## Unsupervised Learning and BERT

### The Need for Contextualized Representations

- Word embeddings like Word2Vec are static: one vector per word.
- Language understanding requires contextual embeddings: “bank” (river vs. finance).
- Transformers enable bidirectional context — understanding a word from both sides.

### BERT Pretraining Objectives

1. Masked Language Modeling (MLM)  
Randomly mask 15% of tokens, predict them:

$$
\text{Loss}_{MLM} = - \sum_{i \in M} \log P(w_i | \text{context})
$$

Encourages bidirectional encoding of meaning.

2. Next Sentence Prediction (NSP)  
Model predicts if sentence B follows sentence A.  
Builds discourse-level coherence and world knowledge.

### Architecture

- Deep bidirectional Transformer encoder.
- Uses special tokens:
  - `[CLS]` – sentence-level classification embedding
  - `[SEP]` – separates segments
- Pretrained on massive text (e.g. Wikipedia, BooksCorpus).
- Fine-tuned for downstream tasks (QA, sentiment, NER, etc.) by adding a simple classifier.

### Significance

BERT shows self-supervised pretraining → transfer learning pipeline:

```text
Pretrain (unsupervised)
   ↓
Fine-tune (supervised)
   ↓
Task-specific adaptation
```


Achieves state-of-the-art on multiple NLP benchmarks with minimal labeled data.  
Learns semantic similarity, coreference, and discourse relations implicitly.



## Grounded and Embodied Language Learning

### Motivation

- Language understanding ultimately involves relating words to the world.
- Humans learn language in context — perception, action, and social interaction.
- Grounded learning aims to give agents multimodal grounding (vision, action, language).

### Grounded Agents

- Combine perceptual input (vision), motor control (actions), and linguistic input/output.
- Train via predictive modeling — anticipate sensory outcomes from language-conditioned actions.
- Enables semantic grounding: linking word “red” to visual color, “pick up” to motor command.

### Predictive and Self-Supervised Paradigms

Agents learn representations by predicting future sensory or linguistic states:

$$
\min_\theta \mathbb{E} [ \| f_\theta(s_t, a_t) - s_{t+1} \|^2 ]
$$

→ Connects to world models and predictive coding principles in neuroscience.  
The agent’s internal model encodes both linguistic meaning and causal structure of the environment.

## Insights from DeepMind Work

- Embodied agents trained in simulated environments exhibit:
  - Systematic generalization (e.g., learning “pick up red object” → generalize to unseen colors).
  - Question answering and instruction following grounded in perception.
  - Transfer from text to embodied tasks, using pretrained linguistic encoders (like BERT) as initialization.

### Conceptual Shift

From pipeline → integrated model:

| Classic Pipeline | Embodied / Interactive Model |
|------------------|------------------------------|
| Letters → Words → Syntax → Meaning → Action | Multimodal loops: Perception ↔ Action ↔ Language ↔ Prediction |


## Conceptual Map: From Representation to Understanding

```text
Word Input
   ↓
Distributed Representations (embedding)
   ↓
Self-Attention Mechanism
   ↓
Multi-Head Parallel Processing
   ↓
Hierarchical Transformer Layers
   ↓
Contextualized Embeddings (BERT)
   ↓
Transfer Learning to Tasks
   ↓
Embodied Agents (Grounded Semantics)
   ↓
Language Understanding as Prediction + Interaction
```

### Key Transitions

Symbol → Vector: Continuous representations enable learning of semantic gradients.

Sequence → Attention: Parallel context integration replaces recurrence.

Text → Context: Pretraining captures knowledge without explicit supervision.

Language → World: Grounding links linguistic representations to sensory and causal models.

### Unifying Principle

Deep language understanding = predictive modeling of structured context
across both linguistic and environmental domains.


| Concept                     | Core Idea                          | Model / Mechanism |
| --------------------------- | ---------------------------------- | ----------------- |
| Distributed representations | Meanings as patterns, not symbols  | Embeddings        |
| Context dependence          | Sense resolution via interaction   | Self-attention    |
| Parallelism                 | All words attend to all others     | Transformer       |
| Bidirectionality            | Context from both sides            | BERT encoder      |
| Transfer learning           | Self-supervised → supervised       | Fine-tuning       |
| Grounding                   | Language tied to perception/action | Embodied agents   |
| Predictive learning         | Understanding as anticipation      | World models      |
