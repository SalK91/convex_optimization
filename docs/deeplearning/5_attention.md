## 1. Attention, Memory, and Cognition

- Attention = ability to focus on relevant signals and ignore distractions.  
  - Enables selective processing (e.g. cocktail party effect).  
  - Allows focusing on one thought or event at a time.

- Memory provides continuity: keeping information over time to guide behavior or reasoning.

- Together, they form the basis of cognition — controlling what to process, store, and recall.

- Neural networks can model aspects of this by learning *what to attend to* and *what to remember*.

- Goal of attention in DL:  
  Reduce complexity by focusing computation on the most informative parts of data or internal state.


## 2. Implicit Attention in Neural Networks

- Neural networks are parametric nonlinear functions $y = f_\theta(x)$ mapping inputs to outputs.  
  They naturally exhibit *implicit attention*: certain input dimensions influence outputs more.

- The Jacobian $J = \frac{\partial y}{\partial x}$ quantifies this sensitivity — shows which input parts the model “pays attention” to.

- Example:  
  In deep RL, sensitivity maps reveal focus on *state-value* vs *action-advantage* components.

- Recurrent Neural Networks (RNNs) extend this to sequences:  
  - Hidden state $h_t$ stores past info.  
  - The *sequential Jacobian* $\frac{\partial y_t}{\partial x_{t-k}}$ shows which past inputs are remembered.  
  - Implicitly attends to relevant time steps (memory through recurrence).

- In tasks like machine translation, implicit attention lets models reorder tokens:
  > “to reach” → “zu erreichen”


## 3. Explicit (Hard) Attention

- Explicit attention introduces a separate *attention mechanism* that decides where to look or what to read.  
  It restricts the data fed to the main network.

### Why explicit attention?
- Efficiency: processes only selected parts of input.  
- Scalability: works on large or variable-size data.  
- Sequential processing: e.g. moving “gaze” across static images.  
- Interpretability: easier to visualize focus regions.

### Model structure
- Network outputs attention parameters $a$ that define a *glimpse distribution* $p(g|a)$ over possible data regions.  
- A glimpse $g$ (subset or window of data) is sampled and passed back as input.  
- System becomes recurrent, even if the base network is not.

### Training (non-differentiable)
- When glimpse selection is discrete or stochastic, use REINFORCE:
  $$
  \nabla_\theta \mathbb{E}[R] = \mathbb{E}[(R - b) \nabla_\theta \log \pi_\theta(g)]
  $$
  where $R$ is the reward (e.g. task loss) and $b$ a baseline for variance reduction.

- Thus, attention acts as a policy $\pi_\theta(g)$ over glimpses.

### Examples
- Recurrent Models of Visual Attention (Mnih et al., 2014): learns a sequence of foveal glimpses for image classification.  
- Multiple Object Recognition with Visual Attention (Ba et al., 2014): attends sequentially to multiple objects.

## 4. Soft Attention

- Hard attention samples discrete glimpses → non-differentiable → needs RL.  
- Soft attention computes a *weighted average* over all glimpses → differentiable → trainable by backprop.

### Basic idea
- Attention parameters $a$ define weights $w_i$ over input features $v_i$:
  $$
  v = \sum_i w_i v_i, \quad \sum_i w_i = 1
  $$
  The readout $v$ is a smooth combination of inputs.

- Replaces sampling by *expectation* → continuous, differentiable.

### Benefits
- Trained end-to-end with gradients.  
- Easier and more stable than hard attention.  
- Allows *focus distribution* rather than a single point.

### Variants
- Location-based attention: focuses by spatial position (e.g. Gaussian over coordinates).  
- Content-based attention: focuses by similarity of key $k$ to data vectors $x_i$ via score $S(k, x_i)$, usually normalized by softmax:
  $$
  w_i = \frac{\exp(S(k, x_i))}{\sum_j \exp(S(k, x_j))}
  $$

### Applications
- Handwriting synthesis: RNN learns soft “window” over text sequence.  
- Neural Machine Translation: associative attention aligns words between languages.  
- DRAW model: uses Gaussian filters to read/write parts of an image.

- Soft attention = *data-dependent dynamic weighting* (similar to convolution with adaptive filters).

## 5. Introspective Attention and Memory

- So far: attention over external data.  
- Now: attention over internal state or memory → “introspective attention.”  
  - Lets the network *read* or *write* selectively to memory locations.  
  - Enables reasoning, recall, and algorithmic behavior.

### Neural Turing Machine (NTM)
- Adds a differentiable memory matrix $M \in \mathbb{R}^{N \times W}$.  
- Controller (RNN) interacts with memory using differentiable attention mechanisms.

Operations
- Write: modify selected rows in $M$ using attention weights $w_t$.  
- Read: output weighted sum of memory slots:
  $$
  r_t = \sum_i w_{t,i} M_i
  $$
- Addressing modes:
  - *Content-based*: match key vector $k_t$ to memory contents (via cosine similarity).  
  - *Location-based*: shift attention by relative position.

Training: fully differentiable — end-to-end via backprop.

Example task: copying sequences of variable length — learns algorithmic generalization.


### Differentiable Neural Computer (DNC) 
- Successor to NTM with richer memory access:
  - Tracks temporal links between writes.  
  - Supports dynamic memory allocation.  
  - Improves stability and scalability.

Application: synthetic QA tasks (bAbI dataset) — answers questions requiring multiple supporting facts and temporal reasoning.

 
Key insight:  
Attention provides *selective access* to memory, acting like “addressing” in a differentiable data structure.

## 6. Transformers and Self-Attention

- Transformers: remove recurrence and convolution entirely — rely only on attention.

### Self-Attention
- Each token attends to all others in the sequence:
  $$
  \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
  $$
  where:
  - $Q, K, V$ are query, key, and value matrices (learned linear projections of input embeddings).
  - Produces context-aware representations for all tokens in parallel.

### Multi-Head Attention
- Multiple attention “heads” ($H$) learn different relationships:
  $$
  \text{MultiHead}(Q,K,V) = [h_1; h_2; \dots; h_H] W^O
  $$
  Each head captures a distinct pattern (syntax, semantics, position, etc.).

### Transformer Block
- Structure:
  1. Multi-head self-attention  
  2. Add & LayerNorm  
  3. Feedforward (ReLU + linear)  
  4. Add & LayerNorm  
- Skip connections improve gradient flow and allow top-down signal mixing.

### Positional Encoding
- Since model is permutation-invariant, inject position information:
  $$
  \text{PE}_{(pos,2i)} = \sin(pos / 10000^{2i/d})
  $$
  $$
  \text{PE}_{(pos,2i+1)} = \cos(pos / 10000^{2i/d})
  $$
  Added to input embeddings.

### Intuition
- Self-attention generalizes RNN memory:
  - Recurrent → sequential access  
  - Transformer → *direct pairwise access* between all tokens.
- Enables long-range dependencies and parallelization.

### Key result
- Attention-only models achieve SOTA in translation and NLP tasks.  
- Forms basis for BERT, GPT, and modern large language models.

## 7. Adaptive Computation Time (ACT) and Summary

### Adaptive Computation Time (ACT)
- Proposed by Graves (2016): allows networks to “ponder” variable amounts of time per input.  
- Each step computes a halting probability $p_t$; total halt when $\sum_t p_t = 1$.
- Output is a weighted sum of intermediate states:
  $$
  y = \sum_t p_t h_t
  $$
- Encourages efficient use of computation — more steps for harder inputs, fewer for easy ones.
- Regularized by a *time penalty* to avoid overthinking.

### Universal Transformers  
- Extend Transformers with recurrence in depth (same block applied multiple times).  
- Shares parameters across layers — like an RNN unrolled over depth.
- Combine parallel self-attention + iterative refinement + ACT.
- Achieves better generalization and adaptive reasoning on sequence tasks.

## Summary

- Attention = selective processing of relevant information.  
- Implicit attention occurs naturally in deep nets (via sensitivity).  
- Explicit attention can be hard (sampled) or soft (differentiable).  
- Memory networks (NTM, DNC) use attention to read/write differentiable external memory.  
- Transformers unify attention as the core mechanism — fully parallel, context-rich.  
- Adaptive computation gives flexibility in processing time and complexity.

Takeaway:  
Selective attention and memory — biological inspirations — are now core architectural principles driving modern deep learning.
