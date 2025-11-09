# Chapter 3: Modeling Sequence Data in Deep Learning
In machine learning, a sequence is an ordered list of elements (e.g. words, time-series measurements) where the order of elements carries meaning. Formally, a sequence of length $T$ can be written as $(x_1,x_2,\dots,x_T)$, where each element $x_t$ is indexed by its position in the sequence. Elements can repeat (e.g. the word “the” may appear multiple times), and different sequences may have different lengths. Thus sequence data is inherently variable-length and order-dependent.

Sequences are collection of elements where:

- Elements can be repeated.
- Order matters.
- Of variable length.

## Limitations of Traditional Supervised Models: 
Traditional supervised models (e.g. fixed-size feedforward neural networks or classifiers) expect inputs of a fixed dimension and have no built-in notion of order or memory. In practice, applying a standard feedforward net to sequence data – by, say, collapsing the sequence into a fixed-size feature vector – ignores the important temporal or sequential structure. As one summary notes, “feedforward neural networks are severely limited when it comes to sequential data”. Indeed, trying to predict a time-series or next word in a sentence by a fixed snapshot yields poor results. The key missing capability in traditional networks is memory of the past: they cannot readily model how earlier parts of the sequence influence later outputs. 

Concretely, most classifiers assume each input example is independent and fixed-size. A sentence of variable length or a time-series with long-term correlations violates this assumption. Thus, classical models fail because they have no mechanism to store or process long-term context: they either throw away order information or arbitrarily truncate sequences. Feedforward networks also do not share parameters over time, so each time-step would have its own weights (infeasible for long sequences).

## The Simplest Assumption: Independent Words (Bag-of-Words)

A naïve approach to sequence (especially text) is to assume all elements are independent. In language, this is like a bag-of-words model (or unigram model) that ignores word order. In a bag-of-words representation, one simply counts or models each word’s occurrence, treating all words as “independent features.” This ignores sequence structure: “the order of words in the original documents is irrelevant”. Such a model can still do document classification by word frequency, but it cannot predict the next word or capture meaning that depends on word order. Critically, bag-of-words assumes word occurrences are uncorrelated: “bag-of-words assumes words are independent of one another”. In reality, words co-occur in context (“peanut butter” versus “peanut giraffe”) – bag-of-words misses all such dependencies. Thus the independent-words assumption breaks down for sequence modeling, motivating models that explicitly use ordering and context.

## N-gram Models and Fixed-Context Assumptions
To go beyond complete independence, one can incorporate local context by using $n$-gram models. An $n$-gram model makes the (Markov) assumption that the probability of each element depends only on the previous $n-1$ elements. For language, a bigram model (2-gram) assumes $P(w_t\mid w_{t-1})$, a trigram (3-gram) uses $P(w_t\mid w_{t-2},w_{t-1})$, etc. In general, the chain rule with an $N$-gram approximation is

$$
P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{t-N+1}, \ldots, x_{t-1}) \, .
$$


This preserves some order information: the window of the last $N-1$ items is used to predict the next. However, $n$-gram models have well-known downsides:

- Limited context length: They cannot capture dependencies beyond the fixed window. As noted in the literature, language “cannot reason about context beyond the immediate $n$-gram window”, and dependencies span entire sentences or documents. For example, a 3-gram model cannot connect a subject at the start of a sentence to its verb at the end if they are more than two words apart. Thus any longer-range dependency is missed by an $n$-gram.

- Data sparsity and scalability: The number of possible $n$-grams grows exponentially with vocabulary size $V$. For a vocabulary of size $V$, there are $V^N$ possible $N$-grams. Jurafsky & Martin observe that even for Shakespeare’s corpus ($V\approx 29{,}066$), there are $V^2\approx8.4\times 10^8$ possible bigrams and $V^4\approx 7\times 10^{17}$ possible 4-grams. Most of these never occur, so the resulting probability tables are extremely sparse. Training requires huge corpora to observe enough $n$-gram counts, and storing these tables is impractical for large $N$ or $V$. In practice, language models become “ridiculously sparse” and unwieldy.

- No parametrization (non-differentiable): Traditional $n$-gram models are simply tables of counts with smoothing. They are not learned via gradient descent, so integrating them into larger neural pipelines (or backpropagating through them) is not straightforward. They lack nonlinearity and share no features across contexts.

In summary, while $n$-grams preserve local order up to length $N$, they suffer from fixed-window limitations and massive tables, motivating more compact, learnable alternatives.

## Learnable Context Models: Vectorization and Neural Nets

Modern sequence models address these issues by representing context with vectors and training parametric models. Key features of a learnable sequential model include:

- Vector representation (embedding) of words and context: Each element (e.g. a word) is mapped to a continuous vector. Context (the recent history) can be summarized by combining or encoding these vectors into a fixed-size context vector. This preserves order by using the positions of the context vectors in the encoding.

- Order sensitivity: Unlike bag-of-words, the model output depends on the order of context elements. For example, we might concatenate or otherwise encode a sequence of word embeddings, ensuring different sequences yield different context vectors.

- Variable-length compatibility: The model should handle inputs of differing lengths. For instance, recurrent or attention models can process a variable number of inputs sequentially. Context-vectors built from the sequence (such as by a recurrent state) grow as needed. As noted, context-vector methods can “operate in variable length of sequences”.

- Differentiability: The mapping from context vector to next-word probability should be a differentiable function (e.g. a neural network) so we can train by gradient descent. This requires using continuous, learnable transformations (matrices, nonlinearities) instead of fixed count tables.

- Nonlinearity: Neural networks allow complex (nonlinear) interactions among inputs. A simple linear model on concatenated embeddings might be too weak, so one often uses at least one hidden layer with a nonlinear activation (e.g. tanh, ReLU).

For example, one could take the last few words, map each to an embedding $\mathbf{x}{t-N+1},\dots,\mathbf{x}{t-1}$, concatenate them into one large vector, and feed it into a multilayer perceptron (MLP) to predict the next word’s probability. This would be order-sensitive and differentiable. However, it still fixes the context window size ($N-1$) and uses a separate weight for each position, so it’s not efficient or variable-length. 

A more flexible approach is to encode arbitrary prefixes of the sequence into a single context (memory) vector using a recurrent or recursive process. One introduces a context vector $\mathbf{h}_t$ that evolves as the sequence is read. Such a context-vector “acts as memory” summarizing the past. A context-vector model has crucial advantages: it preserves order, handles variable-length inputs, and is fully trainable (differentiable). In short, vectorized context models can “learn” how much each part of the past matters, via backpropagation, while maintaining the sequence structure.

## Recurrent Neural Networks (RNNs)
These considerations lead naturally to Recurrent Neural Networks (RNNs) – models specifically designed for sequences. An RNN processes one element at a time, maintaining a hidden state (context vector) that is updated recurrently. At each time step $t$, the RNN takes the current input $\mathbf{x}t$ and the previous hidden state $\mathbf{h}{t-1}$ and computes a new hidden state $\mathbf{h}_t$. The simplest RNN update is:

$$
h_t = \phi(W_h h_{t-1} + W_x x_t + b) \, .
$$


where $\phi$ is a nonlinear activation (often $\tanh$) and $W_h,W_x$ are weight matrices. The same weight matrices $W_h,W_x$ are reused at every time step (this is parameter sharing), which gives the RNN the ability to handle sequences of any length. As noted, this weight sharing means the model uses constant parameters across time.

Intuitively, the RNN’s hidden state $\mathbf{h}_t$ “remembers” the information from all prior inputs up to time $t$. The final hidden state (or the hidden state at each step) can then be fed to an output layer to make predictions. Typically, we compute an output distribution over the next element via a softmax layer:

$$
y_t = \mathrm{softmax}(W_y h_t + b_y) \, .
$$


so that $P(x_{t+1}=w \mid \mathbf{h}_t)$ is given by the corresponding component of $\mathbf{y}_t$. In language modeling, for instance, $y_t$ gives a probability for each word in the vocabulary. As described in practice, “RNNs predict the output from the last hidden state along with output parameter $W_y$; a softmax function to ensure the probability over all possible words”. 

In summary, RNNs explicitly model order and context via their hidden state updates and shared parameters. They can be seen as a recurrent generalization of feedforward networks: an “MLP with shared weights across time.” At time $t$, the RNN effectively takes the previous state and new input and feeds them through a nonlinear layer to compute the new state. Because information flows from each state to the next, the RNN can, in principle, capture long-range dependencies: any input can influence all future hidden states.

## Unrolling and Backpropagation Through Time (BPTT)
Training an RNN is done by backpropagation through time. Conceptually, we unfold or unroll the RNN across $T$ time steps, creating a deep feedforward network of depth $T$ (each layer corresponds to one time step) with tied weights. One then applies standard backpropagation on this unfolded network. Formally, the total loss (e.g. sum of cross-entropies at each step) depends on the sequence of outputs, and gradients are computed by propagating errors backward through the unfolded time dimension. As one overview explains, “the network needs to be expanded, or unfolded, so that the parameters could be differentiated ... – hence backpropagation through time (BPTT)”. In practice, each weight matrix $W$ receives gradient contributions from each time step, effectively summing gradients as they propagate back. BPTT thus accounts for how current errors depend on all previous inputs through the recurrent hidden state. Because parameters are shared across time, the gradient at each step flows through multiple copies of the layer. BPTT differs from ordinary backpropagation only in that errors are summed at each time step due to weight sharing. Concretely, if $L = -\sum_t \log P(x_t\mid \mathbf{h}_{t-1})$ is the loss, then for each $W$ we compute

$$
\frac{\partial L}{\partial W} = \sum_{t} \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W} \, .
$$


taking into account the influence of $W$ at every time step. In implementation, we typically use truncated BPTT (backprop through a limited number of steps) for efficiency on long sequences. But in principle, gradients propagate through all time steps, linking distant inputs to distant outputs.

## Vanishing and Exploding Gradients
A critical challenge in training RNNs is that the repeated nonlinear transformations can cause gradients to vanish or explode during BPTT. Mathematically, the derivative $\partial \mathbf{h}t/\partial \mathbf{h}{t-1}$ involves the Jacobian of the activation and the recurrent weights. Over many steps, the gradient involves a product of many such Jacobians. Just as multiplying many numbers less than 1 quickly goes to zero, multiplying many matrices with spectral radius $<1$ causes the gradients to shrink exponentially (vanishing), while if the spectral radius is $>1$ they blow up (exploding). The exploding gradient problem arises when the norm of the gradient grows exponentially (due to eigenvalues $>1$), whereas the vanishing gradient problem occurs when long-term components of the gradient go “exponentially fast to norm 0”. Formally, for a linearized RNN one can show that if the largest eigenvalue $\lambda_{\max}$ of the recurrent weight matrix satisfies $|\lambda_{\max}|<1$, long-term gradients vanish as $t\to\infty$, and if $|\lambda_{\max}|>1$ they explode. 

Vanishing gradients mean that inputs from the distant past have almost no effect on the gradient of the loss, so the model learns only short-term dependencies. Exploding gradients make training unstable (weights take huge jumps). Both phenomena are well-documented: “when long term components go to zero, the model cannot learn correlation between distant events.” In practice, it is common to observe gradients either shrinking toward zero over time or blowing up and causing numerical issues in RNNs, especially with long sequences.


## Gated Architectures: LSTM and GRU
To mitigate the vanishing gradient, gated RNN architectures such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) were introduced. These architectures incorporate learnable “gates” that control the flow of information and create paths for gradients to propagate more easily. Long Short-Term Memory (LSTM): An LSTM cell augments the basic RNN with a cell state $\mathbf{C}_t$ and three gates: input ($\mathbf{i}_t$), forget ($\mathbf{f}_t$), and output ($\mathbf{o}t$) gates. Each gate is a sigmoid unit that decides how much information to let through. Formally, at time $t$ with input $\mathbf{x}t$ and previous hidden $\mathbf{h}{t-1}$ and cell $\mathbf{C}{t-1}$, the gates and cell update are given by (all operations are elementwise):

​$$
\begin{aligned}
i_t &= \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i), \\
f_t &= \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f), \\
o_t &= \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o), \\
\tilde{C}_t &= \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c) \, .
\end{aligned}
$$

The new cell state $\mathbf{C}_t$ is then updated by combining the old state and the candidate:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \, .
$$


where $\odot$ denotes elementwise multiplication. Finally, the hidden state (output of the LSTM) is

$$
h_t = o_t \odot \tanh(C_t) \,
$$

The intuition is that the forget gate $\mathbf{f_t}$ can reset or retain the old memory $\mathbf{C}_{t-1}$, the input gate $\mathbf{i}_t$ controls how much new information $\tilde{\mathbf{C}}_t$ to write, and the output gate $\mathbf{o}_t$ controls how much of the cell state to expose as $\mathbf{h}_t$. By design, if the forget gate is near 1 and input gate near 0, the cell state is simply carried forward unchanged; gradients can flow through this constant path, avoiding vanishing. In practice, LSTMs “alleviate the vanishing gradient problem,” making it easier to train on long sequences. The gating architecture enables the network to learn to keep or discard information over many time steps. 

In practice, using LSTM or GRU units yields much better performance on sequence tasks like language modeling or translation than vanilla RNNs.


## Optimization Challenges and Solutions
Even with gating, training RNNs can be tricky. Besides architectural fixes, optimization techniques are crucial:

- Gradient clipping: To handle exploding gradients, one common technique is gradient clipping. Before updating parameters, one clips the norm of the gradient vector to some threshold (rescaling if too large). This prevents any single update from blowing up. As Pascanu et al. note, clipping “solves the exploding gradients problem” by limiting gradient norm. Clipping was key to many RNN successes (e.g. in language modeling), and it is standard practice in modern frameworks.

- Orthogonal (or careful) initialization: Choosing a good initial recurrent weight matrix can help. Initializing $W_h$ as an (scaled) orthogonal matrix ensures its eigenvalues have magnitude 1, which prevents immediate vanishing/exploding. In fact, orthogonal matrices preserve the norm of vectors, so repeated multiplications neither decay nor explode. As one tutorial explains, “Orthogonal initialization is a simple yet relatively effective way of combating exploding and vanishing gradients,” ensuring stable gradient propagation. In practice, some implementations initialize $W_h$ to random orthogonal (or unitary) matrices to encourage long memory.

- Layer normalization or gating enhancements: Techniques like layer normalization inside LSTM cells, or using newer architectures (e.g. LayerNorm-LSTM, transformer-like attention), also alleviate training difficulties.

- Regularization: Some works add penalties to encourage $W_h$ to have a controlled spectral radius, or use techniques like weight noise or dropout to stabilize training.

In summary, sequence modeling requires architectures and training methods that explicitly handle order, context, and long-range information. Traditional models fail because they lack memory and flexibility. N-gram models give a glimpse of sequential structure but cannot scale or generalize. Recurrent models – especially gated RNNs – provide a powerful framework: mathematically, they define hidden states $\mathbf{h}_t$ updated by $\mathbf{h}t = f(\mathbf{h}{t-1},\mathbf{x}_t)$ with shared weights, and training via BPTT. Gating (LSTM/GRU) adds control mechanisms that preserve gradients and selective memory. With appropriate initialization, clipping, and optimization, these RNN-based models form the foundation of modern sequence learning. 