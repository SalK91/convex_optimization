# Chapter 2: Convolutional Neural Networks (CNNs)  

## 1. Core Principles: Locality and Translation Invariance

Before understanding convolutional networks, it’s crucial to grasp why they exist — the structural priors they impose on data.

### 1.1 Locality

In many real-world signals (e.g., images, audio, text), nearby elements are highly correlated, while distant ones are less related. This is called the principle of locality.

For example:

- Adjacent pixels in an image often belong to the same object or texture.
- Neighboring audio samples belong to the same phoneme.
- Nearby words in a sentence influence each other’s meaning.

MLPs treat every input dimension as *independent*, ignoring these spatial correlations. CNNs fix this by restricting connections: each neuron sees only a small, *local region* of the input, called its receptive field.

Formally, for an input $x \in \mathbb{R}^{H \times W}$, a neuron at position $(i,j)$ in a CNN depends only on values in a small window $\Omega(i,j)$:
$$
h_{i,j} = \sigma\!\left(\sum_{(m,n)\in \Omega(i,j)} W_{m,n} \, x_{i+m, j+n} + b\right)
$$
This allows CNNs to learn spatially local filters, like edge detectors or texture extractors.

---

### 1.2 Translation Invariance

Natural patterns are repeatable across locations — the same feature (e.g., an edge, a cat’s ear) can appear *anywhere* in the image.

An MLP would need to learn a separate detector for each position.  
CNNs overcome this through weight sharing: the same filter $W$ is applied across all spatial positions.

Mathematically:
$$
(I * W)(i,j) = \sum_{m,n} I(i+m,j+n)\, W(m,n)
$$

This operation — convolution — ensures translation equivariance:
$$
f(T_\Delta I) = T_\Delta f(I)
$$
meaning if the input shifts by $\Delta$, the output shifts by the same amount.  
After pooling, this becomes translation invariance, i.e. the output doesn’t change under small shifts.

These two properties — locality and translation invariance — are the foundation of convolutional architectures.

 
## 2. Motivation: Why Convolutions?

While MLPs are universal function approximators, they are inefficient for data with spatial or local structure, such as images, audio, or videos.  
An MLP flattens input data into a 1D vector, destroying spatial relationships and requiring a huge number of parameters.

Example:  
For a 256×256 RGB image (≈200K input features), even one hidden layer with 1,000 neurons requires:
$$(256 \times 256 \times 3) \times 1000 = 196\,\text{million weights}.$$

Moreover, the MLP learns redundant patterns (e.g., the same edge in multiple regions).

Convolutional Neural Networks address this by exploiting spatial locality, translation invariance, and weight sharing.

 
## 3. The Convolution Operation

### 3.1 Discrete Convolution
A convolution is a linear operation where a small filter (kernel) slides over an input and computes local weighted sums.

For 2D inputs (e.g. images):

$$
S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) K(m,n)
$$

- $I$ — input (image)
- $K$ — kernel (filter)
- $S$ — output feature map

Each filter detects a specific local pattern (edges, corners, textures).

 
### 3.2 Convolution in Neural Networks

In CNNs, the convolution becomes a learnable operation:

$$
h_{i,j,k} = \sigma\left( \sum_{c=1}^{C_\text{in}} (W_{k,c} * x_c)_{i,j} + b_k \right)
$$

- $x_c$: input channel $c$ (e.g. R, G, B)
- $W_{k,c}$: kernel for output channel $k$ and input channel $c$
- $b_k$: bias for output channel $k$
- $\sigma$: nonlinearity (ReLU, etc.)

This produces $C_\text{out}$ feature maps, each representing a learned spatial pattern.

Weight sharing drastically reduces parameters:  
Each kernel might be $3 \times 3$ or $5 \times 5$ — independent of image size.

 
## 4. Building Blocks of CNNs

### 4.1 Convolutional Layer
Performs learnable filtering and produces feature maps.

If input has shape $(H, W, C_\text{in})$:
- Kernel: $(k_H, k_W, C_\text{in}, C_\text{out})$
- Output: $(H', W', C_\text{out})$

### 4.2 Nonlinear Activation
After convolution, apply nonlinearity (commonly ReLU):
$$
\text{ReLU}(x) = \max(0, x)
$$

### 4.3 Pooling Layer
Reduces spatial dimensions and increases invariance.

- Max pooling: selects the largest value in a patch.
- Average pooling: takes mean value.

Formally:
$$
y_{i,j} = \max_{(m,n)\in \Omega(i,j)} h_{m,n}
$$

Pooling introduces translation invariance — small shifts in input don’t drastically change outputs.

### 4.4 Flatten + Fully Connected Layers
At the top of CNNs, feature maps are flattened and passed into MLP layers for classification or regression.

 
## 5. CNN Architecture as a Computation Graph

A typical CNN defines a differentiable map:

$$
f_\theta(x) = W_L (\text{Flatten}(h_{L-1})) + b_L
$$

where each layer $h_l$ is defined recursively as:

$$
h_l = \sigma(\text{Conv}(h_{l-1}; W_l) + b_l), \quad l = 1, \dots, L-1
$$

Here, `Conv` represents the convolution operation.

Each layer is spatially local, translation-equivariant, and differentiable — meaning backpropagation works seamlessly, just as in MLPs.

 
## 6. Backpropagation Through Convolutions

The gradient computation is a direct extension of the chain rule.

### 6.1 Forward Pass
Compute:
$$
y = W * x + b
$$

### 6.2 Backward Pass
We need:
- Gradient w.r.t. weights:  
  $\frac{\partial L}{\partial W} = x * \frac{\partial L}{\partial y}$
- Gradient w.r.t. input:  
  $\frac{\partial L}{\partial x} = \text{flip}(W) * \frac{\partial L}{\partial y}$

The flipping arises from the mathematical property of convolution.  
Modern frameworks handle this efficiently via *convolution transpose* operations.

Optimization viewpoint:  
Convolution layers remain linear in their weights — the nonlinearity and local parameter sharing define their expressive power.

 

## 7. Inductive Biases in CNNs

Convolutional architectures embed *strong inductive biases*:

| Property | Mathematical Mechanism | Effect |
|-----------|------------------------|---------|
| Local connectivity | Small kernels (3×3, 5×5) | Exploits spatial locality |
| Weight sharing | Same filter across space | Reduces parameters drastically |
| Translation equivariance | Convolution operation | Same pattern detection anywhere |
| Pooling invariance | Spatial downsampling | Robust to small shifts/noise |

These biases make CNNs data-efficient and easy to train — especially compared to fully connected networks on images.

 

## 8. Optimization and Training Dynamics

Training CNNs is similar to MLPs — we use gradient-based optimizers (SGD, Adam, etc.) — but with different landscape geometry:

- Parameter sharing makes the loss smoother (less overfitting).
- Batch normalization stabilizes gradient flow:
  $$
  \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
  $$
- Regularization via dropout or weight decay improves generalization.
- Learning rate scheduling (cosine, step decay, warm restarts) accelerates convergence.

Empirical finding: CNNs optimize faster and generalize better on spatial data due to structured parameterization.

 
## 9. CNN Architectures Through History

| Model | Year | Key Innovation | Depth | Inductive Bias |
|--------|------|----------------|--------|----------------|
| LeNet-5 | 1998 | First practical CNN for handwritten digits | 7 layers | Local receptive fields |
| AlexNet | 2012 | GPU training, ReLU, dropout | 8 layers | Data augmentation |
| VGG | 2014 | Deep stacks of small 3×3 filters | 19 layers | Uniform architecture |
| ResNet | 2015 | Skip connections for gradient flow | 152 layers | Identity mapping |
| DenseNet | 2016 | Feature reuse via dense connectivity | 201 layers | Multi-scale learning |
| EfficientNet | 2019 | Compound scaling | variable | Optimized parameter scaling |

 
## 10. CNNs and the Optimization Landscape

CNNs reshape the optimization problem compared to MLPs:

- Reduced parameter redundancy → fewer degenerate directions in gradient space.
- Structured weight sharing → smoother loss surface, fewer sharp minima.
- Skip connections (ResNets) introduce *identity mappings*, improving conditioning of the Jacobian and preventing vanishing gradients.

In optimization terms, CNNs are better-conditioned models of the input–output mapping.

 
## 11. Beyond Classical CNNs

Modern vision architectures have evolved:
- Residual Networks (ResNets): skip connections allow training very deep models.
- Depthwise Separable Convolutions (MobileNet, EfficientNet): reduce parameter count.
- Dilated Convolutions: expand receptive field without extra parameters.
- Convolution + Attention hybrids: combine locality (CNN) with global context (Transformers).

 

## 12. Mathematical Summary

| Concept | Formula | Description |
|----------|----------|-------------|
| Convolution | $(I * K)(i,j) = \sum_m \sum_n I(i+m,j+n) K(m,n)$ | Weighted local sum |
| CNN Layer | $h = \sigma(W * x + b)$ | Convolution + nonlinearity |
| Pooling | $y_{i,j} = \max_{(m,n)\in \Omega(i,j)} h_{m,n}$ | Downsampling |
| Gradient wrt weights | $\frac{\partial L}{\partial W} = x * \frac{\partial L}{\partial y}$ | Backprop step |
| Gradient wrt input | $\frac{\partial L}{\partial x} = \text{flip}(W) * \frac{\partial L}{\partial y}$ | Sensitivity propagation |

 

## 13. Intuitive Summary

Convolutional networks are:
- Local → they process neighborhoods of data.
- Hierarchical → deeper layers build on lower-level features.
- Translation-equivariant → same pattern anywhere is treated the same.
- Efficient → far fewer parameters than MLPs.

They form the backbone of modern computer vision, speech recognition, and even some transformer hybrids (ConvNeXt, ViT hybrids).

