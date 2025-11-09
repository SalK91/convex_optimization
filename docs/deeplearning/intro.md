# An Introduction to Neural Networks

 
## 1. Neural Networks as Computation Graphs

Modern neural networks are best understood as differentiable computation graphs. 
They are not just layered algebraic systems but structured compositions of primitive mathematical operations.

Each node in this graph corresponds to a function:

$$z_i = f_i(x_1, \dots, x_k)$$

and the entire network defines a composite function:

$$f_\theta(x) = f_L \circ f_{L-1} \circ \dots \circ f_1(x)$$

where $\theta = \{W_i, b_i\}$ denotes all learnable parameters.

### Formal Structure
For a Multilayer Perceptron (MLP):

$$h_0 = x, \quad h_i = \sigma(W_i h_{i-1} + b_i), \quad i=1,\dots,L-1, \quad \hat{y} = W_L h_{L-1} + b_L$$

with: $W_i \in \mathbb{R}^{d_i \times d_{i-1}}, \quad b_i \in \mathbb{R}^{d_i}$

Each layer is a small differentiable function. When we connect them, we form a composite map — the fundamental abstraction underlying *autodiff*, *backprop*, and *learning*.

Key property: Because every node in the graph is differentiable, the entire function $f_\theta(x)$ is differentiable with respect to both input $x$ and parameters $\theta$.

Graphically, the network is a directed acyclic graph (DAG):

- Edges: carry tensor values.
- Nodes: represent differentiable functions.
- Forward pass: evaluates node outputs.
- Backward pass: propagates sensitivities (gradients) backward.

> This graph abstraction unifies all architectures — CNNs, RNNs, Transformers, Diffusion Models — as differentiable computation graphs.

 
## 2. Gradients, Jacobians, and Differentiation

For any function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian matrix $J_f(x)$ encodes local derivatives:

$$[J_f(x)]_{ij} = \frac{\partial f_i}{\partial x_j}$$

In neural networks, we often deal with a scalar loss function:

$$L(\theta) = \ell(f_\theta(x), y)$$

and want: 

$$\nabla_\theta L = \frac{\partial L}{\partial \theta}$$

However, computing full Jacobians is computationally infeasible — for a network with millions of parameters, explicit Jacobians would have trillions of entries.  
Instead, automatic differentiation (autodiff) computes vector–Jacobian products efficiently.

For scalar loss $L$: $\nabla_\theta L = J_{f_\theta}(x)^T \nabla_{f_\theta} L$

where $J_{f_\theta}(x)$ is the Jacobian of the output w.r.t. parameters.

This operation can be done efficiently in reverse-mode autodiff — the heart of backpropagation.

---

## 3. Forward and Backward Passes

#### Forward Pass

Given input $x$ and parameters $\theta$:

1. Compute layer outputs sequentially: $h_i = \sigma(W_i h_{i-1} + b_i)$
2. Compute loss $L = \ell(f_\theta(x), y)$
3. Store intermediate activations $h_i$ for reuse during backpropagation.

This pass evaluates the function $L(\theta)$.

#### Backward Pass
The backward pass applies the chain rule in reverse, computing derivatives of the loss with respect to each parameter:

$\frac{\partial L}{\partial \theta_i} = 
\frac{\partial L}{\partial h_L}
\frac{\partial h_L}{\partial h_{L-1}}
\dots
\frac{\partial h_{i+1}}{\partial \theta_i}$

The chain rule guarantees that this derivative can be factored into local derivatives of each layer, which can be computed efficiently.

Reverse-mode autodiff (backprop) algorithm:

1. Initialize $\bar{h}_L = \frac{\partial L}{\partial h_L} = 1$.
2. For each layer $l = L, L-1, \dots, 1$:
   - Compute local derivative $\frac{\partial h_l}{\partial h_{l-1}}$
   - Accumulate gradient:  
     $\bar{h}_{l-1} = \bar{h}_l \frac{\partial h_l}{\partial h_{l-1}}$
   - Compute parameter gradients:  
     $\frac{\partial L}{\partial W_l} = \bar{h}_l (h_{l-1})^T$
3. Return all $\nabla_\theta L$.

This process requires the cached activations from the forward pass, which explains the memory cost of backpropagation.

 

## 4. Chain Rule, Backpropagation, and Automatic Differentiation

The chain rule underpins all gradient computation.  
For scalar functions:

$\frac{dL}{dx} = \frac{dL}{dz} \frac{dz}{dx}$

and recursively for multivariate functions:

$\nabla_x L = J_{z}(x)^T \nabla_z L$

Autodiff implements this automatically, performing either:

- Forward-mode AD: propagates derivatives forward, efficient when #inputs ≪ #outputs.
- Reverse-mode AD: propagates derivatives backward, efficient when #outputs ≪ #inputs (our case).

Reverse-mode AD ≡ backpropagation.

Computational Complexity:
- Cost ≈ 2× forward pass (one forward, one backward).
- Memory ≈ size of stored activations.

Optimization viewpoint:   Autodiff converts the learning problem into an optimization problem over parameters:

$\min_\theta L(\theta)$

where $L$ is differentiable but nonconvex. Backprop provides the exact gradient needed by optimization algorithms.
s
## 5. From Gradients to Optimization

The Learning Problem - Training a neural network means solving:

$\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}} [\,\ell(f_\theta(x), y)\,]$

Since the true data distribution $\mathcal{D}$ is unknown, we use empirical risk minimization (ERM):

$\min_\theta \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$

This is a high-dimensional, nonconvex optimization problem. The parameter space may have millions (or billions) of dimensions.Despite this, gradient-based methods — powered by backpropagation — reliably find good solutions.


### First-Order Optimization Algorithms

All modern deep learning optimization relies on gradients:

$\nabla_\theta L = \frac{\partial L}{\partial \theta}$

The basic rule: update parameters in the direction of *negative gradient*:

$\theta_{t+1} = \theta_t - \eta \nabla_\theta L_t$

where $\eta$ is the learning rate.

#### Stochastic Gradient Descent (SGD)
We use mini-batches instead of full data:

$\theta_{t+1} = \theta_t - \eta \nabla_\theta \frac{1}{|B_t|}\sum_{i \in B_t} \ell(f_\theta(x_i), y_i)$

- Cheap per-step computation.
- Introduces *gradient noise*, which helps escape shallow minima and saddle points.

#### Momentum
Accelerates learning by accumulating a velocity vector:

$v_{t+1} = \mu v_t - \eta \nabla_\theta L_t, \quad \theta_{t+1} = \theta_t + v_{t+1}$

Momentum smooths oscillations and stabilizes descent on curved loss surfaces.

#### Adam (Adaptive Moment Estimation)
Maintains exponentially weighted averages of gradients and squared gradients:

$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla_\theta L_t$

$v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla_\theta L_t)^2$

Bias-corrected updates:

$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

Adam adapts the learning rate per-parameter, combining momentum with RMS normalization.

#### Second-Order and Curvature-Aware Methods
While first-order methods use only gradients, second-order methods consider curvature (Hessian):

$H = \frac{\partial^2 L}{\partial \theta^2}$

Newton’s update:

$\theta_{t+1} = \theta_t - H^{-1}\nabla_\theta L$

is theoretically optimal for quadratic loss but computationally infeasible for deep nets.  
Approximations like L-BFGS, K-FAC, and natural gradient descent use low-rank or structured approximations to curvature.

 
### Optimization Landscape and Gradient Flow

Although neural network loss surfaces are highly nonconvex, they possess *favorable geometry*:

- Most critical points are saddle points, not local minima.
- Wide, flat minima generalize better (implicit regularization of SGD).
- Gradient noise helps explore valleys in high-dimensional space.

Gradient flow (continuous limit of SGD):

$\frac{d\theta(t)}{dt} = - \nabla_\theta L(\theta(t))$

describes a trajectory in parameter space governed by the vector field of gradients.

The optimization algorithm defines the *dynamics* of this flow (e.g., momentum adds inertia).

 

## 6. What MLPs Can’t Do?

### (a) Multiplicative Interactions
MLPs compute sums of weighted activations — inherently *additive* operations:

$h = \sigma(Wx + b)$

They cannot naturally represent multiplicative relationships (like $x_1 x_2$) unless approximated via nonlinear stacking, which is inefficient.

Architectures with multiplicative gates (LSTMs, Transformers) encode such interactions directly, improving optimization dynamics by linearizing multiplicative effects.

### (b) Attention and Dynamic Routing
MLPs have static connectivity. Attention mechanisms compute data-dependent weights, enabling context-sensitive computation:

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$

Optimization over attention parameters effectively learns a dynamic kernel, something MLPs cannot emulate efficiently.

### (c) Metric Learning and Inductive Bias
MLPs lack structural priors about similarity or geometry.  
Optimization in unstructured parameter spaces can overfit and fail to generalize relational properties.

Architectures like CNNs (translation equivariance), GNNs (permutation invariance), and Transformers (contextual attention) bake inductive biases into the computation graph, making optimization more efficient — the landscape becomes smoother and gradients more informative.

 

## 7. Beyond Backprop: Curvature, Generalization, and Geometry

Advanced optimization in neural networks goes beyond plain gradient descent.

### Natural Gradient
Instead of minimizing loss directly in parameter space, we minimize it in *function space*:

$\Delta \theta = - \eta F^{-1} \nabla_\theta L$

where $F$ is the Fisher information matrix:

$F = \mathbb{E}\left[\nabla_\theta \log p_\theta(x) \nabla_\theta \log p_\theta(x)^T\right]$

Natural gradients move along directions that respect the underlying information geometry of the model.

### Implicit Bias of Gradient Descent
Even in overparameterized models, gradient descent tends to find *low-norm* or *flat* minima that generalize better — a phenomenon not yet fully understood but deeply tied to the optimization path and noise structure of SGD.

### Optimization as Inference
Many modern perspectives view training as approximate inference:

$p(\theta | D) \propto e^{-L(\theta)/T}$

Gradient descent samples from this energy landscape as $T \to 0$; stochastic variants like SGD approximate Bayesian inference under certain limits.
