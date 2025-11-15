Chapter 3 — KL Divergence, f-Divergences, Jensen–Shannon Divergence, and Wasserstein Distance

This chapter introduces the major ways to quantify how different two probability distributions are. These measures underpin many areas of modern machine learning, including generative models (VAEs, GANs, flows), reinforcement learning, Bayesian inference, and representation learning. The goal is to build an intuitive and mathematical understanding suitable for a beginner, while still maintaining the depth needed for practical ML reasoning.

 
## 1. KL Divergence: Measuring Distribution Mismatch

The Kullback–Leibler (KL) divergence measures how different two probability distributions are. For distributions $p$ and $q$:

$$
D_{\text{KL}}(p\|q)
= \sum_x p(x)\log\frac{p(x)}{q(x)}.
$$

KL divergence quantifies the inefficiency incurred when encoding samples drawn from $p$ using a code optimized for $q$. If $q$ assigns very low probability to events that occur frequently under $p$, the KL divergence becomes large.

### Key properties of KL divergence

1. Non-negative  
   $$
   D_{\text{KL}}(p\|q) \ge 0.
   $$

2. Zero only when the two distributions are identical.

3. Asymmetric  
   $$
   D_{\text{KL}}(p\|q) \ne D_{\text{KL}}(q\|p).
   $$

4. Not a true metric, since it fails the triangle inequality.

5. Can be infinite when $p(x) > 0$ but $q(x) = 0$.  
   This is a crucial issue in generative modeling, where such mismatches occur frequently.

 
## 2. KL Divergence in Machine Learning

KL divergence appears throughout machine learning, often in subtle ways. The direction of KL used in an algorithm profoundly affects how the resulting model behaves.

 
### 2.1 Maximum likelihood as forward KL minimization

Training a model by maximum likelihood is equivalent to minimizing the forward KL divergence:

$$
\theta^*
= \arg\min_\theta D_{\text{KL}}(p_{\text{data}} \,\|\, q_\theta).
$$

The model is penalized heavily for failing to assign probability mass to any region where real data occurs. As a result, maximum-likelihood models attempt to cover all modes of the data distribution.

This produces mode-covering behavior, which is characteristic of:

- normalizing flows  
- autoregressive models  
- density estimation models trained via log-likelihood  

 
### 2.2 KL divergence in variational inference (VI)

Variational inference relies on minimizing the reverse KL divergence between an approximate posterior $q(z|x)$ and the true posterior $p(z|x)$:

$$
D_{\text{KL}}(q(z|x)\|p(z|x)).
$$

Since the true posterior is typically intractable, VAEs approximate this with:

$$
D_{\text{KL}}(q(z|x)\|p(z)).
$$

Reverse KL heavily penalizes placing probability mass in regions where the target distribution has little or none. This leads the model to concentrate on a single high-density mode and avoid uncertain areas.

This behavior is known as mode seeking. In VAEs, it contributes to smooth or blurry reconstructions, because the model often collapses to a conservative “safe” solution.

 
### 2.3 KL divergence in reinforcement learning

Modern policy gradient methods constrain policy updates using KL divergence. For example, TRPO and PPO penalize large deviations between the previous policy and the new one:

$$
D_{\text{KL}}(\pi_{\text{old}} \,\|\, \pi_{\text{new}}).
$$

This keeps learning stable by preventing abrupt policy changes that might harm performance.

 
### 2.4 KL divergence in distillation and compression

KL divergence compares two probability distributions directly and is used for:

- teacher–student distillation  
- compressing large models into smaller ones  
- aligning probability distributions across layers  
- calibrating output probabilities  

Whenever we want one model to imitate another, KL divergence naturally appears.

 
## 3. Understanding KL Behavior: Mode Covering vs. Mode Seeking

The two directions of KL divergence behave very differently. Understanding this distinction is central to understanding why VAEs blur, GANs collapse, and flows cover all modes.

 
### Forward KL: $D_{\text{KL}}(p\|q)$  
*(Used in maximum likelihood, flows → mode covering)*

Forward KL asks whether the model $q$ assigns sufficient probability wherever the data distribution $p$ has mass:

> “Does the model assign enough probability to every place where the data occurs?”

If $q$ misses even a small region where $p$ has mass, the divergence becomes very large. The model is therefore encouraged to spread probability across all data modes.

Result: **mode covering**  
The model covers every part of the data distribution, even rare modes. It tolerates false positives (assigning probability where there is no data) but avoids false negatives (missing data modes).

Flows and MLE-based models display this behavior.

 
### Reverse KL: $D_{\text{KL}}(q\|p)$  
*(Used in VI, VAEs, GAN-like behavior → mode seeking)*

Reverse KL asks the opposite question:

> “Is the model placing probability in places where the data distribution is very small or zero?”

Reverse KL heavily penalizes placing mass in low-density regions of $p$, making the model conservative.

Result: **mode seeking**  
The model places most of its mass at a single safe mode, often ignoring minor modes. This produces sharp or collapsed samples, depending on the context.

VAEs, many VI methods, and GAN-like formulations exhibit mode seeking.

 

## 4. f-Divergences: A Unified Family of Divergences

KL divergence belongs to a larger family called f-divergences. An f-divergence is defined by a convex function $f$:

$$
D_f(p\|q) = \sum_x q(x)\, f\!\left(\frac{p(x)}{q(x)}\right).
$$
 

## 5. Jensen–Shannon Divergence: The Original GAN Divergence

The Jensen–Shannon (JS) divergence measures how different two distributions are using a mixture distribution:

$$
\text{JS}(p\|q)
= \frac12 D_{\text{KL}}(p\|m)
+ \frac12 D_{\text{KL}}(q\|m)
$$

where the mixture is:

$$
m = \frac12(p+q).
$$

JS divergence is symmetric and always lies between 0 and $\log 2$.

### Why JS appears in GANs

GANs train a discriminator using binary cross entropy. When the discriminator is trained to optimality, the resulting generator objective becomes:

$$
\text{JS}(p\|q) - \log 2.
$$

Thus, GANs naturally minimize JS divergence without explicitly choosing it. This symmetry and boundedness initially made JS seem ideal.

 
## 6. Why JS Divergence Causes GAN Instability

At the beginning of GAN training, real samples and generated samples usually do not overlap. When the supports of $p$ and $q$ are disjoint:

$$
\text{JS}(p\|q) = \log 2.
$$

In this regime, JS divergence becomes constant and the gradient becomes zero.

Consequences:

1. The discriminator immediately becomes perfect.  
2. The generator stops receiving meaningful gradients.  
3. Training often collapses, oscillates, or diverges.  

This gradient-vanishing problem motivated the development of Wasserstein GANs.

 
## 7. Total Variation and Hellinger Distances

Unlike KL or JS, these are true metrics: symmetric, finite, and geometrically meaningful.

 
### 7.1 Total Variation (TV) Distance

$$
\text{TV}(p,q) = \frac12\sum_x |p(x)-q(x)|.
$$

TV measures the maximum possible difference in probabilities assigned to events by the two distributions. It corresponds to the minimum amount of probability mass that must be moved to transform $p$ into $q$.

Applications in ML:

- Robustness under distribution shift  
- Generalization bounds (PAC-Bayes)  
- Fairness and safety  

 
### 7.2 Hellinger Distance

$$
H^2(p,q)
= \frac12 \sum_x\left(\sqrt{p(x)} - \sqrt{q(x)}\right)^2.
$$

Hellinger distance compares the square roots of probabilities, producing a smooth and bounded measure between 0 and 1.

Uses in ML include:

- Robust statistics  
- Domain adaptation  
- Generalization theory  
- Some GAN formulations  

 
## 8. Wasserstein Distance: Geometry of Probability Distributions

The Wasserstein-1 (Earth Mover) distance measures how much work is needed to move probability mass from one distribution to another.

 
### 8.1 Primal form (Earth Mover interpretation)

$$
W(p,q)
= \inf_{\gamma \in \Gamma(p,q)}
\mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|].
$$

It seeks the transport plan $\gamma$ requiring the least expected effort to turn $p$ into $q$.

 
### 8.2 Dual form (used in WGAN)

$$
W(p,q)
= \sup_{\|f\|_L\le 1}
\left(\mathbb{E}_p[f(x)]
     - \mathbb{E}_q[f(x)]\right).
$$

GANs implement $f$ as a neural network called a critic. The critic must be 1-Lipschitz to ensure stable gradients.

 
### 8.3 Why Wasserstein solves GAN instability

Wasserstein distance has several advantages:

- Provides informative gradients even with no overlap  
- Reflects the actual geometry of the data space  
- Avoids the saturation and vanishing gradients of JS divergence  
- Works reliably in high-dimensional spaces  

These properties make Wasserstein GANs far more stable than classical GANs.

 
### 8.4 WGAN-GP: Gradient Penalty

To enforce the Lipschitz condition, WGAN-GP adds a gradient penalty:

$$
\lambda(\|\nabla_x f(x)\|_2 - 1)^2.
$$

This produces smoother and more stable training compared to weight clipping.

 
## 9. Divergence versus Distance

Divergences such as KL and JS:

- may be infinite  
- are asymmetric  
- do not behave well when distributions have disjoint support  

Distances such as Wasserstein, TV, and Hellinger:

- are symmetric  
- obey triangle inequality  
- remain meaningful under distribution shift  

In machine learning:

- Divergences are useful for inference and likelihood  
- Distances are useful for generative modeling and geometry  

 
## 10. Why Divergences Fail in High Dimensions

In high-dimensional spaces:

- Real and generated samples rarely overlap  
- KL divergence often becomes infinite  
- JS divergence becomes flat  
- Gradients vanish  

Wasserstein distance solves these issues by relying on geometric structure rather than probability ratios.

---

 
KL divergence quantifies mismatch between distributions and plays a central role in likelihood-based learning, variational inference, reinforcement learning, and distillation. The choice between forward and reverse KL determines whether a model exhibits mode-covering or mode-seeking behavior.

The f-divergence family generalizes KL and provides a unified view of GAN objectives. Jensen–Shannon divergence arises naturally in classical GAN training but suffers from gradient-vanishing problems when real and fake data do not overlap.

Total Variation and Hellinger distances offer robust, metric-based ways to compare distributions. Wasserstein distance introduces a geometric perspective that overcomes the limitations of KL and JS, enabling stable GAN training via WGAN and WGAN-GP.

