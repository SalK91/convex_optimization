Sampling methods provide numerical techniques for approximating integrals, expectations, and posterior distributions that are analytically intractable. They are an essential component of Bayesian inference and appear in many areas of machine learning, including reinforcement learning, probabilistic modeling, and generative models.

This chapter introduces sampling in a structured sequence, beginning with independent sampling, progressing to Monte Carlo estimation, extending to Markov chain Monte Carlo (MCMC), and concluding with advanced techniques and ML-specific applications.



## 1. The Goal of Sampling

Many problems require computing expectations of the form

$$
\mathbb{E}_{p(z)}[f(z)] = \int f(z)\,p(z)\,dz,
$$

or evaluating posterior quantities such as

$$
p(z|x) = \frac{p(x,z)}{p(x)}.
$$

Direct computation is rarely feasible because the integral may be high-dimensional or have no closed form.

Sampling provides a way to approximate these quantities using draws from the distribution.

 

## 2. Independent Sampling

Independent sampling methods produce samples where each draw does not depend on the previous one.

These methods work best when:

- sampling directly from $p(z)$ is tractable  
- the distribution is low-dimensional  
- the support is simple (e.g., Gaussian, uniform)  

 
### 2.1 Direct Sampling

When the distribution has an invertible CDF $F(z)$:

1. sample $u \sim \text{Uniform}(0,1)$  
2. compute $z = F^{-1}(u)$  

This yields exact samples. It is commonly used in:

- uniform sampling  
- exponential distributions  
- simple discrete distributions  

 
### 2.2 Importance Sampling

When sampling from $p(z)$ is difficult but evaluating $p(z)$ is easy, importance sampling draws samples from a proposal distribution $q(z)$ and reweights them:

$$
\mathbb{E}_{p(z)}[f(z)]
=
\mathbb{E}_{q(z)}\left[f(z)\frac{p(z)}{q(z)}\right].
$$

The weights

$$(z)=
\frac{q(z)}{p(z)}
$$	​

correct for the fact that samples were drawn from $q$ instead of $p$.

The main challenge is weight variability. If $q(z)$ does not closely match $p(z)$, the ratio $p(z)/q(z)$ becomes extremely uneven: most samples have tiny weights, while a few samples have very large weights. This produces high-variance estimates because the estimator becomes dominated by a handful of rare but extremely influential samples. In high-dimensional spaces, designing a proposal $q(z)$ that covers the important regions of $p(z)$ is especially difficult, making importance sampling unreliable unless the proposal distribution is carefully chosen.
 
### 2.3 Rejection Sampling

Rejection sampling draws exact samples from a target distribution $p(z)$ by using a simpler proposal distribution $q(z)$ and accepting or rejecting candidate samples based on how well $q$ covers $p$. The method requires a constant $M$ such that
$$
p(z) \le M q(z) \quad \text{for all } z.
$$

This condition ensures that $Mq(z)$ forms an envelope that completely contains $p(z)$.

Procedure:

1. sample $z \sim q(z)$  
2. accept with probability $\frac{p(z)}{M q(z)}$  

If accepted, $z$ is guaranteed to be an exact draw from $p$.

Rejection sampling is conceptually simple and does not distort the target distribution, but it becomes inefficient in many settings. When $q(z)$ is a poor match for $p(z)$, the constant $M$ must be large, which means that most samples are rejected. In high-dimensional spaces, the mismatch between $p$ and $q$ typically worsens exponentially, making the acceptance probability extremely small. As a result, rejection sampling is rarely practical for modern high-dimensional machine-learning models, although it remains useful in low-dimensional problems or when $q$ can be chosen to closely match $p$.
 
## 3. Monte Carlo Estimation

Monte Carlo approximates expectations by:

$$
\mathbb{E}_{p(z)}[f(z)] \approx \frac{1}{N}\sum_{i=1}^N f(z_i),
\quad z_i \sim p(z).
$$

Key properties:

- error scales as $\mathcal{O}(1/\sqrt{N})$  
- works in high dimensions  
- accuracy depends on sampling quality  

Monte Carlo is the backbone of almost all probabilistic computation.



## 4. Markov Chain Monte Carlo (MCMC)

When sampling directly from $p(z)$ is hard, Markov Chain Monte Carlo constructs a Markov chain

$$
z_1 \to z_2 \to z_3 \to \cdots
$$

whose stationary distribution is $p(z)$.

After a burn-in period, samples approximate $p(z)$ even if individual states are dependent.

MCMC is widely applicable because it does not require the normalization constant of $p(z)$:

$$
p(z|x) \propto p(x,z).
$$


### 4.1 Metropolis–Hastings Algorithm

The Metropolis–Hastings (MH) algorithm constructs a Markov chain whose stationary distribution is the target distribution $p(z)$, even when $p(z)$ is known only up to a proportionality constant. This makes MH suitable for Bayesian inference, where the posterior is often available only in unnormalized form:

$$p(z \mid x) \propto p(x, z)$$

MH works by proposing a new point based on the current state and then accepting or rejecting it according to how well it aligns with the target distribution.


Given a current sample $z$, the algorithm proceeds as follows:

1. propose a new sample $z'$ using a proposal distribution $z' \sim q(z' \mid z)$.

2. compute the acceptance probability
     $$
     \alpha = \min\left(1, \frac{p(z')\,q(z|z')}
     {p(z)\,q(z'|z)}
     \right).
     $$

3. accept the proposal with probability $\alpha(z,z')$
otherwise remain at the current state If accepted, set $z_{t+1} = z'$, otherwise keep $z_{t+1} = z$.


This simple rule ensures that the Markov chain satisfies detailed balance and converges to the desired distribution $p(z)$.

Metropolis–Hastings is flexible and works with virtually any distribution from which we can evaluate $p(z)$ up to a constant. However, its efficiency depends strongly on the proposal distribution. If the proposal steps are too small, the chain performs a random walk and mixes slowly. If the steps are too large, most proposals are rejected. Choosing or adapting the proposal distribution is therefore crucial for performance, especially in high-dimensional settings.
 

### 4.2 Gibbs Sampling

Gibbs sampling is a special case of MCMC designed for multivariate distributions where sampling from the full conditional distributions is easy. Instead of proposing a new state and accepting or rejecting it, Gibbs sampling updates one variable at a time by drawing directly from its exact conditional distribution.

For a latent vector:
$$z = (z_1, z_2, \dots, z_d)$$


a Gibbs update for coordinate $i$ samples:

$$
z_i \sim p(z_i \mid z_{-i}).
$$

where $z_{-i}$ denotes all components except $z_i$.

By cycling through all coordinates repeatedly, the Markov chain eventually converges to the target joint distribution $p(z)$.

The key requirement is that each full conditional distribution

$$p(z_i \mid z_{-i})$$

must be analytically tractable and easy to sample from. When this holds, Gibbs sampling is simple to implement and avoids the accept–reject step of Metropolis–Hastings.

However, Gibbs sampling can mix slowly when variables are strongly correlated, since updating one coordinate at a time may explore the space inefficiently. Gibbs sampling is widely used in models where conditional distributions are naturally available, including:

- topic models such as Latent Dirichlet Allocation (LDA)
- hidden Markov models
- Gaussian graphical models
- Bayesian networks with conjugate priors

 
### 4.3 Slice Sampling

Slice sampling chooses a height $u$ and samples uniformly along the slice:

$$
\{z : p(z) > u \}.
$$

It adapts automatically to the local shape of the distribution and requires minimal tuning.

 
## 5. Reducing Random-Walk Behaviour

Basic MCMC methods suffer from slow exploration due to random-walk behavior.  
Advanced methods reduce this inefficiency.

 
### 5.1 Hamiltonian Monte Carlo (HMC)

HMC introduces momentum variables and uses Hamiltonian dynamics to propose long-distance moves with high acceptance probability.

Advantages:

- avoids random walk behaviour  
- efficient in high dimensions  
- uses gradients of $\log p(z)$  

HMC is widely used in probabilistic programming systems (Stan, PyMC).

 
### 5.2 Overrelaxation

Overrelaxation proposes samples that are negatively correlated with the previous ones, improving mixing speed.

---


Sampling methods approximate expectations and posterior distributions when closed-form solutions are unavailable. Independent methods such as importance and rejection sampling are simple but limited. Monte Carlo estimation provides a general framework for approximating integrals, and MCMC allows sampling from complex, high-dimensional distributions by constructing Markov chains. Advanced methods such as Hamiltonian Monte Carlo improve mixing and efficiency.

Sampling is a central tool for Bayesian inference and underlies many modern machine learning models, from deep generative architectures to reinforcement learning algorithms.
