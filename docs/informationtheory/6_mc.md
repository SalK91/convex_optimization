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

Slice sampling is an MCMC method that avoids choosing a proposal distribution by sampling uniformly from the region (the slice) where the probability density is above a randomly chosen threshold. Given the current point $z$, slice sampling introduces an auxiliary variable $u$:

1. Draw a height

     $$u \sim \text{Uniform}(0,\, p(z))$$


2. Define the horizontal slice

     $$S = \{ z' : p(z') > u \}$$

3. Sample the next state

     $$z' \sim \text{Uniform}(S)$$

This procedure constructs a Markov chain whose stationary distribution is $p(z)$. Intuitively, the algorithm first chooses a horizontal level $u$ below the current density value and then samples uniformly from the region of the density that lies above this level.

Slice sampling adapts naturally to the local geometry of the target distribution: narrow peaks produce narrow slices, and broad regions produce wide slices, without requiring manual tuning of proposal scales. In practice, slice sampling often mixes better than basic random-walk proposals, especially when the target density varies in amplitude across different regions.

However, slice sampling requires an efficient way to identify or approximate the slice region, which may be challenging in high-dimensional or multimodal settings.
 
## 5. Reducing Random-Walk Behaviour

Basic MCMC algorithms such as Metropolis–Hastings often move in small, local steps and therefore explore the state space slowly. This random-walk behaviour leads to poor mixing and long autocorrelation times, especially in high-dimensional or strongly correlated distributions.

Several advanced MCMC techniques attempt to reduce random-walk dynamics by proposing more informed or distant moves.

### 5.1 Hamiltonian Monte Carlo (HMC)

Hamiltonian Monte Carlo reduces random-walk behaviour by introducing an auxiliary momentum variable and using Hamiltonian dynamics to propose transitions. Instead of taking small, diffusive steps, HMC simulates the motion of a particle moving smoothly through the probability landscape.

Let $z$ denote the position (the latent variable) and let $r$ denote an auxiliary momentum variable. The joint density of $(z, r)$ is defined through a Hamiltonian:

$$
H(z, r) = -\log p(z) + \frac{1}{2} r^\top r.
$$

The first term acts like a potential energy, and the second acts like kinetic energy. The total Hamiltonian is approximately conserved under Hamiltonian dynamics governed by:

$$
\frac{dz}{dt} = r, \qquad
\frac{dr}{dt} = \nabla_z \log p(z).
$$

Following these dynamics moves the system along continuous trajectories that remain mostly in high-probability regions, allowing the sampler to travel long distances without being rejected.

  
#### Leapfrog Integration and Step Size

Exact Hamiltonian dynamics cannot be simulated analytically, so HMC uses a numerical integrator, typically the leapfrog method. Leapfrog integration updates position and momentum in small steps of size $\epsilon$:

1. half-step momentum update  
   $$
   r_{t+\tfrac{1}{2}} = r_t + \frac{\epsilon}{2}\,\nabla_z \log p(z_t)
   $$

2. full-step position update  
   $$
   z_{t+1} = z_t + \epsilon\, r_{t+\tfrac{1}{2}}
   $$

3. half-step momentum update  
   $$
   r_{t+1} = r_{t+\tfrac{1}{2}} + \frac{\epsilon}{2}\,\nabla_z \log p(z_{t+1})
   $$

These updates are repeated $L$ times, producing a proposal $(z', r')$ after a simulated trajectory of length $L \epsilon$.

The step size $\epsilon$ strongly influences performance:

- if $\epsilon$ is too large, numerical integration becomes inaccurate and proposals are rejected  
- if $\epsilon$ is too small, trajectories progress slowly and exploration becomes inefficient  

Adaptive schemes such as dual averaging automatically tune $\epsilon$.


#### Acceptance Step

Although leapfrog integration nearly preserves energy, numerical error accumulates. Therefore HMC applies a Metropolis acceptance step:

$$
\alpha = \min\left(1,\,
\exp\big(-H(z',r') + H(z,r)\big)
\right).
$$

Because leapfrog integration is reversible and volume-preserving, acceptance rates remain high even for long trajectories.

 
#### Choosing the Trajectory Length

The number of leapfrog steps $L$ (or total integration time $L\epsilon$) affects how far the sampler travels:

- small $L$ results in short trajectories, similar to random-walk proposals  
- large $L$ explores further but may waste computation or return near the starting point  

The No-U-Turn Sampler (NUTS) automatically selects an appropriate integration length and forms the basis of modern HMC implementations such as Stan.

 
#### Summary of Advantages
Hamiltonian Monte Carlo reduces random-walk behaviour by introducing an auxiliary momentum variable and using Hamiltonian dynamics to propose transitions. Instead of taking small, diffusive steps, HMC simulates the motion of a particle moving through the probability landscape.

Hamiltonian Monte Carlo offers several advantages:

- efficient exploration in high-dimensional or correlated distributions  
- large, directed moves that avoid random-walk behaviour  
- low autocorrelation between samples  
- high acceptance rates due to approximate energy conservation  
- uses gradients of $\log p(z)$ to guide proposals  

HMC is widely used in probabilistic programming frameworks such as Stan, PyMC, and NumPyro, largely because of its scalability and efficiency in challenging Bayesian inference problems.


 
### 5.2 Overrelaxation

Overrelaxation modifies proposals so that successive samples are negatively correlated. Instead of randomly perturbing the current state, overrelaxation proposes a point on the opposite side of the conditional mean.

Intuitively, if the current sample lies above the mean, the overrelaxation proposal nudges it below the mean, and vice versa. This helps the chain avoid local sticking and oscillation.

Overrelaxation is most effective when the conditional distribution is approximately Gaussian or when the model exhibits strong linear structure.

 
## 6. Sensitivity to Step Size

The efficiency of MCMC algorithms depends critically on the choice of step size (or proposal scale):

- If the step size is too small, the chain takes tiny moves and mixes slowly.  
- If the step size is too large, most proposals are rejected.

Finding an appropriate step size is essential for balancing exploration and acceptance.  

For random-walk Metropolis–Hastings:

- acceptance rates near 0.2–0.4 often work well in high dimensions  
- smaller dimensions tolerate larger acceptance rates

For HMC, step size affects both the numerical integration quality and the acceptance probability. Too large a step size causes integration error and rejections; too small a step size results in slow exploration.

Adaptive MCMC methods automatically tune the step size to achieve target acceptance rates.

 
## 7. When to Stop: Convergence and Diagnostics

Running an MCMC chain forever is impossible, so practical inference requires diagnosing convergence.

Several indicators are commonly used:

### 7.1 Burn-in

The initial part of the chain (the burn-in period) may not represent the target distribution. These early samples are discarded until the chain reaches a stable region.

### 7.2 Autocorrelation

High autocorrelation indicates slow mixing. Effective sample size (ESS) measures the number of independent samples equivalent to the correlated MCMC draws.

### 7.3 Multiple chains

Running several independent chains allows comparison. If chains converge to the same region, the sampler is more likely to have reached equilibrium.

### 7.4 Gelman–Rubin statistic (R-hat)

R-hat compares within-chain and between-chain variance. Values close to 1 indicate convergence.

### 7.5 Visual inspection

Trace plots, autocorrelation plots, and histograms provide qualitative insight into mixing and stability.

There is no single perfect test, but combining multiple diagnostics provides reasonable confidence that the Markov chain has approximated the target distribution.
 

--- 
Sampling methods approximate expectations and posterior distributions when closed-form solutions are unavailable. Independent methods such as importance and rejection sampling are simple but limited. Monte Carlo estimation provides a general framework for approximating integrals, and MCMC allows sampling from complex, high-dimensional distributions by constructing Markov chains. Advanced methods such as Hamiltonian Monte Carlo improve mixing and efficiency.

Sampling is a central tool for Bayesian inference and underlies many modern machine learning models, from deep generative architectures to reinforcement learning algorithms.

Random-walk behaviour limits the efficiency of basic MCMC methods. Hamiltonian Monte Carlo reduces this by exploiting gradient information and simulating Hamiltonian dynamics, while overrelaxation introduces negative correlation to speed up mixing. Step size must be chosen carefully to balance exploration and acceptance. Convergence diagnostics such as burn-in, effective sample size, and R-hat help determine when to stop sampling and assess the quality of the generated samples.
