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

Importance sampling is widely used for:

- likelihood estimation  
- off-policy reinforcement learning  
- correcting distribution mismatch  

It suffers when $p(z)/q(z)$ has high variance.

 
### 2.3 Rejection Sampling

Rejection sampling uses a proposal distribution $q(z)$ and a constant $M$ such that

$$
p(z) \le M q(z) \quad \text{for all } z.
$$

Procedure:

1. sample $z \sim q(z)$  
2. accept with probability $\frac{p(z)}{M q(z)}$  

It produces exact samples from $p(z)$, but can be extremely inefficient in high dimensions.

 
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

Given the current state $z$, propose $z'$ using a proposal distribution $q(z'|z)$.  
Accept with probability:

$$
\alpha = \min\left(1, 
\frac{p(z')\,q(z|z')}
     {p(z)\,q(z'|z)}
\right).
$$

If accepted, set $z_{t+1} = z'$, otherwise keep $z_{t+1} = z$.

Metropolis–Hastings forms the foundation for most MCMC methods.

 

### 4.2 Gibbs Sampling

Gibbs sampling updates one variable at a time by sampling from its conditional distribution:

$$
z_i \sim p(z_i \mid z_{-i}).
$$

This requires all conditionals to be tractable.

Applications include:

- topic models (LDA)  
- hidden Markov models  
- Bayesian networks  

 
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
