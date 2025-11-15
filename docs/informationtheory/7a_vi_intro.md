## Chapter — Optimization-Based Inference: MAP, EM, and the Path to Variational Inference

Monte Carlo methods provide a sampling-based approach to approximate expectations and posterior distributions. Although sampling is flexible and asymptotically exact, it can be computationally expensive, difficult to tune, or slow to converge in high dimensions. For many models, especially those involving latent variables or large datasets, it is more practical to replace sampling with optimization.

This chapter introduces three optimization-based inference strategies:

1. Maximum a posteriori (MAP) estimation  
2. Expectation–Maximization (EM)  
3. Variational inference (VI), in its simplest introductory form  

Together, these methods motivate the full treatment of variational inference in the following chapter.

---

## 1. Motivation for Optimization-Based Inference

Bayesian inference requires the posterior

$$
p(z|x) = \frac{p(x,z)}{p(x)}.
$$

The challenge lies in computing the marginal likelihood

$$
p(x) = \int p(x,z)\,dz,
$$

which is almost always intractable. Monte Carlo sampling approximates this integral using samples, but sampling may be slow or unreliable for:

- high-dimensional latent spaces  
- multimodal posteriors  
- large datasets  
- models requiring gradient-based learning  

This motivates an alternative strategy: instead of drawing samples, we can transform inference into an optimization problem.

---

## 2. Maximum A Posteriori (MAP) Estimation

MAP estimation finds the most likely value of a latent variable or parameter after observing the data. Starting from Bayes’ rule:

$$
p(\theta|x) = \frac{p(x|\theta)p(\theta)}{p(x)},
$$

MAP chooses the mode of the posterior:

$$
\theta_{\text{MAP}} 
= \arg\max_\theta p(\theta|x).
$$

Since $p(x)$ does not depend on $\theta$, this is equivalent to:

$$
\theta_{\text{MAP}} 
= \arg\max_\theta \big[ \log p(x|\theta) + \log p(\theta) \big].
$$

MAP is efficient and easy to compute. It reduces inference to optimization and incorporates prior knowledge through $p(\theta)$. However, it returns only a point estimate and does not capture uncertainty.

MAP is thus a limited but useful form of Bayesian inference, often interpreted as maximum likelihood augmented with a regularization term.

---

## 3. Expectation–Maximization (EM)

EM is designed for models with latent variables. The log-likelihood of the observed data is:

$$
\log p_\theta(x) 
= \log \sum_z p_\theta(x,z).
$$

Direct optimization is difficult because of the sum over latent variables. EM solves this using two alternating steps:

### E-step  
Compute the posterior over latent variables under the current parameters:

$$
q(z) = p_\theta(z|x).
$$

### M-step  
Maximize the expected complete-data log-likelihood:

$$
\theta \leftarrow 
\arg\max_\theta 
\mathbb{E}_{q(z)}[\log p_\theta(x,z)].
$$

EM guarantees that the likelihood increases with each iteration. It is widely used in:

- mixture of Gaussians  
- hidden Markov models  
- probabilistic PCA  
- clustering and density estimation  

EM can be interpreted as a form of variational inference where the variational distribution is constrained to be the exact posterior $q(z) = p_\theta(z|x)$.

---

## 4. EM and MAP: MAP-EM

EM typically performs maximum likelihood estimation, but it can be modified to perform MAP estimation by including a prior:

$$
\theta_{\text{MAP}} 
= 
\arg\max_\theta 
\left[
\mathbb{E}_{p(z|x,\theta)}[\log p(x,z|\theta)]
+ \log p(\theta)
\right].
$$

This version, often called MAP-EM, incorporates prior structure into the estimation procedure.

---

## 5. Limitations of MAP and EM

Both MAP and EM have limitations that motivate more general methods:

1. MAP returns only a point estimate and discards posterior uncertainty.  
2. EM requires exact posterior computation in the E-step:
   $$
   q(z) = p_\theta(z|x),
   $$
   which is often intractable.  
3. EM struggles with:
   - multimodal posteriors  
   - high-dimensional latent spaces  
   - arbitrary likelihood forms  

These limitations lead naturally to variational inference.

---

## 6. A Brief Introduction to Variational Inference (VI)

Variational inference generalizes EM by replacing the exact posterior with a tractable approximation. Instead of requiring

$$
q(z) = p_\theta(z|x),
$$

VI chooses a family of distributions

$$
q_\phi(z|x) \in \mathcal{Q}
$$

and optimizes it to be close to the true posterior. The objective is:

$$
\phi^* = 
\arg\min_\phi D_{\text{KL}}(q_\phi(z|x)\|p(z|x)).
$$

Because $p(z|x)$ contains the intractable marginal likelihood, VI rewrites this using the Evidence Lower Bound (ELBO):

$$
\log p(x)
=
\mathcal{L}(x;\phi,\theta)
+
D_{\text{KL}}(q_\phi(z|x)\|p(z|x)).
$$

Maximizing the ELBO yields a tractable approximation to Bayesian posterior inference.

VI:

- generalizes MAP (when $q$ is a delta function)  
- generalizes EM (when $q = p_\theta(z|x)$)  
- supports flexible approximations  
- scales to large datasets  
- is the backbone of VAEs, Bayesian deep models, and many modern generative models  

The next chapter explores variational inference in detail.

---

## 7. Summary of the Chapter

Monte Carlo sampling approximates integrals using random samples, but can be slow or difficult to tune. Optimization-based inference provides an alternative strategy.

MAP estimation chooses the most likely parameter value given the data and the prior. EM handles models with latent variables by alternating between inference (E-step) and optimization (M-step). Variational inference generalizes EM by allowing the E-step to use tractable approximations rather than the exact posterior.

MAP, EM, and variational inference all represent the shift from sampling-based methods toward optimization-based approaches. These methods form the conceptual foundation for the next chapter on full variational inference and the ELBO.

