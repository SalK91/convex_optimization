# Appendix E : Convexity in Probability and Statistics

Convex analysis is not just geometry and optimization — it is deeply woven into probability, statistics, and information theory.  
Many statistical models, estimators, and loss functions are convex because convexity guarantees stability, uniqueness, and tractability of inference.

This appendix surveys how convexity arises naturally in probabilistic and statistical contexts.

 
## E.1 Convexity of Expectations

Let \(f:\mathbb{R}^n\!\to\!\mathbb{R}\) be convex and \(X\) a random vector.  
Then by Jensen’s inequality (Appendix A):

\[
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)].
\]

### Consequences
- Expectations preserve convexity:  
  if each \(f(\cdot,\xi)\) is convex, then \(F(x)=\mathbb{E}_\xi[f(x,\xi)]\) is convex.
- Stochastic objectives in ML — e.g. expected loss \(\mathbb{E}_{(a,b)}[\ell(a^\top x,b)]\) — are convex when the sample-wise loss is convex.

Hence almost all *empirical risk minimization* problems are discrete approximations of convex expectations.

 
## E.2 Convexity of Log-Partition and Moment-Generating Functions

For a random variable \(X\), the moment-generating function (MGF) and cumulant-generating function (CGF) are

\[
M_X(t)=\mathbb{E}[e^{tX}], \qquad
K_X(t)=\log M_X(t).
\]

Fact: \(K_X(t)\) is always convex in \(t\).

Reason: \(K_X''(t)=\mathrm{Var}_t(X)\ge0\);  
variance is nonnegative.  

### Implications
- \(K_X(t)\) acts as a convex “potential” controlling exponential families.
- The log-partition function in statistics,
  \[
  A(\theta)=\log \int e^{\langle \theta,T(x)\rangle}\,h(x)\,dx,
  \]
  is convex in \(\theta\) (strictly convex for full exponential families).
- Its gradient gives the mean parameter: \(\nabla A(\theta)=\mathbb{E}_\theta[T(X)]\).

Thus convexity of \(A\) guarantees a one-to-one mapping between natural and mean parameters — a foundation of exponential-family inference.

## E.3 Exponential Families and Dual Convexity

An exponential-family density has the form
\[
p_\theta(x)=\exp\big(\langle\theta,T(x)\rangle-A(\theta)\big)h(x).
\]

Properties:

1. \(A(\theta)\) is convex, smooth, and serves as a potential function.
2. Its convex conjugate \(A^*(\mu)\) defines the entropy of the family:
   \[
   A^*(\mu)=\sup_\theta(\langle\mu,\theta\rangle-A(\theta)) = -H(p_\mu),
   \]
   where \(H\) is the Shannon entropy of the distribution with mean \(\mu\).

Hence maximum-likelihood estimation in exponential families is a convex optimization problem, and maximum-entropy estimation is its Fenchel dual.

 

## E.4 Convex Divergences and Information Measures

### (a) Kullback–Leibler (KL) Divergence
For densities \(p,q\),
\[
D_{\mathrm{KL}}(p\|q)=\int p(x)\log\frac{p(x)}{q(x)}\,dx.
\]

- \(D_{\mathrm{KL}}\) is jointly convex in \((p,q)\).  
- Proof: the function \((u,v)\mapsto u\log(u/v)\) is convex on \(\mathbb{R}_+^2\).  
- Consequently, mixtures of distributions cannot increase KL divergence — a key fact in variational inference and EM.

### (b) Bregman Divergences
Given a differentiable convex \(\phi\), define
\[
D_\phi(x\|y)=\phi(x)-\phi(y)-\langle\nabla\phi(y),x-y\rangle.
\]
KL divergence is a Bregman divergence for \(\phi(p)=\sum_i p_i\log p_i\).  
Thus information-theoretic distances are *geometric shadows* of convex functions.

### (c) f-Divergences
A general convex generator \(f\) with \(f(1)=0\) yields
\[
D_f(p\|q)=\int q(x)\,f\!\left(\frac{p(x)}{q(x)}\right)dx.
\]
Convexity of \(f\) ⇒ convexity of \(D_f\).  
Common choices recover KL, χ², Hellinger, and Jensen–Shannon divergences.

 

## E.5 Convex Loss Functions in Statistics and Machine Learning

Convexity ensures estimators are globally optimal and algorithms converge.

| Setting | Loss / Negative Log-Likelihood | Convexity |
|----------|--------------------------------|------------|
| Gaussian noise | \(\tfrac12\|Ax-b\|_2^2\) | quadratic, strongly convex |
| Laplace noise | \(\|Ax-b\|_1\) | convex, nonsmooth |
| Logistic regression | \(\log(1+e^{-y a^\top x})\) | convex, smooth |
| Poisson regression | \(e^{a^\top x}-y a^\top x\) | convex, exponential |
| Huber loss | piecewise quadratic/linear | convex, robust |

Convexity of the negative log-likelihood follows from convexity of the log-partition function \(A(\theta)\) in exponential families.

 

## E.6 Convexity and Bayesian Inference

In Bayesian inference, convexity appears in:

- Log-concave posteriors:  
  If the likelihood and prior are log-concave, the posterior \(p(x|y)\propto \exp(-f(x))\) is also log-concave ⇒  
  \(\log p(x|y)\) concave, \(f(x)\) convex.

- MAP estimation:  
  Maximizing \(\log p(x|y)\) ≡ minimizing a convex function when \(p(x|y)\) is log-concave ⇒ global optimum guaranteed.

- Variational inference:  
  The ELBO is a concave function of the variational parameters because it is a linear minus KL divergence (convex).  
  Optimizing it is equivalent to minimizing a convex divergence.

Thus convexity guarantees stable Bayesian updates and efficient approximate inference.

 

## E.7 Statistical Risk and Convex Surrogates

Convex surrogate losses replace nonconvex 0–1 loss with convex approximations:

- Hinge loss (\(\max(0,1-y a^\top x)\)) → support-vector machines.  
- Logistic loss → probabilistic classification (cross-entropy).  
- Exponential loss → AdaBoost.

These convex surrogates retain calibration (minimizing expected convex loss yields correct decision boundaries) while enabling tractable optimization.

