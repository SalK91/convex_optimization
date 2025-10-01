# Optimization:  Algos

Optimization is the **core of machine learning**: training a model is an optimization problem where we search for parameters $\theta$ that **minimize a loss or maximize a likelihood**. The choice of optimization algorithm depends on **problem type, scale, constraints, smoothness, and stochasticity**. 


## Gradient descent

We want to minimize a function over a feasible set:

$$
\min_{x \in \mathcal{X}} f(x).
$$

At iteration $t$, given the current point $x_t$, we approximate $f$ around $x_t$ using a first-order Taylor expansion plus a quadratic regularization term. The square term penalizes moving too far from the current point $x_t$. You can think of it as saying: “I trust my first-order approximation, but only locally. If I move too far, the approximation might be bad — so I add a cost for large steps.”
 
$$
f(x) \approx f(x_t) + \langle \nabla f(x_t), x - x_t \rangle + \frac{1}{2\eta}\|x - x_t\|^2,
$$

where $\eta > 0$ is the step size (also called the learning rate).

Thus, the update rule is defined as the solution of:

$$
x_{t+1} = \arg\min_{x \in \mathcal{X}} \Big[ f(x_t) + \langle \nabla f(x_t), x - x_t \rangle + \tfrac{1}{2\eta} \|x - x_t\|^2 \Big].
$$

To find $x_{t+1}$, take the derivative of the objective inside the brackets with respect to $x$:

$$
\nabla f(x_t) + \frac{1}{\eta}(x - x_t) = 0.
$$

Rearranging gives:

$$
x - x_t = -\eta \nabla f(x_t).
$$

Therefore, the update rule is:

$$
x_{t+1} = x_t - \eta \nabla f(x_t).
$$
- Under smoothness and strong convexity, **gradient descent converges linearly**, improving by a fixed fraction at each iteration.




## Subgradient Method
We want to minimize a convex (possibly nonsmooth) function:

$$
\min_{x \in \mathcal{X}} f(x),
$$

where $f$ is convex but may not be differentiable everywhere.


At iteration $t$, we compute a **subgradient** $g_t \in \partial f(x_t)$, where $\partial f(x_t)$ is the set of subgradients at $x_t$.

The update rule is:

$$
x_{t+1} = \Pi_{\mathcal{X}} \left( x_t - \eta_t g_t \right),
$$

where:
- $g_t$ is a subgradient of $f$ at $x_t$,
- $\eta_t > 0$ is the step size (which may change with $t$),
- $\Pi_{\mathcal{X}}$ is the projection onto the feasible set $\mathcal{X}$.

If $\mathcal{X} = \mathbb{R}^n$, the update reduces to:

$$
x_{t+1} = x_t - \eta_t g_t.
$$


Unlike gradient descent, the subgradient method does **not** achieve linear convergence. Instead:

- If $f$ is convex (not strongly convex), with a **diminishing step size** such as $\eta_t = \frac{c}{\sqrt{t}}$, we obtain a **sublinear convergence rate**:

$$
f(\bar{x}_T) - f(x^\star) \leq \frac{R \, G}{\sqrt{T}},
$$

where:
- $x^\star$ is an optimal solution,
- $\bar{x}_T = \frac{1}{T}\sum_{t=1}^T x_t$ is the average iterate,
- $R = \|x_0 - x^\star\|$ is the distance from the initial point to the optimum,
- $G$ is a bound on the subgradients ($\|g_t\| \leq G$).

Thus the convergence rate is:

$$
O\!\left(\frac{1}{\sqrt{T}}\right).
$$


## Accelerated Gradient Descent: Momentum
The standard gradient descent update is:

$x_{t+1} = x_t - \eta \nabla f(x_t)$

where:  
- $x_t$ is the current point  
- $\eta$ is the step size (learning rate)  
- $\nabla f(x_t)$ is the gradient at $x_t$  

Intuition:  
Imagine rolling a ball on a hill: the ball moves in the steepest downhill direction at each step. This works, but can be slow if the valley is long and narrow: the ball zig-zags and takes many small steps to reach the bottom.


Momentum adds the idea of inertia:

$v_t = x_t - x_{t-1}$

- $v_t$ represents the velocity of the ball  
- Instead of reacting only to the current slope, the ball remembers its previous speed and direction, helping it roll faster through shallow or consistent slopes  
- Momentum reduces zig-zagging in steep valleys and accelerates movement along flat directions

Analogy:  
- No momentum → ball stops after each step, carefully following the slope  
- With momentum → ball keeps moving, building speed along the valley, only slowing when gradients push against it


## Gradient Descent + Momentum

The update rule with momentum:

$x_{t+1} = x_t - \eta \nabla f(x_t) + \beta (x_t - x_{t-1})$

where:  
- $\beta \in [0,1)$ controls how much past velocity is retained  

Breakdown:  
1. Gradient step: $-\eta \nabla f(x_t)$ → move downhill  
2. Momentum step: $\beta (x_t - x_{t-1})$ → continue moving along previous direction  

Intuition:  
- If gradients keep pointing roughly in the same direction, momentum accelerates the steps  
- If gradients oscillate, momentum smooths the path, reducing overshooting


## Convergence Intuition

- For convex and smooth functions, momentum accelerates convergence:  
  - Standard GD: $O(1/t)$  
  - GD + Momentum / Nesterov: $O(1/t^2)$  

- Each step combines the current slope and accumulated speed from past steps  
- Momentum acts like a rolling ball in a frictionless valley: the more it rolls in the right direction, the faster it reaches the minimum



### Key Takeaways

1. Momentum is memory: it remembers the direction of previous steps  
2. It smooths oscillations in narrow valleys  
3. It accelerates convergence along consistent gradient directions  
4. Hyperparameter $\beta$ controls the inertia: higher $\beta$ → longer memory, faster but riskier steps
