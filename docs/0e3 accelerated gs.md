# Accelerated Gradient Descent: Momentum

The standard **gradient descent (GD)** update is:

$$
x_{t+1} = x_t - \eta \nabla f(x_t),
$$

where:  
- $x_t$ is the current iterate,  
- $\eta > 0$ is the step size (learning rate),  
- $\nabla f(x_t)$ is the gradient at $x_t$.

> **Intuition**: Imagine rolling a ball on a hill. The ball moves in the steepest downhill direction at each step.  
> In long, narrow valleys, standard GD can zig-zag, taking many small steps to reach the bottom.

---

## 1. Momentum: Adding Inertia

Momentum adds the idea of **velocity**, allowing the optimization to "remember" previous directions:

$$
v_t = x_t - x_{t-1},
$$

where $v_t$ represents the **velocity** of the iterate.  

- The update now combines the current gradient and the previous motion.  
- Momentum helps move faster along flat or consistent slopes and reduces zig-zagging in steep valleys.  

> **Analogy**:  
> - No momentum → ball stops after each step, carefully following the slope.  
> - With momentum → ball keeps rolling, building speed along the valley, only slowing when gradients push against it.

---

## 2. Gradient Descent with Momentum

The **update rule with momentum** is:

$$
x_{t+1} = x_t - \eta \nabla f(x_t) + \beta (x_t - x_{t-1}),
$$

where:  
- $\beta \in [0,1)$ is the **momentum parameter**, controlling how much past velocity is retained.

### Breakdown

1. **Gradient step**: $-\eta \nabla f(x_t)$ → moves downhill.  
2. **Momentum step**: $\beta (x_t - x_{t-1})$ → continues moving along previous direction.

> **Intuition**:  
> - If gradients consistently point in the same direction, momentum accelerates the steps.  
> - If gradients oscillate, momentum smooths the path, reducing overshooting.

---

## 3. Alternative Form: Velocity Update

Another common formulation introduces an explicit velocity variable $v_t$:

$$
\begin{aligned}
v_{t+1} &= \beta v_t - \eta \nabla f(x_t) \\
x_{t+1} &= x_t + v_{t+1}
\end{aligned}
$$

- Here, $v_t$ accumulates the past updates weighted by $\beta$.  
- This makes the analogy to a **rolling ball** more explicit.

---

## 4. Convergence Intuition

- For **convex and smooth functions**, momentum accelerates convergence:  
  - Standard GD: $O(1/t)$  
  - GD + Momentum / Nesterov: $O(1/t^2)$

- Momentum combines **current slope** and **accumulated speed from past steps**.  
- Acts like a **frictionless ball in a valley**: keeps moving in the right direction, accelerating convergence.

> Key idea: Momentum builds up speed along consistent gradient directions and smooths oscillations along steep valleys.

---

## 5. Practical Remarks

1. **Momentum is memory**: it remembers the direction of previous steps.  
2. **Reduces oscillations** in narrow valleys.  
3. **Accelerates convergence** along consistent gradient directions.  
4. **Hyperparameter $\beta$** controls inertia:  
   - Higher $\beta$ → longer memory, faster but potentially riskier steps.  
   - Typical values: $\beta = 0.9$ or $0.99$.  
5. Can be combined with **Nesterov acceleration** for theoretically optimal rates.

---

## 6. Summary

Momentum modifies gradient descent by combining:

- Immediate gradient information ($-\eta \nabla f(x_t)$)  
- Past velocity ($\beta (x_t - x_{t-1})$)  

Effectively, it allows the optimizer to **roll through valleys faster**, reduce zig-zagging, and achieve **accelerated convergence**, especially for convex and smooth functions.

---
