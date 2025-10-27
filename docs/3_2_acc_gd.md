Gradient descent’s $O(1/k)$ convergence on smooth convex problems is suboptimal: there are algorithms that attain faster rates. Accelerated gradient methods, pioneered by Yurii Nesterov, achieve an optimal $O(1/k^2)$ convergence rate for first-order convex optimization. The core idea is to incorporate momentum — using information from past iterations to build up speed in the right direction.

**Momentum Intuition: Heavy Ball Analogy**
Imagine rolling a ball down a convex hill. Standard gradient descent is like a heavy ball with extremely high friction: at each step, you come to a stop and then roll a tiny bit further based purely on the current slope. In narrow valleys, this leads to a slow, zig-zag descent because the ball is constantly stopping and re-accelerating down the sides of the valley. Momentum provides the ball with inertia, so it keeps moving in the previous direction even as it responds to the current slope. This can smooth out oscillations and traverse flat regions faster.

In practical terms, momentum means we introduce a velocity term $v_k$ that accumulates past gradients. A basic gradient descent with momentum update is:

$$
\begin{aligned}
v_{k+1} &= \beta v_k - \alpha \nabla f(x_k), \\
x_{k+1} &= x_k + v_{k+1}.
\end{aligned}
$$


where $\beta\in[0,1)$ is the momentum coefficient controlling how much of the past velocity $v_k$ to retain. When written fully in terms of positions $x$, this is equivalent to:


$$
x_{k+1} = x_k - \alpha \nabla f(x_k) + \beta (x_k - x_{k-1})
$$

so the step has two components: a usual gradient descent step $(-\alpha \nabla f(x_k))$ plus a push in the direction of the previous motion $(x_k - x_{k-1})$. If recent gradients have been pointing in roughly the same direction, the velocity term $\beta(x_k - x_{k-1})$ adds a significant boost, effectively increasing the step size in low-curvature directions. If the gradient direction changes (oscillations), the momentum term can partially cancel out the back-and-forth, preventing full oscillation.

Benefits: In convex problems with long narrow valleys (high condition number), momentum helps “fly through” the valley floor instead of tediously zig-zagging. It’s like smoothing the path by dampening orthogonal oscillations and amplifying movement aligned with consistent gradients. Empirically, momentum often yields faster decrease in objective value than vanilla gradient descent.

**Heavy-ball vs. Nesterov acceleration:** The above momentum scheme is often called Polyak’s heavy-ball method. Nesterov’s Accelerated Gradient (NAG) is a slight modification that delivers rigorous guarantees. In NAG, one first takes a step in the old velocity direction to a predicted intermediate point, then evaluates the gradient there:

​$$
\begin{aligned}
y_k &= x_k + \beta (x_k - x_{k-1}), \\
x_{k+1} &= y_k - \alpha \nabla f(y_k).
\end{aligned}
$$



This “lookahead” (evaluating $\nabla f$ at $y_k$) is the key difference – it anticipates where momentum is taking the iterate and then corrects the course with a gradient measured at that extrapolated point. The Nesterov update can be seen as adding momentum in a way that does not overshoot the optimum: by the time you apply the gradient, you’re already partway there, so you don’t “coast” blindly beyond the minimizer.

Remarkably, Nesterov proved that with a specific sequence of $\beta$ values, his method achieves

- $O(1/k^2)$ convergence in function value for convex, $L$-smooth $f$, and

- $O((1-\sqrt{\mu/L})^k)$ linear convergence for $\mu$-strongly convex, $L$-smooth $f$,

both of which are optimal in the sense that no first-order method can attain better worst-case rates in those scenarios. In other words, accelerated gradient descent (AGD/Nesterov’s method) is provably as fast as possible for large-scale convex optimization using only gradient information.

**Geometric perspective:** Momentum methods can be interpreted as a discrete approximation to a second-order ordinary differential equation of motion (like a damped oscillator). The heavy-ball method corresponds to a physical mass sliding with friction on the landscape $f(x)$. Nesterov’s method adds a clever adjustment akin to a lookahead that ensures stability and optimal speed. In continuous time, both can be seen as putting a $\ddot{x}$ (acceleration) term into the dynamic, which leads to faster approach to the minimum compared to the $\dot{x}$ (first-order) dynamic of plain gradient flow.

**Acceleration in Practice**

For practical implementation, one typically fixes $\beta \approx 0.9$ or adjusts it adaptively, and uses a fixed or decaying $\alpha$. Nesterov’s variant is only slightly more involved than heavy-ball and is often used by default in deep learning (many optimizers like Adam incorporate Nesterov momentum by computing gradients at a predicted step). The momentum parameter $\beta$ close to 1 gives long memory (the velocity is a long running average of gradients), which can greatly speed up convergence on well-behaved problems but might cause overshooting if the landscape changes abruptly. Empirically, momentum with $\beta\in[0.9,0.99]$ tends to work well, accelerating progress especially in the early stages of convex optimization.

It’s worth noting that heavy-ball momentum does not always guarantee faster convergence in theory (for general convex functions it can even diverge if mis-tuned), whereas Nesterov’s momentum comes with the aforementioned guarantees. For strongly convex quadratic objectives, heavy-ball momentum achieves the accelerated linear rate $O((1-\sqrt{\mu/L})^k)$ as well, but for non-quadratics Nesterov’s method is preferred due to its robustness.

Summary: Accelerated gradient methods introduce a memory of past gradients that propels iterates forward more forcefully than standard gradient descent. The result is a significant speed-up on convex problems – roughly, acceleration takes $O(\sqrt{\kappa}\log(1/\varepsilon))$ iterations instead of $O(\kappa\log(1/\varepsilon))$ for $\kappa$-conditioned problems. In intuitive terms, momentum helps the optimizer build velocity along directions of consistent descent and dampen oscillations in awkward directions. Nesterov’s Accelerated Gradient (AGD) refines this idea to guarantee the fastest possible convergence rate $O(1/k^2)$ for smooth convex minimization. This has made Nesterov’s method a cornerstone in both theoretical optimization and practical algorithms (it’s often the default choice when one needs faster convergence from gradient steps).