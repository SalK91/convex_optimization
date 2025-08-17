## Huber Penalty Loss

The **Huber loss** is a robust loss function that combines the advantages of **squared loss** and **absolute loss**, making it less sensitive to outliers while remaining convex. It is defined as:

$$
L_\delta(r) = 
\begin{cases} 
\frac{1}{2} r^2 & \text{if } |r| \le \delta, \\
\delta (|r| - \frac{1}{2}\delta) & \text{if } |r| > \delta,
\end{cases}
$$

where $r = y - \hat{y}$ is the residual, and $\delta > 0$ is a threshold parameter.  

### Key Properties
- Quadratic for small residuals ($|r| \le \delta$) → behaves like least squares.  
- Linear for large residuals ($|r| > \delta$) → reduces the influence of outliers.  
- Convex, so standard convex optimization techniques apply.  

### Use
- Commonly used in **robust regression** to estimate parameters in the presence of outliers.
- Balances **efficiency** (like least squares) and **robustness** (like absolute loss).
