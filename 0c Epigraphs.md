# Epigraphs and Convex Optimization

## 1. Definition of an Epigraph

For a function $f : \mathbb{R}^n \to \mathbb{R}$, the **epigraph** is the set of points lying **on or above** its graph:

$$
\operatorname{epi}(f) = \{ (x, t) \in \mathbb{R}^n \times \mathbb{R} \;\mid\; f(x) \le t \}.
$$

- $(x, t)$ is a point in $(n+1)$-dimensional space.  
- For each $x$, the condition $f(x) \le t$ means $t$ is **at or above** the function value.  

 
## 2. Intuition

- If you draw a 2D function $f(x)$, the **epigraph** is the region **above the curve**.  
- For example, if $f(x) = x^2$, then the epigraph is everything above the parabola.  

So:  

- **Graph** = the curve itself.  
- **Epigraph** = the curve + everything above it.  



## 3. Convexity via Epigraphs

A function $f$ is **convex** if and only if its **epigraph is a convex set**.  

- A set is convex if the line segment between any two points in the set lies entirely within the set.  
- Geometrically: the "roof" (epigraph) above the function must form a **bowl-shaped region**, not a cave.  



## 4. Examples

1. **Convex function**: $f(x) = x^2$  
   - Epigraph = everything above the parabola.  
   - This region is convex: if you connect any two points above the parabola, the line stays above the parabola.  
   - ⇒ $f(x)$ is convex.

2. **Non-convex function**: $f(x) = -x^2$  
   - Epigraph = everything above an upside-down parabola.  
   - This region is **not convex**: connecting two points above the parabola can dip below it.  
   - ⇒ $f(x)$ is not convex.


## 5. Why Epigraphs Matter in Optimization

Many optimization problems can be written in **epigraph form**:

$$
\min_x f(x) \quad \equiv \quad 
\min_{x,t} \; t \quad \text{s.t. } f(x) \le t.
$$

- We "lift" the problem into one extra dimension.  
- The feasible region is the **epigraph** of $f$.  
- Optimization over convex sets (epigraphs) is much more tractable.



### ✅ Summary

- The **epigraph** of a function is the region above its graph.  
- A function is **convex iff its epigraph is convex**.  
- Epigraphs let us reformulate optimization problems in a way that makes convexity clear and usable.  
