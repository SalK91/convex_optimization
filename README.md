https://www.youtube.com/watch?v=d2jF3SXcFQ8
    1. https://web.stanford.edu/class/ee364a/lectures.html
https://www.youtube.com/watch?v=N3V2AdTImJE&list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h&index=10


https://www.youtube.com/watch?v=trs0RI39uWI

https://www.youtube.com/watch?v=t8sp9rv-Dqo


### Current state

1. Stanford Lecture Stanford EE364A Convex Optimization
   https://www.youtube.com/watch?v=zVO4ccvrwrI&list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h&index=15

   30min

   Slide 355

    
   https://www.youtube.com/watch?v=U3lJAObbMFI&list=PLGrZHfnLv54ksX-0TMKYPCZDx9NZyYh9c&index=1

   https://bloomberg.github.io/foml/#home

   https://www.youtube.com/playlist?list=PLZSO_6-bSqHQHBCoGaObUljoXAyyqhpFW

  pyOD
  earh mover distance
  branch and bound
  variational inferences?

2. MIT 6.824: Distributed Systems:
   https://www.youtube.com/@6.824

2. LLM
   https://www.youtube.com/watch?v=Ub3GoFaUcds
3. MCP

4. Non Convex Articel

5. RL Articel

6. LLMs
   https://github.com/aishwaryanr/awesome-generative-ai-guide

7. Amazon

8. Pytorch.

9. Autograd lecture my Andre Kapathry


---------------------

1. LP limitation and scalability
2. Why postivie semi deifniate matrix

# Convex Optimization and Case Studies

## Overview
This repository is a curated collection of **concepts**, **algorithms**, and **case studies** in **convex optimization** — a unifying framework that sits at the intersection of applied mathematics, computer science, and engineering.

We focus on:
- **Theoretical foundations** — understanding what makes a problem convex and why convexity matters.
- **Practical algorithms** — from classical methods like simplex and gradient descent to modern interior-point and first-order methods.
- **Real-world case studies** — demonstrating convex optimization in machine learning, control, finance, and beyond.


---- 

### Why Convex Optimization?
Convex optimization problems are those where the **objective function** is convex and the **feasible set** (set of points satisfying all constraints) is also convex.  
This structure gives us three remarkable advantages:

1. **Global optimality**  
   - Any local minimum is automatically a global minimum.
2. **Strong theory**  
   - Tools like *duality theory*, *optimality conditions*, and *sensitivity analysis* work elegantly.
3. **Algorithmic efficiency**  
   - Many convex problems can be solved to high precision in polynomial time, even at large scale.

This makes convex optimization a **cornerstone** in:
- Machine learning and AI
- Control systems
- Signal processing
- Operations research
- Finance and portfolio optimization
- Supply chain and logistics
- Resource allocation problems



---

### Common Standard Forms
Many convex optimization problems can be expressed in **standard forms**:

1. **Linear Program (LP)**  
   $$
   \min_x \; c^T x \quad
   \text{s.t.} \; Ax \leq b, \; Ex = d
   $$
   - $f_0$ is linear.
   - Feasible set is a polyhedron.

2. **Quadratic Program (QP)**  
   $$
   \min_x \; \frac{1}{2} x^T Q x + c^T x \quad
   \text{s.t.} \; Ax \leq b
   $$
   - $Q \succeq 0$ ensures convexity.

3. **Second-Order Cone Program (SOCP)**  
   $$
   \|A_i x + b_i\|_2 \leq c_i^T x + d_i
   $$
   - Captures problems with norm constraints.

4. **Semidefinite Program (SDP)**  
   $$
   \min_X \; \text{tr}(CX) \quad
   \text{s.t.} \; \mathcal{A}(X) = b, \; X \succeq 0
   $$
   - Variable is a positive semidefinite matrix.



### Geometric Interpretation
- The **objective function** shapes the “height” of the landscape.
- The **constraints** carve out the feasible region.
- In convex problems, the feasible region is **bowl-like or flat-faced**, so the global minimum lies where the lowest contour of the objective touches the region.
