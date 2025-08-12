# Galactic Cargo Delivery Optimization — LP Formulation

You are the logistics commander of an interstellar fleet tasked with delivering vital supplies across the galaxy. Your fleet consists of **$N$ starship pilots**, and you have **$K$ distinct types of cargo crates** to deliver. Each cargo type $j$ has a known volume $v_j$, representing the number of crates that must reach their destinations.

To maintain fleet balance and operational efficiency, **each pilot must carry the same total number of crates**. Your mission is to assign crates to pilots to **minimize the total expected delivery time**, accounting for each pilot’s unique speed and proficiency with different cargo types.


### Notation

- $i = 1, \ldots, N$: indices for starship pilots  
- $j = 1, \ldots, K$: indices for cargo types  
- $v_j$: volume (number of crates) of cargo type $j$  
- $d_{ij}$: estimated delivery time for pilot $i$ to deliver one crate of type $j$



### Decision Variables

$x_{ij} \geq 0$

Number of crates of cargo type $j$ assigned to pilot $i$.

### Objective Function

Minimize the total delivery time across all pilots:

$$\min \sum_{i=1}^N \sum_{j=1}^K d_{ij} x_{ij}$$

Or equivalently, in vector form:

$$\min c^T x$$

where

$$
c = \begin{bmatrix} d_{11}, d_{12}, \ldots, d_{1K}, d_{21}, \ldots, d_{NK} \end{bmatrix}^T,
\quad
x = \begin{bmatrix} x_{11}, x_{12}, \ldots, x_{1K}, x_{21}, \ldots, x_{NK} \end{bmatrix}^T
$$


### Constraints

1. **All crates must be delivered:**

$$\sum_{i=1}^N x_{ij} = v_j, \quad \forall j = 1, \ldots, K$$

2. **Each pilot carries the same total number of crates:**


$$\sum_{j=1}^K x_{ij} = \frac{V}{N}, \quad \forall i = 1, \ldots, N
\quad \text{where} \quad V = \sum_{j=1}^K v_j$$

3. **Non-negativity:**
$$x_{ij} \geq 0, \quad \forall i,j$$



## LP Formulation

$$
\begin{aligned}
\min_{x \in \mathbb{R}^{N \times K}} \quad & c^T x \\
\text{subject to} \quad &
\begin{cases}
A_{eq} x = b_{eq} \\
x \geq 0
\end{cases}
\end{aligned}$$

Where:

- $A_{eq} \in \mathbb{R}^{(N + K) \times (N \cdot K)}$ encodes the equality constraints for cargo delivery and load balancing.
- $b_{eq} \in \mathbb{R}^{N + K}$ combines the crate volumes $v_j$ and equal load targets $\frac{V}{N}$.

