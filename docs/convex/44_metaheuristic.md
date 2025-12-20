# Chapter 21: Metaheuristic and Evolutionary Algorithms

When optimization problems are highly nonconvex, discrete, or black-box, deterministic methods often fail to find good solutions.  In these settings, metaheuristic algorithms—inspired by nature, biology, and collective behavior—provide robust and flexible alternatives.

Metaheuristics are general-purpose stochastic search methods that rely on repeated sampling, adaptation, and survival of the fittest ideas. They are especially effective when the landscape is rugged, multimodal, or not well understood.



## Principles of Metaheuristic Optimization

All metaheuristics share three key principles:

1. Population-Based Search:  
   Maintain multiple candidate solutions simultaneously to explore diverse regions of the search space.

2. Variation Operators:  
   Create new solutions via mutation, recombination, or stochastic perturbations.

3. Selection and Adaptation:  
   Favor candidates with better objective values, guiding the search toward promising regions.

Unlike local methods, metaheuristics balance exploration (global search) and exploitation (local refinement).


## Genetic Algorithms (GA)

### Biological Inspiration

Genetic Algorithms mimic natural evolution, where populations evolve toward higher fitness through selection, crossover, and mutation.

### Representation

A solution (individual) is represented as a chromosome—often a binary string, vector of reals, or permutation.  
Each position (gene) encodes part of the decision variable.

### Algorithm Outline

1. Initialize a population $\{x_i\}_{i=1}^N$ randomly.  
2. Evaluate fitness $f(x_i)$ for all individuals.  
3. Select parents based on fitness (e.g., tournament or roulette-wheel selection).  
4. Apply:

     - Crossover: combine genetic material of two parents.  
     - Mutation: randomly alter some genes to maintain diversity.  

5. Form a new population and repeat until convergence.

### Crossover and Mutation Examples
- Single-point crossover: exchange genes after a random index.  
- Gaussian mutation: add small noise to continuous parameters.  

### Strengths and Weaknesses
| Strengths | Weaknesses |
|----------------|----------------|
| Highly parallel, robust, domain-independent | Requires many function evaluations |
| Effective for combinatorial and discrete optimization | Parameter tuning (mutation, crossover rates) is nontrivial |

 
## Differential Evolution (DE)

Differential Evolution is a simple yet powerful algorithm for continuous optimization.

### Core Idea
Mutation is performed using differences of population members:
$$
v_i = x_{r1} + F(x_{r2} - x_{r3}),
$$
where $r1, r2, r3$ are random distinct indices and $F \in [0,2]$ controls mutation amplitude.

Then crossover forms trial vectors:
$$
u_i = \text{crossover}(x_i, v_i),
$$
and selection chooses between $x_i$ and $u_i$ based on objective value.

### Features
- Self-adaptive exploration of the search space.
- Suitable for continuous, multimodal functions.
- Simple to implement, with few control parameters.

 
## Particle Swarm Optimization (PSO)

Inspired by social behavior of birds and fish, Particle Swarm Optimization maintains a swarm of particles moving through the search space.

Each particle $i$ has position $x_i$ and velocity $v_i$, updated as:
$$
v_i \leftarrow w v_i + c_1 r_1 (p_i - x_i) + c_2 r_2 (g - x_i),
$$
$$
x_i \leftarrow x_i + v_i,
$$
where:

- $p_i$ = personal best position of particle $i$,
- $g$ = best global position found by the swarm,
- $w$, $c_1$, $c_2$ are weight and learning coefficients,
- $r_1$, $r_2$ are random numbers in $[0,1]$.

Particles balance individual learning (self-experience) and social learning (group knowledge).

### Convergence Behavior
Initially, the swarm explores widely; as iterations proceed, velocities decrease, and the swarm converges near optima.

### Strengths
- Few parameters, easy to implement.
- Works well for noisy or discontinuous problems.
- Naturally parallelizable.

## Simulated Annealing (SA)

Simulated Annealing is one of the earliest and most fundamental stochastic optimization algorithms. It is inspired by annealing in metallurgy — a physical process in which a material is heated and then slowly cooled to minimize structural defects and reach a low-energy crystalline state. The key idea is to imitate this gradual “cooling” in the search for a global minimum.

 
### Physical Analogy

In thermodynamics, a system at temperature $T$ has probability of occupying a state with energy $E$ given by the Boltzmann distribution:

$$
P(E) \propto e^{-E / (kT)}.
$$

At high temperature, the system freely explores many states. As $T$ decreases, it becomes increasingly likely to remain near states of minimal energy.

Simulated Annealing maps this principle to optimization by treating:

- The objective function $f(x)$ as the system’s energy.
- The solution vector $x$ as a configuration.
- The temperature $T$ as a control parameter determining randomness.


### Algorithm Outline

1. Initialization
   
      - Choose an initial solution $x_0$ and initial temperature $T_0$.
      - Set a cooling schedule $T_{k+1} = \alpha T_k$, with $\alpha \in (0,1)$.

2. Iteration
   
      - Generate a candidate $x'$ from $x_k$ via a small random perturbation.
      - Compute $\Delta f = f(x') - f(x_k)$.
      - Accept or reject based on the Metropolis criterion:
     
     $$
     P_{\text{accept}} = 
     \begin{cases}
     1, & \text{if } \Delta f \le 0, \\
     e^{-\Delta f / T_k}, & \text{if } \Delta f > 0.
     \end{cases}
     $$

3. Cooling
      - Reduce the temperature gradually according to the schedule.

      - Repeat until $T$ becomes sufficiently small or the system stabilizes.

### Interpretation

- At high temperatures, SA accepts both better and worse moves → exploration.  

- At low temperatures, it becomes increasingly selective → exploitation.

This balance allows SA to escape local minima and approach the global optimum over time.


### Cooling Schedules

The temperature schedule determines convergence quality:

| Type | Formula | Behavior |
|-----------|--------------|---------------|
| Exponential | $T_{k+1} = \alpha T_k$ | Simple, widely used |
| Linear | $T_{k+1} = T_0 - \beta k$ | Faster cooling, less exploration |
| Logarithmic | $T_k = \frac{T_0}{\log(k + c)}$ | Theoretically convergent (slow) |
| Adaptive | Adjust based on recent acceptance rates | Practical and self-tuning |

A slower cooling schedule improves accuracy but increases computational cost.




## Ant Colony Optimization (ACO)

### Biological Basis
Ant Colony Optimization models how real ants find shortest paths using pheromone trails.

Each artificial ant builds a solution step by step, choosing components probabilistically based on pheromone intensity $\tau_{ij}$ and heuristic visibility $\eta_{ij}$:
$$
P_{ij} = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_k [\tau_{ik}]^\alpha [\eta_{ik}]^\beta}.
$$

### Pheromone Update
After all ants construct their tours:
$$
\tau_{ij} \leftarrow (1 - \rho)\tau_{ij} + \sum_{\text{ants}} \Delta \tau_{ij},
$$
where $\rho$ controls evaporation and $\Delta\tau_{ij}$ reinforces paths used by good solutions.

ACO excels at combinatorial problems like the Traveling Salesman Problem (TSP) and scheduling.

## Exploration vs. Exploitation

Every metaheuristic must balance:

- Exploration: sampling diverse regions to escape local minima.  
- Exploitation: refining known good solutions to reach local optima.

| High Exploration | High Exploitation |
|-----------------------|-----------------------|
| GA with strong mutation | PSO with low inertia |
| DE with high $F$ | ACO with low evaporation rate |
| Random restarts | Local refinement |

Adaptive control of parameters (e.g., mutation rate, inertia weight) helps maintain balance dynamically.


## Hybrid and Memetic Algorithms

Hybrid (or memetic) algorithms combine global metaheuristic exploration with local optimization refinement.

Example:

1. Use PSO or GA to explore broadly.  
2. Apply gradient descent or Nelder–Mead locally near promising candidates.

This hybridization often yields faster convergence and improved accuracy.

 
## Performance and Practical Tips

| Aspect | Guideline |
|-------------|---------------|
| Initialization | Use wide, random distributions to promote diversity |
| Parameter Tuning | Use adaptive schedules (e.g., cooling, inertia decay) |
| Population Size | Larger for global search, smaller for fine-tuning |
| Parallelism | Evaluate populations concurrently for efficiency |
| Stopping Criteria | Use both iteration limits and stagnation detection |

Metaheuristics are heuristic by design — they do not guarantee global optimality, but offer practical success across many fields. Metaheuristic and evolutionary algorithms transform optimization into a process of adaptation and learning. Through populations, randomness, and natural analogies, they enable search in landscapes too complex for calculus or convexity.

