# Chapter 2: Metaheuristic Optimization Methods

Metaheuristics are **general-purpose stochastic search algorithms** inspired by natural or social processes.
They do not require gradient information and are particularly powerful for **non-convex**, **discrete**, and **black-box** problems.

---

## 2.1 Core Principles

Metaheuristics operate by balancing two key dynamics:

* **Exploration:** Searching new, unvisited regions of the solution space
* **Exploitation:** Refining promising areas to improve solution quality

Algorithms differ in how they maintain this balance — through temperature schedules, populations, or probabilistic moves.

---

## 2.2 Major Families of Metaheuristics

| Category                       | Examples                                                         | Inspiration                |
| ------------------------------ | ---------------------------------------------------------------- | -------------------------- |
| **Trajectory-based**           | Simulated Annealing (SA)                                         | Thermodynamics             |
| **Evolutionary algorithms**    | Genetic Algorithms (GA), Differential Evolution (DE)             | Natural selection          |
| **Swarm intelligence**         | Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO) | Collective animal behavior |
| **Physics/chemistry inspired** | Harmony Search, Firefly Algorithm, Gravitational Search          | Physical processes         |

---

## 2.3 Simulated Annealing (SA)

* Mimics the cooling of metals.
* Accepts worse moves with a probability `exp(-ΔE/T)`, allowing escape from local minima.
* Temperature `T` decreases over time.

> **Key idea:** Controlled randomness to avoid premature convergence.

---

## 2.4 Genetic Algorithms (GA)

* Maintain a **population** of solutions.
* Apply **selection**, **crossover**, and **mutation** to evolve toward better candidates.
* Works well for discrete, combinatorial, or mixed-variable problems.

> **Mathematical insight:** Balances exploitation (selection) and exploration (mutation).

---

## 2.5 Swarm-Based Methods

Inspired by collective behaviors in nature:

* **PSO:** Particles move based on personal and social bests
* **ACO:** Agents deposit pheromones to guide search
* **Firefly Algorithm:** Movement toward brighter (better) peers

> These methods excel in **continuous** search spaces and dynamic environments.

---

## 2.6 Theoretical Considerations

Although metaheuristics lack strong convex guarantees, their **stochastic convergence** can be studied via:

* Markov chain analysis
* Expected improvement over iterations
* Diversity measures within populations

They often converge **probabilistically** to a near-optimal region rather than a single guaranteed optimum.
