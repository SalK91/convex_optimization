# Introduction

What is "distributed system":

A group of computers cooperating to provide a service


## Why?

1. to increase capacity via parallel processing
2. to tolerate faults via replication
3. to match distribution of physical devices e.g. sensors
5. to increase security via isolation

##  Challanges:

- concurrency
- complex interactions
- performance bottlenecks
- partial failure

## Key Topics

### Fault tolerance:

- 1000s of servers, big network -> always something broken
- We'd like to hide these failures from the application.
- "High availability": service continues despite failures
- Big idea: replicated servers. If one server crashes, can proceed using the other(s).

### Consistency:

- General-purpose infrastructure needs well-defined behavior. E.g. "read(x) yields the value from the most recent write(x)."
- Achieving good behavior is hard! e.g. "replica" servers are hard to keep identical.

### Performance:

- The goal: scalable throughput. Nx servers -> Nx total throughput via parallel CPU, RAM, disk, net.
- Scaling gets harder as N grows:
    - Load imbalance.
    - Slowest-of-N latency.

### Tradeoffs:

- Fault-tolerance, consistency, and performance are enemies.
- Fault tolerance and consistency require communication
    - e.g., send data to backup server
    - e.g., check if cached data is up-to-date
    - communication is often slow and non-scalable
- Many designs sacrifice consistency to gain speed.
    - e.g. read(x) might *not* yield the latest write(x)!
    - Painful for application programmers (or users).
  
### Implementation:

- RPC, threads, concurrency control, configuration.
 