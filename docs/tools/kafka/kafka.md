# Apache Kafka 
## 1. What Is Kafka?
Apache Kafka is a distributed event streaming platform designed for:
- High-throughput data ingestion  
- Real-time analytics  
- Decoupling microservices  
- Reliable event storage and replay  

Kafka acts like a conveyor belt for data: producers place events on the belt; consumers pick them up independently.

> fundamental ideas behind Kafka: messages sent and received through Kafka require a user specified distribution strategy
## 2. Why Kafka? (Problems It Solves)

### A. The Problem With Tightly Coupled Microservices
Traditional microservices call each other directly (REST, RPC).

Issues:
1. Tight Coupling  
   If Service A depends on Service B, and B slows or fails → A also fails.

2. Single Point of Failure  
   One service outage disrupts the entire chain.

3. Scalability Limitations  
   Upstream must handle downstream load directly.

4. Lost Analytics Data  
   If analytics is down, events are lost.

Kafka solves this by decoupling communication and introducing an event broker.

 
## 3. Kafka as an Event Broker
Microservices no longer call each other.  
Instead:

- Producers publish events to Kafka.
- Topics organize those events.
- Consumers independently subscribe and process them.

This supports:
- Asynchronous communication  
- Fault isolation  
- Independent scaling  
- Reliable buffering  
- Replay of past events  

  
## 4. Core Concepts

### A. Events
An event records that something happened.

Structure:
- Key
- Value
- Timestamp
- Optional metadata (headers)

Example:
``` yaml
Key: orderId=123
Value: {"status": "CREATED", "amount": 200}
```

 
### B. Topics
A topic is a category for events of the same type (e.g., `orders`, `payments`).

#### Where Are Topics Stored?
On Kafka brokers, distributed into partitions.

 
### C. Partitions
Partition = append-only, ordered event log.

Benefits:
- Scalability (multiple partitions)
- High throughput
- Parallel processing
- Ordering (guaranteed only within a partition)


> A topic is a logical grouping of messages. A partition is a physical grouping of messages. A topic can have multiple partitions, and each partition can be on a different broker. Topics are just a way to organize your data, while partitions are a way to scale your data.

### D. Kafka Broker
A broker is a Kafka server.

Responsibilities:
- Store topic partitions  
- Handle reads/writes from producers/consumers  
- Replicate data for fault tolerance  

Replication:
- Each partition has:
  - Leader (handles requests)
  - Followers (replicas for failover)

 
### E. Consumers & Consumer Groups

#### Consumer
Reads records from topics.

#### Consumer Group
A group of consumers sharing the work of reading a topic.

Rules:
- One consumer per partition (within a group)
- But multiple groups can read the same topic independently

Examples:
- One group processes orders  
- Another group replays events for analytics  
- Another group trains ML models  

 
## 5. Kafka vs Traditional Message Queues (RabbitMQ, ActiveMQ)

| Feature | Kafka | Traditional MQ |
|--------|--------|----------------|
| Message deletion | Stored for retention period | Deleted after consumption |
| Replay messages | Yes | No |
| Storage | Disk-backed log | Often in-memory |
| Primary use | Event streaming & analytics | Task queues |

Kafka is not a database, but it *reliably stores events* and allows consumers to replay them.

 
## 6. Kafka for Real-Time Processing

### A. Kafka Streams API
A powerful library for real-time event processing.

Features:
- Continuous streaming (no polls)
- Functional transformations:  
  - map  
  - filter  
  - join  
  - aggregate  
- Windowing  
- Stateful processing  

Use cases:
- Fraud detection  
- Real-time dashboards  
- Metric aggregation  
- Data enrichment  

 
## 7. Kafka Architecture Components (Summary Table)

| Component | Role |
|----------|------|
| Producer | Writes events to topics |
| Topic | Category of events |
| Partition | Ordered log segment for scalability |
| Consumer | Reads events |
| Consumer Group | Enables parallel processing |
| Broker | Stores partitions and handles traffic |
| Cluster | Multiple brokers working together |

 
## 8. Kafka Coordination: Zookeeper vs Kafka Raft (KRaft)

### Originally: Zookeeper
Used for:
- Metadata management  
- Leader election  
- Cluster configuration  

### Now: Kafka KRaft (Kafka Raft)
- Built-in consensus protocol  
- No Zookeeper dependency  
- Simplifies deployment  
- Better scalability and reliability  

Zookeeper is being phased out.

 
## 9. Key Benefits of Kafka
- High throughput  
- Low latency  
- Durable storage  
- Scalable horizontally  
- Fault-tolerant  
- Can replay historical events  
- Decouples microservices  
- Supports real-time analytics  
 

https://www.youtube.com/watch?v=QkdkLdMBuL0

https://www.youtube.com/watch?v=B7CwU_tNYIE