# 6.5840 — Lecture 2 (2025): Threads and RPC

## Introduction: Implementing Distributed Systems

This lecture introduces:
- Go threads (goroutines)
- Concurrency challenges
- The web crawler example
- Remote Procedure Calls (RPC)

Go is the language used for this writeup.

## Why Go?

Go is well-suited for distributed systems:

- Excellent thread (goroutine) support  
- Convenient RPC library
- Type- and memory-safe
- Garbage-collected (safe with concurrency)
- Simpler than many other languages
- Commonly used in production distributed systems

👉 After the tutorial, read Effective Go:  
https://golang.org/doc/effective_go.html


# Threads (Goroutines)

## What is a Thread?

A thread is a “thread of execution”:

- Allows a program to do multiple things at once
- Executes sequentially (like a program), but shares memory with other threads
- Has its own program counter, registers, and stack

In Go, threads are called goroutines.



## Why Use Threads?
Three type of threads:

### 1. I/O Concurrency

- Client sends requests to many servers at once  
- Server handles many clients concurrently  
- When one thread blocks on I/O, another can run

### 2. Multicore Performance
Use multiple CPU cores simultaneously.

### 3. Convenience
Run background tasks (e.g., periodic health checks).


## Alternative: Event-Driven Systems

Instead of threads:

- Use a single-threaded system with an event loop
- Explicitly interleave different activities
- Maintain state tables for each ongoing operation

Pros:  

- Good for I/O concurrency  
- No thread overhead

Cons:  

- No multicore usage  
- Hard to program and maintain

# Threading Challenges

## 1. Safe Data Sharing

Race example:
```go
n = n + 1
```
Two threads modifying `n` at the same time → race condition.

A race is when:
- Two threads access the same memory
- At least one is a write
- And there's no synchronization

Fixes:
- Use `sync.Mutex`
- Avoid sharing mutable data


## 2. Coordination (Producer–Consumer)

- One thread produces data  
- Another consumes it  
- Need a way for consumers to wait and wake up

Tools:
- Go channels
- `sync.Cond`
- `sync.Wait`, `sync.WaitGroup`

## 3. Deadlock

When threads wait on each other forever.  
Can happen via:

- Locks
- Channels
- RPC


# Web Crawler Example

A web crawler:

- Fetches web pages recursively starting from a URL
- Follows links
- Avoids revisiting pages
- Avoids cycles
- Exploits I/O concurrency for speed


## 1. Serial Crawler

- Depth-first traversal  
- A shared map tracks visited URLs  
- Simple and correct  
- Very slow — only fetches one page at a time  

Adding `go` before recursive calls breaks correctness:

- Many threads may fetch same URL  
- Finishing detection becomes difficult


# 2. Concurrent Crawler with Mutex

### How it Works

- Launch a goroutine per page
- Shared `fetched` map  
- Mutex ensures only one thread fetches each URL

### Why the Mutex?

#### 1. Avoid Logical Races
Two threads may check the same URL at once:
- Both see `fetched[url] == false`
- Both fetch → wrong

Mutex ensures:
- One thread checks + sets at a time

#### 2. Avoid Map Corruption
Go maps are not thread-safe.



## What If Lock Is Removed?

- Program may appear to work sometimes  
- But races still occur  
- Use the race detector:

```
go run -race crawler.go
```


## Completion Detection Using sync.WaitGroup

- `Add(n)` increments  
- `Done()` decrements  
- `Wait()` blocks until count is zero  

Ensures main thread waits for all children.


# 3. Concurrent Crawler with Channels

Channels provide:

- Communication  
- Synchronization  

### Channel Basics

```go
ch := make(chan int)

ch <- x   // send (blocks)
y := <-ch // receive (blocks)
```

---

## Coordinator + Workers Model

- Coordinator creates workers via goroutines  
- Workers fetch a page and send resulting URLs via channel  
- Coordinator receives URLs, checks visited set

### Why No Mutex?

- Shared state is only in coordinator  
- Workers never mutate shared maps  
- Therefore no races

## Channel Safety

Example:

- Worker creates slice of URLs
- Sends it to channel
- Coordinator reads it

Safe because:

- Worker writes slice before send completes
- Coordinator reads slice after receive completes

No overlap → no race.



## Why Some Sends Need a Goroutine?

Without a goroutine:

- send blocks  
- coordinator may not reach the receive  
- → deadlock

## Locks vs Channels

Both are powerful. Use whichever matches intuition:

- State-focused logic → locks
- Communication-focused logic → channels

In 6.5840 labs:
- Use sharing + locks for state
- Use channels, `sync.Cond`, or sleep-based polling for notifications


# Remote Procedure Call (RPC)

RPC enables easy client-server communication.

## Goals

- Hide network details
- Provide a procedure-call interface
- Automatically marshal/unmarshal data
- Enable portability across systems


## RPC Architecture

```
Client               Server
  request ---->
            <---- response
```

Software structure:
```
Client App        Server Handlers
Client Stubs      Dispatcher
RPC Library  ---- RPC Library
 Network     ---- Network
```

# Go RPC Example: Key/Value Store

Handlers:
- `Put(key, value)`
- `Get(key) -> value`



## Client Side

- Use `Dial()` to connect  
- Call RPC using:

```go
Call("KVServer.Get", args, &reply)
```

RPC library:
- Marshals args  
- Sends request  
- Waits for reply  
- Unmarshals reply  
- Returns error if something went wrong  

## Server Side

Server must:
1. Declare a type with exported RPC methods  
2. Register the type  
3. Accept TCP connections and let RPC library handle them  

RPC library:

- Creates goroutine per request  
- Unmarshals request  
- Dispatches handler  
- Marshals reply  
- Sends reply  

Handlers must use locks since multiple RPCs run concurrently.


## RPC Details

### Binding
Client must know `"server:port"` to dial.

### Marshalling Rules
- Sends strings, arrays, structs, maps  
- Cannot send channels or functions  
- Only exported fields in structs are marshaled  
- Pointers are sent by copying the pointee


# RPC Failures

Client may never get a reply:

Could mean:

- Server never received request  
- Server crashed after executing  
- Reply lost in network  
- Network or server slow  

RPC ≠ local function call.


# Best-Effort RPC

Algorithm:
1. Send request  
2. Wait  
3. If no reply, resend  
4. After several tries → give up  

### Problems

Example:
```
Put("k", 10)
Put("k", 20)
```

Retries can reorder or duplicate operations.


## When Is Best-Effort OK?

- Read-only operations  
- Idempotent operations (safe to repeat)


#  At-Most-Once Semantics

Go RPC provides:

- One TCP connection  
- Sends each request once  
- No retries → no duplicates  

But:

- Errors returned on timeouts  
- Hard to build replicated fault-tolerant systems without retries  

Later labs explore stronger semantics.

