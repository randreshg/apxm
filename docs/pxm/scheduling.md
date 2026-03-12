---
title: "Scheduling and Execution in Program Execution Models"
description: "A comparative analysis of scheduling and execution across classical and modern PXMs, and how A-PXM synthesizes these ideas for agent workflows."
---

# Scheduling and Execution in Program Execution Models

A Program Execution Model (PXM) defines three interrelated concerns: how execution order is determined, how parallelism is discovered, and how synchronization enforces correctness. Every computing system -- from a single-core CPU to a distributed data processing cluster -- embeds a PXM, whether formalized or not. This document examines six foundational PXMs, then shows how A-PXM synthesizes their strongest properties into a scheduling and execution model purpose-built for agentic AI workflows.

---

## 1. Von Neumann Sequential Execution

### How Execution Order Is Determined

The von Neumann model is defined by a single abstraction: the **program counter** (PC). The PC holds the address of the next instruction. After each instruction executes, the PC advances to the next sequential address unless a branch or jump redirects it. Execution order is therefore a total order encoded in the instruction stream -- every instruction has a unique predecessor and successor.

The control flow is explicit and linear:

```
PC = 0x0000
fetch instruction at PC
decode
execute
PC = PC + instruction_size  (or branch target)
repeat
```

Branch prediction is a hardware optimization that speculates on the next PC value before the branch condition is resolved. Modern CPUs achieve prediction accuracy above 95% for well-structured code, but prediction is transparent to the programming model -- it does not change the semantics, only the throughput.

### How Parallelism Is Discovered and Exploited

In the pure von Neumann model, parallelism is **invisible**. The instruction stream specifies a single thread of control. Any parallelism must be introduced explicitly by the programmer through multi-threading, vectorization pragmas, or parallel library calls. The hardware sees one instruction at a time; the programmer bears the full burden of decomposition, mapping, and synchronization.

This creates the **von Neumann bottleneck**: the processor and memory communicate through a single narrow channel (the bus), and all data flows through the sequential instruction stream regardless of whether true data dependencies exist.

### How Synchronization Is Handled

Synchronization in sequential execution is trivially satisfied by program order. Instruction N completes before instruction N+1 begins. When explicit parallelism is introduced (threads, processes), synchronization must be layered on through locks, semaphores, barriers, condition variables, or memory fences -- mechanisms external to the core execution model.

---

## 2. Dataflow Execution (Manchester Machine)

### How Execution Order Is Determined

The Manchester Dataflow Machine (1981) eliminated the program counter entirely. Execution order is determined by **data availability**, not instruction address. Each operation is a node in a dataflow graph. An operation fires -- becomes eligible for execution -- when **all** of its input tokens have arrived. There is no sequential instruction stream; there is only data flowing through a network of operations.

The firing rule is:

```
for each operation O:
    if all input_tokens(O) are present:
        fire(O)
        consume input tokens
        produce output tokens on outgoing arcs
```

This is a fundamentally different model of computation. Two operations with no data dependency between them are **inherently concurrent** -- neither waits for the other, and no scheduler needs to discover this fact. The graph topology is the schedule.

### How Parallelism Is Discovered and Exploited

Parallelism is **intrinsic** to the model. It is not discovered by analysis passes or requested by annotations -- it is a structural property of the dataflow graph. If two operations share no common input tokens, they can execute simultaneously. The degree of available parallelism is bounded only by the width of the graph at any given wavefront.

The Manchester Machine implemented this with a token-matching store: tokens were tagged with their destination operation and stored in an associative memory. When all tokens for an operation had arrived, the operation was dispatched to a processing element. Multiple processing elements could fire independent operations simultaneously.

### How Synchronization Is Handled

Synchronization is implicit in the token-matching mechanism. A join point (an operation with multiple inputs) naturally blocks until all inputs are present. There are no locks, no barriers, and no race conditions in the classical sense -- the data dependencies in the graph are the synchronization.

The challenge historically was performance: the associative token store required expensive content-addressable memory, and the overhead of tagging and matching tokens made dataflow machines slower than optimized von Neumann processors for sequential workloads. These hardware costs were prohibitive in the 1980s, but the model itself -- data-driven execution with structural parallelism -- remains theoretically elegant.

---

## 3. Out-of-Order Execution (Tomasulo, Scoreboarding)

### How Execution Order Is Determined

Out-of-order (OoO) execution is a hardware technique that preserves the **illusion** of sequential execution while internally reordering instructions based on data readiness. The programmer and compiler see a von Neumann machine; the microarchitecture implements something closer to restricted dataflow.

**Scoreboarding** (CDC 6600, 1964) introduced hardware tracking of register dependencies. Each functional unit was tagged with the registers it read and wrote. An instruction could issue only when its source registers were available and its destination register was not being written by another instruction. This detected true data dependencies (RAW) and prevented conflicts (WAW, WAR).

**Tomasulo's algorithm** (IBM 360/91, 1967) went further with **register renaming**. By assigning physical registers dynamically via reservation stations, Tomasulo's algorithm eliminated false dependencies (WAW and WAR hazards) entirely. Instructions were dispatched to reservation stations where they waited for their operands. When operands became available (broadcast via the Common Data Bus), the instruction fired.

The result: execution order is determined by true data dependencies, not by program order, but results are committed (retired) in program order to maintain sequential semantics.

### How Parallelism Is Discovered and Exploited

Parallelism is discovered by the hardware at run time through dependency analysis on a sliding window of instructions (the reorder buffer, typically 100-300 instructions deep in modern designs). Independent instructions within this window execute concurrently on multiple functional units. The programmer does not need to specify parallelism; the hardware extracts it from the sequential stream.

The effective parallelism is bounded by:
- The reorder buffer depth (how far ahead the hardware can look)
- The number of functional units (how many operations can execute simultaneously)
- True data dependencies (the irreducible serialization in the algorithm)

### How Synchronization Is Handled

Synchronization is maintained through the **reorder buffer** and **in-order retirement**. Although instructions execute out of order, they are committed to the architectural state in the original program order. This ensures that interrupts, exceptions, and memory ordering guarantees are preserved. Memory ordering is further enforced through load-store queues and memory barriers.

---

## 4. Task-Based Parallelism (Cilk, TBB, Tokio)

### How Execution Order Is Determined

Task-based systems model computation as a **directed acyclic graph of tasks** (a task DAG), where edges represent dependencies. The programmer decomposes work into tasks and declares dependencies through language constructs:

- **Cilk**: `cilk_spawn` creates a child task; `cilk_sync` waits for children to complete. The serial elision (removing all Cilk keywords) produces a valid sequential program.
- **TBB (Threading Building Blocks)**: `task_group`, `parallel_for`, `flow_graph`. Tasks are objects with `execute()` methods scheduled by the TBB runtime.
- **Tokio**: `async fn` defines a task; `.await` yields control until a future resolves. The Tokio runtime multiplexes many tasks onto a small thread pool.

Execution order is partially ordered: within a task, instructions execute sequentially; across tasks, order is determined by explicit dependency declarations.

### How Parallelism Is Discovered and Exploited

Parallelism is **programmer-directed but runtime-scheduled**. The programmer marks tasks and their dependencies; the runtime decides when and where to execute them. The key scheduling innovation shared by Cilk and Tokio is **work stealing**:

1. Each worker thread maintains a local deque of ready tasks.
2. When a thread has no local work, it steals from another thread's deque.
3. The stealing strategy (typically random victim selection) provides probabilistic load balancing.

Cilk's work-stealing scheduler achieves a provable bound: for a computation with work T1 (total operations) and span T_inf (critical path length), execution on P processors completes in time O(T1/P + T_inf). This is asymptotically optimal.

Tokio adds cooperative multitasking within each worker thread: tasks yield at `.await` points, allowing the runtime to interleave thousands of I/O-bound tasks on a small number of OS threads.

### How Synchronization Is Handled

Synchronization is explicit in the programming model:
- **Cilk**: `cilk_sync` acts as a barrier for all spawned children.
- **TBB**: `task_group::wait()`, `flow_graph` edges.
- **Tokio**: `.await` on a `JoinHandle`, `tokio::join!` for concurrent awaiting, channels for message passing, `Mutex`/`Semaphore` for shared state.

The programmer must reason about which tasks can run concurrently and where synchronization points are needed. Errors in this reasoning lead to data races, deadlocks, or incorrect results. The async/await pattern simplifies the syntax but does not eliminate the cognitive burden of concurrent reasoning.

---

## 5. MapReduce / Spark

### How Execution Order Is Determined

MapReduce and Spark implement a **bulk-synchronous** execution model organized into discrete stages separated by global synchronization barriers.

**MapReduce** has a fixed two-stage pipeline:
1. **Map stage**: Apply a function independently to each input record, producing key-value pairs.
2. **Shuffle barrier**: All map outputs are partitioned by key, sorted, and transferred to reducers. No reducer can start until all mappers have finished and the shuffle is complete.
3. **Reduce stage**: Aggregate all values for each key.

**Spark** generalizes this to arbitrary DAGs of stages. A Spark job is a DAG of **stages**, where each stage contains a set of **tasks** that can run in parallel. Stage boundaries occur at **shuffle dependencies** (wide dependencies) where data must be repartitioned across the cluster.

Within a stage, execution order is determined by the data partition -- each task processes one partition independently. Across stages, execution order is determined by the shuffle barriers in the stage DAG.

### How Parallelism Is Discovered and Exploited

Parallelism is **data-parallel within stages**: each partition is processed by an independent task. The degree of parallelism equals the number of partitions, which is configurable. Within a stage, tasks are embarrassingly parallel -- they share no mutable state and communicate only through the shuffle.

Across stages, parallelism is limited. Spark's DAG scheduler can overlap independent stages (stages with no shuffle dependency between them), but the shuffle barrier between dependent stages forces full materialization of intermediate data before the next stage begins.

### How Synchronization Is Handled

Synchronization is handled through **shuffle barriers**. A shuffle barrier is a global synchronization point: all tasks in the upstream stage must complete and their outputs must be materialized before any task in the downstream stage can begin. This is coarse-grained synchronization -- it avoids the complexity of fine-grained locking but introduces latency at stage boundaries.

Fault tolerance is achieved through **lineage**: if a partition is lost, Spark recomputes it from its parent partitions using the recorded transformation DAG, rather than relying on replication.

---

## 6. Petri Nets

### How Execution Order Is Determined

A Petri net is a bipartite directed graph with two kinds of nodes: **places** (drawn as circles) and **transitions** (drawn as bars or rectangles). Arcs connect places to transitions and transitions to places. Places hold **tokens** -- the dynamic state of the system. The distribution of tokens across places is called the **marking**.

A transition is **enabled** when all of its input places contain at least one token. An enabled transition may **fire**: it atomically removes one token from each input place and adds one token to each output place. If multiple transitions are simultaneously enabled, the firing order is non-deterministic.

```
Firing rule:
  transition t is enabled iff:
    for all p in input_places(t): marking(p) >= 1

  fire(t):
    for all p in input_places(t): marking(p) -= 1
    for all p in output_places(t): marking(p) += 1
```

Execution order emerges from the marking: at any point, the set of enabled transitions determines what can happen next. The marking evolves as transitions fire, enabling new transitions and disabling others.

### How Parallelism Is Discovered and Exploited

Parallelism in Petri nets is structural: if two transitions are enabled and share no common input or output places, they can fire concurrently. This is a natural model for concurrent systems -- the graphical structure makes concurrency, conflict (mutual exclusion), and synchronization visually explicit.

Petri nets model **true concurrency** rather than interleaving. Two concurrent transitions are not ordered relative to each other, even in the formal semantics. This makes Petri nets a powerful formalism for reasoning about concurrent protocols, resource allocation, and workflow systems.

### How Synchronization Is Handled

Synchronization is modeled through the **place-transition structure** itself:

- **Fork**: A transition with multiple output places distributes tokens, modeling fan-out.
- **Join**: A transition with multiple input places requires all inputs before firing, modeling a synchronization barrier.
- **Mutual exclusion**: A shared place with a single token enforces that only one of several competing transitions can fire at a time.
- **Choice**: A place with multiple output transitions models non-deterministic or priority-based selection.

The elegance of Petri nets lies in their ability to model all common synchronization patterns without additional mechanisms. The firing rule itself is the synchronization primitive.

---

## Comparative Summary

| Property | Von Neumann | Dataflow (Manchester) | OoO (Tomasulo) | Task-Based (Cilk/Tokio) | MapReduce/Spark | Petri Nets |
|---|---|---|---|---|---|---|
| **Order determination** | Program counter | Data availability | Data readiness (hardware) | Task DAG + runtime | Stage DAG + shuffle | Token marking |
| **Parallelism discovery** | Manual (programmer) | Automatic (structural) | Automatic (hardware window) | Semi-automatic (programmer decomposes, runtime schedules) | Automatic within stage (data-parallel) | Structural |
| **Synchronization** | Program order / locks | Token matching (implicit) | Reorder buffer (hardware) | async/await, barriers, channels | Shuffle barriers | Place-transition firing rule |
| **Granularity** | Instruction | Operation | Instruction | Task (function) | Stage (partition) | Transition |
| **Developer burden** | Full concurrency reasoning | None for parallelism | None (transparent) | Decomposition + sync points | None within stages | Model construction |

---

## How A-PXM Schedules and Executes Agent Workflows

A-PXM (Agent Program Execution Model) synthesizes ideas from the models above into a scheduling and execution system designed for the specific characteristics of agentic AI workloads: high-latency operations (LLM calls measured in seconds), heterogeneous operation types (memory queries, tool invocations, reasoning steps), and multi-agent coordination requirements.

### Token-Based Dataflow Scheduler

The A-PXM scheduler is a **token-counting dataflow scheduler** directly inspired by the Manchester Machine. The core mechanism is identical in principle: operations fire when all input data is available. But where the Manchester Machine used expensive associative hardware for token matching, A-PXM replaces it with an efficient software implementation using atomic counters.

Every operation (node) in the execution DAG maintains a **pending counter** initialized to its number of input edges (its in-degree). The scheduling loop is:

1. When an upstream operation completes, it **produces tokens** on each of its outgoing edges.
2. Each produced token is routed to downstream consumers. The consumer's pending counter is **atomically decremented**.
3. When a counter reaches **zero**, the operation is **ready** and is inserted into the ready set and priority queue for dispatch.

```
on_token_produced(token_id):
    for each consumer of token_id:
        count = atomic_decrement(consumer.pending_count)
        if count == 0:
            mark_ready(consumer)
            enqueue(consumer, priority)
```

This provides **O(1) readiness detection**: determining whether an operation can fire requires a single atomic decrement and comparison. No graph traversal is needed. No dependency resolution is performed at fire time. The counter is the complete readiness oracle.

The overhead per operation is approximately **7.5 microseconds**, broken down as:

| Phase | Time | Share |
|---|---|---|
| Ready set update | 1.8 us | 24.0% |
| Priority queue dequeue | 0.9 us | 12.0% |
| Dependency resolution | 1.4 us | 18.7% |
| Operation dispatch | 2.4 us | 32.0% |
| Token routing | 1.0 us | 13.3% |

This overhead is negligible against the millisecond-to-second latencies of LLM calls and tool invocations that constitute the actual work in agent workflows.

### Automatic Parallelism Without async/await

A-PXM's central design thesis is that the DAG structure **is** the parallelism specification. If two operations have no edge between them in the DAG, they are independent and will execute concurrently. The developer never writes `async`, `await`, `Promise.all()`, `tokio::join!`, or thread management code.

Consider a fan-out/fan-in pattern where a reasoning step triggers three independent tool calls:

```
REASON --> INV tool_a (500ms)
       --> INV tool_b (800ms)
       --> INV tool_c (300ms)
                        \
                         --> WAIT_ALL --> ASK (summarize)
```

In a sequential framework, wall-clock time for the tool calls is 500 + 800 + 300 = **1600ms**. In A-PXM, the three INV operations have no edges between them, so they fire concurrently. Wall-clock time is max(500, 800, 300) = **800ms** -- a 2x speedup with zero developer effort.

This is fundamentally different from task-based systems like Cilk or Tokio, where the developer must explicitly spawn tasks and await their results. In A-PXM, the DAG edges declared during workflow authoring carry the complete information needed for the scheduler to extract all available parallelism. The separation is clean: **the developer declares data dependencies; the runtime exploits independence**.

### Work-Stealing Executor Pool

While the scheduling model is dataflow, the execution substrate uses a **work-stealing thread pool** inspired by Cilk's scheduler. Each worker thread maintains a local FIFO deque of ready operations. The stealing strategy operates in three tiers:

1. **Local deque** (O(1), no contention): Pop from the worker's own queue.
2. **Global priority injectors** (low contention): Steal from global queues, checking Critical > High > Normal > Low priority levels in order.
3. **Peer worker queues** (round-robin): Steal from other workers' deques when both local and global queues are empty.

This hybrid of dataflow scheduling and work-stealing execution combines the best of both models: the dataflow token system determines **what** is ready; the work-stealing pool determines **where** it runs. The result adapts automatically to heterogeneous operation durations (a 10-second REASON call and a 100ms ASK call on the same graph) without requiring a centralized load balancer.

### Typed Operation Dispatch

Unlike Petri nets, where transitions are untyped (they simply consume and produce tokens), A-PXM operations are **typed** with operation-specific executors. Each AIS instruction category has a dedicated handler:

| Executor | Operations | Responsibility |
|---|---|---|
| Memory | QMEM, UMEM, FENCE | Three-tier memory hierarchy access |
| LLM | ASK, THINK, REASON, PLAN, REFLECT, VERIFY | Model API dispatch with latency budgets |
| Tool | INV | External tool invocation with typed parameter marshalling |
| Control | BRANCH_ON_VALUE, SWITCH | Conditional routing in the DAG |
| Sync | MERGE, WAIT_ALL | Parallel path synchronization |
| Communication | COMM, FLOW_CALL | Cross-agent messaging and flow invocation |

The operation dispatcher routes each fired node to its appropriate handler based on the operation type. This type-awareness enables the scheduler to make informed decisions: LLM operations carry **latency budgets** (ASK ~1s, THINK ~3s, REASON ~10s) that the scheduler uses to prioritize critical-path operations and overlap long-running reasoning with independent fast operations.

### WAIT_ALL and FENCE: Explicit Synchronization in the DAG

A-PXM provides two primary synchronization primitives that exist as first-class nodes in the DAG rather than as runtime library calls:

**WAIT_ALL** is the fan-in synchronization point for parallel execution paths. It maintains its own pending counter (identical to the scheduler's per-node counter) initialized to its number of input edges. As tokens arrive, the counter decrements. When it reaches zero, all values are collected into an ordered tuple and emitted downstream.

```
REASON --> INV_A (500ms) --\
       --> INV_B (800ms) ---+--> WAIT_ALL (counter: 3) --> THINK
       --> INV_C (300ms) --/
```

After 800ms (when the slowest call finishes), all three tokens are present and THINK can fire with the complete result set.

**FENCE** is a memory barrier node. It ensures that all UMEM (memory write) operations that precede it in the DAG complete before any QMEM (memory read) operations that follow it can execute. FENCE carries no data -- it is a pure ordering constraint.

```
UMEM_A --\
          +--> FENCE --> QMEM
UMEM_B --/
```

Both writes commit before the read executes, guaranteeing consistency when the read depends on aggregate state produced by multiple writers.

**MERGE** handles the complementary case: reconvergence after conditional branching. Unlike WAIT_ALL (which waits for all N inputs), MERGE expects exactly one token because it sits after a BRANCH or SWITCH where only one path fires. It passes the arriving token through as its output.

These synchronization primitives are declarative, not imperative. They are nodes in the graph, subject to the same token-based readiness rules as every other operation. The developer expresses synchronization intent by wiring edges; the runtime enforces it through the firing rule.

### Session Lane Guards: Per-Session Serialization

Agent systems commonly serve multiple concurrent users (sessions). Within a single session, operations must execute serially to maintain conversational coherence (message A must be processed before message B for the same user). Across sessions, operations should run concurrently for throughput.

A-PXM's **SessionLaneGuard** solves this with a simple mechanism: a `DashMap<SessionId, Arc<Mutex<()>>>` that maps each session to a per-session mutex. When a request arrives:

```rust
let _permit = session_lane_guard.acquire(session_id).await;
// ... entire DAG execution runs under this permit ...
// permit is released via RAII when execution completes
```

The semantics are:
- Requests for the **same session** are serialized (the mutex ensures only one DAG executes at a time per session).
- Requests for **different sessions** run fully concurrently (each session has its own independent mutex).

This is architecturally distinct from both the dataflow scheduler (which handles intra-DAG parallelism) and the work-stealing pool (which handles operation-to-thread mapping). Session lane guards operate at the **request level**, gating whether an entire DAG execution should begin. The three mechanisms compose cleanly: lane guards serialize session entry, the dataflow scheduler extracts parallelism within the DAG, and work stealing balances load across threads.

### Concurrency Control and Backpressure

The scheduler enforces a configurable **maximum inflight operation count** via a semaphore-based concurrency controller. When the number of executing operations reaches the limit, workers block on permit acquisition until an in-flight operation completes. This provides backpressure to prevent resource exhaustion under bursty workloads (e.g., a deeply parallel DAG that would otherwise launch hundreds of concurrent LLM calls).

Additionally, a **watchdog** monitors progress. If no operation completes and no operation is running for a configurable timeout, the watchdog diagnoses a deadlock and aborts execution with a diagnostic error. This catches structural errors in the DAG (e.g., circular dependencies introduced by dynamic splicing) that would otherwise cause silent hangs.

### Comparison with Classical Models

| Property | Manchester Machine | Petri Nets | Cilk / Tokio | A-PXM |
|---|---|---|---|---|
| **Firing rule** | All tokens present | All input places have tokens | Explicit spawn/await | Pending counter == 0 |
| **Operation types** | Untyped (arithmetic) | Untyped transitions | Typed functions | Typed AIS instructions with category-specific executors |
| **Parallelism** | Structural (graph) | Structural (marking) | Programmer-directed | Structural (DAG edges) |
| **Synchronization** | Token matching | Place-transition structure | async/await, barriers | WAIT_ALL, FENCE, MERGE as DAG nodes |
| **Session isolation** | N/A | Colored tokens (extensions) | Manual (programmer) | SessionLaneGuard (per-session mutex) |
| **Readiness cost** | Associative match (hardware) | Marking check | Runtime queue | O(1) atomic decrement |
| **Scheduling** | Hardware dispatch | Non-deterministic | Work-stealing | Priority-aware work-stealing |
| **Developer burden** | None (hardware) | Model construction | Task decomposition + sync | None -- DAG is the spec |

### Why This Design

The design choices in A-PXM's scheduler are driven by the characteristics of agentic workloads:

1. **Operations are coarse-grained and high-latency.** Unlike CPU instructions (nanoseconds) or Spark tasks (milliseconds to seconds of CPU-bound work), agent operations involve network round-trips to LLM APIs (seconds) and tool invocations (variable). The 7.5-microsecond scheduling overhead is negligible. This makes software-based token counting viable where the Manchester Machine needed dedicated hardware.

2. **Parallelism is abundant but not obvious.** Agent workflows frequently involve multiple independent tool calls, parallel research paths, or concurrent sub-agent executions. Making the developer manually parallelize these with async/await is error-prone and obscures the workflow logic. Structural parallelism from the DAG topology extracts this automatically.

3. **Multi-session serving is the common case.** Production agent systems serve many users concurrently. Session lane guards provide the correct serialization granularity (per-user serial, cross-user parallel) without requiring the developer to implement session management.

4. **Operations have heterogeneous semantics.** A memory read, an LLM reasoning call, and a tool invocation have fundamentally different execution characteristics, failure modes, and resource requirements. Typed operation dispatch with category-specific executors (unlike the uniform transitions of Petri nets) allows the runtime to apply appropriate strategies for each operation type: latency budgets for LLM calls, retry policies for tool invocations, transactional semantics for memory writes.

5. **The DAG is the contract.** By making the dataflow graph the single source of truth for both execution order and parallelism, A-PXM eliminates the gap between specification and execution that plagues imperative orchestration frameworks. The compiler can verify the graph statically. The runtime executes it faithfully. The developer reasons about data flow, not about threads, locks, or async state machines.

---

## References

### Von Neumann Sequential Execution

1. J. von Neumann, "First Draft of a Report on the EDVAC," Moore School of Electrical Engineering, University of Pennsylvania, 1945. Reprinted in *IEEE Annals of the History of Computing*, vol. 15, no. 4, pp. 27--75, 1993.
2. J. Backus, "Can Programming Be Liberated from the von Neumann Style? A Functional Style and Its Algebra of Programs," *Communications of the ACM*, vol. 21, no. 8, pp. 613--641, August 1978.

### Dataflow Execution (Manchester Machine)

3. J. R. Gurd, C. C. Kirkham, and I. Watson, "The Manchester Prototype Dataflow Computer," *Communications of the ACM*, vol. 28, no. 1, pp. 34--52, January 1985.
4. J. B. Dennis, "First Version of a Data Flow Procedure Language," in *Programming Symposium*, Lecture Notes in Computer Science, vol. 19, Springer, pp. 362--376, 1974.
5. Arvind and D. E. Culler, "Dataflow Architectures," *Annual Review of Computer Science*, vol. 1, pp. 225--253, 1986.

### Out-of-Order Execution (Tomasulo, Scoreboarding)

6. R. M. Tomasulo, "An Efficient Algorithm for Exploiting Multiple Arithmetic Units," *IBM Journal of Research and Development*, vol. 11, no. 1, pp. 25--33, January 1967.
7. J. E. Thornton, *Design of a Computer: The Control Data 6600*. Glenview, IL: Scott, Foresman, 1970.
8. J. E. Smith and G. S. Sohi, "The Microarchitecture of Superscalar Processors," *Proceedings of the IEEE*, vol. 83, no. 12, pp. 1609--1624, December 1995.

### Task-Based Parallelism (Cilk, TBB, Tokio)

9. R. D. Blumofe, C. F. Joerg, B. C. Kuszmaul, C. E. Leiserson, K. H. Randall, and Y. Zhou, "Cilk: An Efficient Multithreaded Runtime System," in *Proceedings of the Fifth ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '95)*, pp. 207--216, 1995.
10. R. D. Blumofe and C. E. Leiserson, "Scheduling Multithreaded Computations by Work Stealing," *Journal of the ACM*, vol. 46, no. 5, pp. 720--748, September 1999.
11. Tokio Contributors, "Tokio: An Asynchronous Runtime for the Rust Programming Language," https://tokio.rs/. Accessed 2025.

### MapReduce / Spark

12. J. Dean and S. Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters," in *Proceedings of the 6th USENIX Symposium on Operating Systems Design and Implementation (OSDI '04)*, pp. 137--150, 2004.
13. M. Zaharia, M. Chowdhury, T. Das, A. Dave, J. Ma, M. McCauley, M. J. Franklin, S. Shenker, and I. Stoica, "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing," in *Proceedings of the 9th USENIX Symposium on Networked Systems Design and Implementation (NSDI '12)*, pp. 15--28, 2012.

### Petri Nets

14. C. A. Petri, "Kommunikation mit Automaten," Doctoral dissertation, Universitat Hamburg, 1962. English translation: "Communication with Automata," Technical Report RADC-TR-65-377, Griffiss Air Force Base, New York, 1966.
15. T. Murata, "Petri Nets: Properties, Analysis and Applications," *Proceedings of the IEEE*, vol. 77, no. 4, pp. 541--580, April 1989.

### A-PXM and the Codelet Model

16. G. R. Gao, R. Patel, and T. Sterling, "The Codelet Program Execution Model," in *Proceedings of the International Workshop on Architectures (WiA), held in conjunction with the 40th International Symposium on Computer Architecture (ISCA)*, 2013.
