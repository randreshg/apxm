---
title: "Memory in Program Execution Models"
description: "Comparative analysis of how memory is separated and formalized across classical and modern PXMs, and how APXM's tiered memory hierarchy resolves the agentic memory problem."
---

# Memory in Program Execution Models

Memory is the silent axis of every program execution model (PXM). How a model organizes, isolates, and exposes memory determines what optimizations are possible, what concurrency is safe, and what abstractions the programmer can rely on. This section surveys memory formalization across six established PXMs, then shows how APXM introduces a purpose-built memory architecture for agentic AI.

---

## 1. Von Neumann: Flat Address Space

### Organization and Access

The von Neumann model treats memory as a single, flat, byte-addressable store. The CPU fetches instructions and data from the same linear address space through a shared bus. Every memory location is reachable by any instruction that holds its address, with no structural distinction between code, stack, heap, or persistent data at the architectural level.

In practice, modern implementations overlay a **cache hierarchy** (L1/L2/L3) to mask the latency gap between processor speed and DRAM access time. This hierarchy is transparent to the program -- the ISA presents the illusion of uniform-latency memory, and the hardware handles coherence.

### Isolation Guarantees

Minimal. Any pointer can alias any other pointer. The hardware enforces no semantic boundaries between data regions. Operating systems impose virtual memory and page-level protection, but these are layered above the execution model, not intrinsic to it. Within a single address space, all memory is mutable and accessible.

### Optimization Implications

The flat model is simultaneously enabling and constraining. It enables arbitrary data structures (pointer graphs, self-modifying code) but severely limits what the compiler can prove about aliasing. Key consequences:

- **Alias analysis** is undecidable in general; compilers rely on heuristics (type-based alias analysis, restrict qualifiers) to recover optimization opportunities.
- **Cache behavior** is implicit and unpredictable from the source program. The programmer has no formal mechanism to express data locality.
- **Concurrency** requires explicit synchronization (locks, fences, atomics) because any two threads might access overlapping addresses.

The von Neumann bottleneck -- the single bus between CPU and memory -- is the canonical performance limitation. For agentic systems, this model provides no mechanism to distinguish between working context, long-term knowledge, and execution history.

---

## 2. Dataflow Models: Tokens Replace Addresses

### Organization and Access

Pure dataflow architectures eliminate addressable memory entirely. Data moves as **tokens** along edges of a dataflow graph. An operation fires when all its input tokens are present, consumes them, and produces output tokens. There is no program counter and no memory bus -- data flows directly from producer to consumer.

To reintroduce controlled state, dataflow research introduced:

- **I-structures** (Arvind & Thomas): write-once storage cells. A producer writes a value; any number of consumers can read it, but the cell cannot be overwritten. This prevents write-after-write hazards while permitting safe sharing.
- **M-structures** (Barth, Nikhil): mutable cells with built-in synchronization. A read empties the cell (blocking subsequent readers until a new write), enforcing single-reader semantics without external locks.

### Isolation Guarantees

Strong by construction. Since there is no shared address space, operations cannot interfere through aliased writes. I-structures guarantee single-assignment semantics. M-structures provide mutual exclusion at the cell level. Data races are structurally impossible in the pure model.

### Optimization Implications

The absence of a central memory bottleneck enables massive parallelism -- every operation with available inputs can fire simultaneously. The compiler can freely reorder, duplicate, or eliminate operations without worrying about side effects on shared state. However:

- **Stateful computation** (accumulators, caches, mutable collections) requires explicit I/M-structure allocation, adding overhead to patterns that are trivial in von Neumann models.
- **Spatial locality** is lost -- tokens travel wherever the graph topology dictates, not to nearby cache lines.
- **Practical implementations** (Manchester Dataflow Machine, Monsoon) struggled with token management overhead and limited adoption.

---

## 3. LLVM IR: Memory as a Typed Side Effect

### Organization and Access

LLVM IR models memory through three key instructions:

- **`alloca`**: allocates a typed memory slot on the stack frame, returning a pointer.
- **`store`**: writes a typed value through a pointer.
- **`load`**: reads a typed value through a pointer.

All memory access is mediated by typed pointers. LLVM does not expose raw addresses at the IR level -- every pointer carries a type that constrains how the pointed-to memory is interpreted. The `getelementptr` instruction computes derived pointers with type-safe arithmetic.

LLVM's **MemorySSA** pass constructs an SSA-form representation of memory operations, lifting memory reads and writes into a dependency graph analogous to SSA's value-numbering for registers. Each memory write creates a new "memory version," and each read is linked to the specific write it observes. This transforms the implicit, globally-aliased memory model into an explicit, locally-analyzable dependency structure.

### Isolation Guarantees

Moderate. LLVM's type system prevents type-confusion errors, and `alloca`-based stack memory is naturally scoped. However, heap pointers can alias freely. LLVM provides multiple alias analysis implementations:

- **BasicAA**: type-based and scope-based rules (noalias, restrict).
- **ScopedNoAliasAA**: annotation-driven alias sets.
- **TBAA** (Type-Based Alias Analysis): exploits C/C++ strict aliasing rules.
- **Globals analysis**: proves that certain globals cannot alias local allocations.

### Optimization Implications

MemorySSA enables optimizations that would be impossible against a flat memory model:

- **Dead store elimination**: if a store is overwritten before any load observes it, the first store is dead.
- **Load-store forwarding**: if a load follows a store to the same location with no intervening aliasing write, the load can be replaced with the stored value directly.
- **Loop-invariant code motion**: if alias analysis proves a load is not modified inside a loop body, the load can be hoisted.

The key insight: LLVM does not change the von Neumann memory model at the hardware level, but it imposes enough structure at the IR level to recover optimization opportunities that the raw model forecloses.

---

## 4. Actor Model: Private State, Message Passing

### Organization and Access

In the actor model (Hewitt, 1973), each actor encapsulates **private state** that is inaccessible to any other actor. There is no shared memory. Actors communicate exclusively through asynchronous message passing: an actor sends a message to another actor's mailbox, and the recipient processes messages sequentially from its queue.

State mutation happens only within an actor's message handler. The actor may:
1. Send messages to other actors.
2. Create new actors.
3. Update its own private state.
4. Designate the behavior for the next message it receives.

### Isolation Guarantees

Total. No actor can read or write another actor's state. Memory isolation is not an optimization hint or a convention -- it is a structural invariant of the model. This eliminates data races, deadlocks (in the pure model), and all shared-memory concurrency hazards.

### Optimization Implications

The actor model's isolation enables:

- **Location transparency**: actors can be placed on any node in a distributed system without affecting correctness, since they never share memory.
- **Independent garbage collection**: each actor's heap can be collected independently, avoiding stop-the-world pauses.
- **Mailbox optimization**: since messages are the only interface, the runtime can batch, reorder, or compress messages without violating semantics (within delivery guarantees).

The cost is communication overhead. Patterns that are trivial with shared memory (e.g., a shared counter) require message protocols in the actor model. Erlang/OTP and Akka demonstrate that this trade-off is practical for concurrent, fault-tolerant systems.

---

## 5. GPU/CUDA: Explicit Programmer-Managed Hierarchy

### Organization and Access

The CUDA execution model exposes a **multi-level memory hierarchy** that the programmer must manage explicitly:

| Memory Type | Scope | Latency | Size | Lifetime |
|-------------|-------|---------|------|----------|
| **Registers** | Per-thread | ~1 cycle | ~255 x 32-bit | Thread |
| **Shared memory** | Per-block | ~5 cycles | 48-228 KB | Block |
| **L1/L2 cache** | Per-SM / device | ~30-200 cycles | 128 KB / 6 MB | Managed |
| **Global memory** | All threads | ~400-800 cycles | Up to 80 GB | Application |
| **Constant memory** | All threads (read-only) | ~5 cycles (cached) | 64 KB | Application |
| **Texture memory** | All threads (read-only) | ~5 cycles (cached) | Device limit | Application |

Shared memory is a programmer-allocated scratchpad within a thread block. Threads within a block coordinate through shared memory using `__syncthreads()` barriers. Threads in different blocks cannot directly communicate through shared memory.

### Isolation Guarantees

Hierarchical. Registers are private to a thread. Shared memory is private to a block. Global memory is accessible to all threads but requires explicit synchronization. The model does not prevent data races on global memory -- the programmer is responsible for correct synchronization.

### Optimization Implications

The explicit hierarchy forces the programmer to reason about data placement, but enables extreme throughput:

- **Coalesced access**: when threads in a warp access consecutive global memory addresses, the hardware merges requests into a single transaction.
- **Bank conflict avoidance**: shared memory is divided into banks; conflicting accesses serialize. Optimal access patterns are stride-aware.
- **Occupancy tuning**: register and shared memory usage per block determines how many blocks can run concurrently on an SM.

The CUDA model demonstrates that exposing memory hierarchy to the programmer, rather than hiding it behind caches, unlocks performance that transparent caching cannot achieve. The trade-off is programming complexity.

---

## 6. Functional Models (Haskell/ML): Immutable Values

### Organization and Access

Pure functional languages model computation over **immutable values**. There is no mutable state and no concept of memory locations that can be overwritten. Every "variable" is a binding to a value, not an address. Data structures are persistent -- "modifying" a list produces a new list that shares structure with the original.

Memory management is entirely implicit, handled by a garbage collector that the programmer never directly controls. Haskell's laziness adds another dimension: values are allocated as **thunks** (suspended computations) that are evaluated on demand and replaced with their results.

In Haskell, controlled side effects (including mutable references via `IORef` and `STRef`) are sequenced through the `IO` and `ST` monads, which encode effects in the type system. The `ST` monad is particularly notable: it provides safe, locally-scoped mutable state that is guaranteed (by the type system's rank-2 polymorphism) not to escape its scope.

### Isolation Guarantees

Strong in the pure subset. Since values are immutable, there are no aliasing hazards and no data races. Concurrent access to shared immutable data requires no synchronization. The `ST` monad provides thread-local mutable state with compile-time scoping guarantees.

### Optimization Implications

Immutability enables aggressive optimization:

- **Deforestation**: intermediate data structures can be fused away (e.g., `map f . map g` becomes `map (f . g)`) because the compiler knows no external reference to the intermediate list exists.
- **Sharing**: identical subexpressions can be shared safely because values cannot change after creation.
- **Parallel evaluation**: pure expressions can be evaluated in parallel without synchronization (GHC's `par` and `pseq`).
- **Specialization**: the compiler can inline and specialize polymorphic functions aggressively because there are no side-effectful interactions to preserve.

The cost is allocation pressure. Immutable data structures require more allocation than in-place mutation, placing demands on the garbage collector. GHC's generational GC and compact regions are engineering responses to this fundamental trade-off.

---

## APXM: Tiered Memory for Agentic AI

### The Problem with Existing Models

None of the models above address the memory requirements of agentic AI systems:

- **Von Neumann flat memory** treats all data uniformly -- but an agent needs to distinguish between ephemeral working context, durable learned knowledge, and historical execution traces.
- **Dataflow tokens** carry data between operations but provide no mechanism for persistent recall or cross-session continuity.
- **LLVM's typed memory** optimizes compiler transformations but operates at a level of abstraction far below agent-level semantics.
- **Actor private state** isolates agents but provides no structured memory model within an actor.
- **CUDA's explicit hierarchy** is optimized for throughput computing, not semantic knowledge management.
- **Functional immutability** eliminates mutation hazards but provides no model for an agent's evolving beliefs.

APXM introduces a memory architecture purpose-built for agentic computation, drawing lessons from each of these models while addressing their gaps.

### Three-Tier Memory Hierarchy

APXM formalizes agent memory into three tiers, each with distinct semantics, backing stores, and access patterns:

| Tier | Backing Store | Access Latency | Semantics | Mutability |
|------|---------------|----------------|-----------|------------|
| **STM** (Short-Term Memory) | In-memory KV map (`DashMap`) | Microseconds | Working memory / session scratch | Read-write, volatile |
| **LTM** (Long-Term Memory) | SQLite (WAL mode) | Milliseconds | Persistent knowledge, learned facts | Read-write, durable |
| **Episodic** | Append-only log | Milliseconds | Execution traces for reflection | Append-only, durable |

This hierarchy is not an implementation optimization like CPU caches -- it is a **semantic partition** of the agent's state. Each tier answers a different question:

- **STM**: "What am I working on right now?" -- intermediate results, recent tool output, session-scoped variables.
- **LTM**: "What do I know?" -- user preferences, cached facts, learned associations that survive across sessions.
- **Episodic**: "What have I done?" -- timestamped records of every operation, enabling self-reflection and auditing.

The design mirrors cognitive models of human memory (working, semantic, autobiographical), giving flow authors intuitive semantics when deciding where to store data.

### Memory Operations as First-Class AIS Instructions

Unlike frameworks where memory access is buried in library calls or hidden in function closures, APXM elevates memory operations to first-class instructions in the Agent Instruction Set:

| Instruction | Signature | Semantics |
|-------------|-----------|-----------|
| `QMEM` | `(q: String, sid: SessionID, k: Int) -> Value` | Query memory with tiered fallthrough: STM -> LTM -> Episodic |
| `UMEM` | `(data: Value, sid: SessionID) -> Void` | Write to memory; optionally durable (write-through to LTM) |
| `FENCE` | `() -> Void` | Memory barrier: flush pending writes, enforce ordering |

These are **nodes in the dataflow DAG**, not opaque side effects. The compiler can see every memory read and write, reason about their dependencies, and apply transformations:

```mlir
// UMEM produces an ordering edge, not just a side effect
"ais.umem"(%analysis_result, %session) {
  durable = true,
  belief_key = "latest_analysis"
} : (!ais.value, !ais.session_id) -> ()

// FENCE enforces visibility before downstream QMEM
"ais.fence"() : () -> ()

// QMEM is a proper DAG node with typed output
%context = "ais.qmem"(%query, %session, %k) {
  tier_hint = "ltm"
} : (!ais.string, !ais.session_id, i64) -> !ais.value
```

Because memory operations are DAG nodes, the compiler can:

- **Prove independence**: two QMEM operations on different keys can execute in parallel without a FENCE.
- **Eliminate dead stores**: a UMEM whose key is never subsequently read by any QMEM is dead code.
- **Fuse read-modify-write**: a QMEM-ASK-UMEM chain on the same key can be optimized into a single transactional operation.
- **Refuse to fuse across memory boundaries**: the FuseAskOps optimization pass conservatively declines to merge operations when a UMEM or FENCE intervenes, preserving memory ordering guarantees.

### Typed, Tier-Addressed Memory

APXM memory is addressed by **tier and key**, not by raw numeric addresses. The `MemorySpace` enum (`Stm | Ltm | Episodic`) is a compile-time construct that routes operations to the correct backing store through a memory router:

```
Operation -> Memory Router -> { STM (DashMap) | LTM (SQLite) | Episodic (append log) }
```

Each tier enforces its own invariants:

- **STM** is capacity-bounded. When the entry limit is reached, writes fail rather than silently evicting -- the agent must explicitly manage its working set.
- **LTM** is transactional. Writes marked `durable = true` commit to SQLite within a transaction, providing ACID guarantees. SQLite's WAL mode allows concurrent readers alongside a single writer.
- **Episodic** is append-only. It cannot be overwritten or deleted, preserving a complete audit trail. Reads are sequential scans filtered by execution ID or time range.

### Hybrid Search: BM25 + Vector Embeddings

The Facts API, layered on top of LTM, provides structured semantic retrieval through a multi-stage search pipeline:

1. **Backend hybrid search**: FTS5 full-text indexing (BM25 scoring) combined with optional vector cosine similarity when an embedding model is configured.
2. **Temporal decay**: a 30-day half-life function biases results toward recent facts, reflecting the intuition that newer information is more likely to be relevant.
3. **MMR re-ranking**: Maximal Marginal Relevance (lambda = 0.7) diversifies the result set, reducing redundancy when multiple facts cover similar ground.

This pipeline means QMEM is not just a key-value lookup -- it is a semantic retrieval operation that can answer queries like "what do I know about the deploy server?" even when the stored fact uses different wording than the query. Facts are typed records with metadata (tags, source, session, timestamps), enabling filtered retrieval beyond pure text matching.

### Concurrency Model

Each tier has an **independent concurrency mechanism**, preventing cross-tier contention:

| Tier | Mechanism | Contention Profile |
|------|-----------|-------------------|
| STM | Lock-free (`DashMap` sharding) | No blocking; concurrent reads and writes to different keys |
| LTM | SQLite WAL mode | Concurrent readers; single writer with millisecond commits |
| Episodic | Single-writer append; reader snapshots via WAL | No read-write conflicts |

The `FENCE` instruction provides cross-tier ordering when required. Without a FENCE, the runtime makes no guarantee that a write to one tier is visible from another -- this is intentional, matching the dataflow principle that ordering must be explicit in the graph structure.

### Memory Lifecycle

The memory lifecycle is tied to agent execution:

1. **Execution starts**: STM is initialized empty. LTM and Episodic are opened from their on-disk backing stores (or created if this is the first execution).
2. **During execution**: operations issue QMEM and UMEM as DAG nodes fire. The runtime automatically appends episodic entries at operation boundaries for auditability.
3. **Execution ends**: STM is dropped (volatile by design). LTM and Episodic are flushed to disk and survive for future runs.

This lifecycle means STM serves as a session-scoped workspace -- agents do not accumulate unbounded working memory across sessions -- while LTM accumulates knowledge and Episodic accumulates history across the agent's entire lifetime.

### Cross-Agent Memory Isolation

Agents do not share memory. Cross-agent data exchange uses the `COMM` (Communicate) and `FLOW_CALL` instructions, which are explicit message-passing operations in the DAG. This mirrors the actor model's isolation guarantee: an agent's memory tiers (STM, LTM, Episodic) are private to that agent. A receiving agent can write received data into its own memory via UMEM, but the sender's memory is never directly accessible.

The ordering guarantee table makes this explicit:

| Scenario | Guarantee |
|----------|-----------|
| QMEM after UMEM (same key, same subgraph) | Read sees write (data dependency edge) |
| QMEM after UMEM (different keys) | No guarantee without FENCE |
| UMEM after UMEM (same key) | Last-writer-wins within subgraph ordering |
| Cross-agent memory access | Requires COMM protocol; no shared memory |

### Comparative Summary

| Property | Von Neumann | Dataflow | LLVM IR | Actor | CUDA | Functional | **APXM** |
|----------|-------------|----------|---------|-------|------|------------|----------|
| Addressing | Flat byte addresses | No addresses (tokens) | Typed pointers | Per-actor private | Hierarchical spaces | No addresses (values) | **Tier + key** |
| Persistence | None (OS layer) | None | None | Per-actor | None | None (GC) | **LTM (SQLite), Episodic (log)** |
| Isolation | None (same address space) | Total (no shared state) | Alias analysis | Total (mailbox only) | Hierarchical (block/device) | Total (immutable) | **Per-agent; COMM for exchange** |
| Semantic retrieval | No | No | No | No | No | No | **Yes (BM25 + vector + MMR)** |
| Memory ops in IR | load/store | Token routing | alloca/load/store | send/receive | ld/st with space qualifiers | bind/pattern match | **QMEM/UMEM/FENCE** |
| Compiler visibility | Alias analysis required | Full (DAG structure) | MemorySSA | Opaque (per-actor) | Partial (space annotations) | Full (purity) | **Full (DAG nodes)** |
| Concurrency control | Locks, atomics, fences | Structural (firing rule) | Undefined (target-dependent) | Mailbox serialization | `__syncthreads()`, atomics | None needed (immutable) | **Per-tier independent locks + FENCE** |

### Design Rationale

APXM's memory model is the product of three design decisions:

1. **Memory is not a von Neumann store.** Agents do not need byte-addressable flat memory. They need structured tiers that match how intelligent systems actually use context: immediate working memory, accumulated knowledge, and reflective history.

2. **Memory operations are not side effects.** In most agent frameworks, memory reads and writes are buried in Python function calls, invisible to any optimization pass. In APXM, they are typed DAG nodes with explicit data dependencies, enabling the compiler to reason about ordering, eliminate redundancy, and parallelize independent accesses.

3. **Memory has semantics, not just addresses.** The Facts API with hybrid search (BM25 + vector embeddings + temporal decay + MMR) means QMEM can answer *meaning-based* queries, not just exact-key lookups. This is a fundamental departure from every PXM in the table above, where memory access is either address-based or token-based but never semantically-aware.

---

## References

### Von Neumann Architecture

1. J. Backus, "Can Programming Be Liberated from the von Neumann Style? A Functional Style and Its Algebra of Programs," *Communications of the ACM*, vol. 21, no. 8, pp. 613--641, 1978. (ACM Turing Award Lecture.)
2. J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed. Cambridge, MA: Morgan Kaufmann, 2017.

### Dataflow Models

3. Arvind and R. E. Thomas, "I-Structures: An Efficient Data Type for Functional Languages," MIT Laboratory for Computer Science, Technical Memo TM-178, 1980.
4. P. S. Barth and R. S. Nikhil, "M-Structures: Extending a Parallel, Non-Strict Functional Language with State," in *Proceedings of the 5th ACM Conference on Functional Programming Languages and Computer Architecture (FPCA '91)*, pp. 538--568, 1991.
5. J. R. Gurd, C. C. Kirkham, and I. Watson, "The Manchester Prototype Dataflow Computer," *Communications of the ACM*, vol. 28, no. 1, pp. 34--52, 1985.

### LLVM IR and Compiler Memory Representations

6. C. Lattner and V. Adve, "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation," in *Proceedings of the International Symposium on Code Generation and Optimization (CGO '04)*, pp. 75--86, 2004.
7. D. Novillo, "Memory SSA -- A Unified Approach for Sparsely Representing Memory Operations," in *Proceedings of the GCC Developers' Summit*, Ottawa, Canada, 2007.
8. R. Cytron, J. Ferrante, B. K. Rosen, M. N. Wegman, and F. K. Zadeck, "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph," *ACM Transactions on Programming Languages and Systems*, vol. 13, no. 4, pp. 451--490, 1991.

### Actor Model

9. C. Hewitt, P. Bishop, and R. Steiger, "A Universal Modular ACTOR Formalism for Artificial Intelligence," in *Proceedings of the 3rd International Joint Conference on Artificial Intelligence (IJCAI '73)*, pp. 235--245, 1973.
10. G. Agha, *Actors: A Model of Concurrent Computation in Distributed Systems*. Cambridge, MA: MIT Press, 1986.
11. J. Armstrong, "Making Reliable Distributed Systems in the Presence of Software Errors," Ph.D. dissertation, Royal Institute of Technology (KTH), Stockholm, Sweden, 2003.

### GPU/CUDA Programming Model

12. NVIDIA Corporation, "CUDA C++ Programming Guide," NVIDIA Developer Documentation, v12.x (current). Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
13. J. Nickolls, I. Buck, M. Garland, and K. Skadron, "Scalable Parallel Programming with CUDA," *ACM Queue*, vol. 6, no. 2, pp. 40--53, 2008.
14. D. B. Kirk and W. W. Hwu, *Programming Massively Parallel Processors: A Hands-on Approach*, 3rd ed. Cambridge, MA: Morgan Kaufmann, 2016.

### Functional Programming Models

15. S. L. Peyton Jones, *The Implementation of Functional Programming Languages*. Englewood Cliffs, NJ: Prentice Hall, 1987.
16. P. Wadler, "Deforestation: Transforming Programs to Eliminate Trees," *Theoretical Computer Science*, vol. 73, no. 2, pp. 231–248, 1990. DOI: [10.1016/0304-3975(90)90147-A](https://doi.org/10.1016/0304-3975(90)90147-A)
17. J. Launchbury, "A Natural Semantics for Lazy Evaluation," in *Proceedings of the 20th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL '93)*, pp. 144--154, 1993.

### Cognitive Memory Models

18. R. C. Atkinson and R. M. Shiffrin, "Human Memory: A Proposed System and Its Control Processes," in *The Psychology of Learning and Motivation*, vol. 2, K. W. Spence and J. T. Spence, Eds. New York: Academic Press, 1968, pp. 89--195.
19. E. Tulving, "Episodic and Semantic Memory," in *Organization of Memory*, E. Tulving and W. Donaldson, Eds. New York: Academic Press, 1972, pp. 381--403.
20. A. D. Baddeley, *Working Memory*. Oxford: Oxford University Press, 1986.

### Information Retrieval

21. S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333--389, 2009.
22. J. Carbonell and J. Goldstein, "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries," in *Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 335--336, 1998.
