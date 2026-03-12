---
title: "PXM Foundations: Lessons from Program Execution Models"
description: "How A-PXM draws on decades of computer architecture and compiler research to separate Compute, Memory, State, Optimization, and Scheduling for agent workflows."
---

# PXM Foundations: Lessons from Program Execution Models

A-PXM is not an ad-hoc framework. It is grounded in five decades of research on
how programs should be represented, optimized, and executed. This document traces
each of A-PXM's design decisions back to the program execution models that
inspired them.

---

## 1. Compute Separation

### The Lineage

| Model | How Compute Is Defined | Limitation |
|-------|----------------------|------------|
| **Von Neumann** | Instructions fetched sequentially by program counter | Sequential bottleneck -- independent operations cannot overlap |
| **Dataflow (Manchester Machine, MIT Tagged-Token)** | Operations fire when all operands arrive as tokens | No program counter; parallelism is automatic. Impractical for general programs due to fine granularity |
| **LLVM IR / SSA** | Operations in Static Single Assignment form; each value defined once | Enables optimization (CSE, DCE, constant folding) but still sequentially scheduled |
| **Actor Model (Erlang, Akka)** | Compute = message-driven actors with no shared state | Untyped messages; no compile-time dependency analysis |
| **GPU/CUDA** | Massively parallel warps executing same instruction on different data | SIMT model doesn't fit heterogeneous agent operations |

### How A-PXM Separates Compute

In A-PXM, compute = **typed AIS operations** (ASK, THINK, REASON, INV, PLAN, REFLECT, VERIFY, ...). Each operation is a node in a dataflow graph with:

- **Typed inputs and outputs** -- not opaque function calls
- **Explicit data edges** -- declaring what each operation needs
- **Latency annotations** -- ASK ~1s, THINK ~3s, REASON ~10s

The developer declares *what* to compute. The compiler and runtime decide *when*, *where*, and *in what order*.

**Unlike von Neumann**: independent operations are never artificially sequenced.
**Unlike actors**: operations have typed signatures and explicit dependency edges, not just messages.
**Unlike CUDA**: operations are heterogeneous (LLM calls, tool invocations, memory access), not SIMT.

```ais
// The developer writes this -- sequential-looking code
ask("Research: " + topic) -> research
ask("Critique: " + topic) -> critique
think("Synthesize: " + research + "\n" + critique) -> report

// APXM sees this -- a dataflow graph with two independent roots
//   ask("Research") ──┐
//                     ├── think("Synthesize")
//   ask("Critique") ──┘
// research and critique run in parallel automatically
```

---

## 2. Memory Separation

### The Lineage

| Model | Memory Organization | Limitation |
|-------|-------------------|------------|
| **Von Neumann** | Flat address space; cache hierarchy as optimization | No semantic structure -- everything is bytes at addresses |
| **Dataflow** | Tokens carry data; no addressable memory | Pure dataflow has no persistent state |
| **LLVM** | alloca/load/store with alias analysis; MemorySSA | Powerful optimization but no semantic memory tiers |
| **Actor Model** | Each actor has private state; no shared memory | Isolation is good but cross-actor knowledge sharing is hard |
| **GPU/CUDA** | Registers, shared memory, global memory, constant memory | Explicit hierarchy but programmer must manage manually |
| **Functional (Haskell)** | Immutable values; no mutable state by default | Impractical for agents that must accumulate knowledge |

### How A-PXM Separates Memory

A-PXM provides a **three-tier memory hierarchy** matching how agents actually use context:

| Tier | Purpose | Backing | Access | Analogy |
|------|---------|---------|--------|---------|
| **STM** | Working memory (current session) | In-memory DashMap | ~us | CPU registers/L1 cache |
| **LTM** | Persistent knowledge (facts, preferences) | SQLite + FTS5 + vector embeddings | ~ms | Main memory / database |
| **Episodic** | Execution history (for reflection) | Append-only log | ~ms | Disk / tape archive |

Memory access is through **first-class AIS instructions**:
- `QMEM(query, session, k)` -- read from memory hierarchy
- `UMEM(data, session)` -- write to memory (with optional durability)
- `FENCE()` -- memory barrier ensuring write visibility

**Unlike von Neumann flat memory**: tiers have semantic meaning, not just speed.
**Unlike actor private state**: memory is accessible through typed instructions with clear isolation rules.
**Unlike CUDA manual management**: tier selection can be compiler-assisted (`tier_hint`).

---

## 3. State Separation

### The Lineage

| Model | What Is "State" | Limitation |
|-------|----------------|------------|
| **Von Neumann** | PC + registers + memory; mutable everywhere | State is implicit in memory layout |
| **BDI (Beliefs-Desires-Intentions)** | Agent state decomposed into B, D, I | Conceptual framework, not machine-executable |
| **Actor Model** | Private state per actor; transitions via message processing | No cross-actor state model |
| **Functional** | State threaded explicitly via monads | Difficult for accumulating agent knowledge |
| **FSM** | Explicit states and transitions | Too rigid for dynamic agent behavior |
| **Dataflow** | No PC; state = set of available tokens | No persistent state concept |

### How A-PXM Separates State

The **Agent Abstract Machine (AAM)** formalizes agent state as:

```
AAM = (B, G, C)
  B: Beliefs   -- Map<Key, TypedValue>   -- what the agent knows
  G: Goals     -- PriorityQueue<Goal>    -- what it's trying to achieve
  C: Capabilities -- Map<Name, Signature> -- what it can do (tools)
```

Every AIS instruction is a **deterministic state transition**:

```
d(AAM, Instr) -> AAM'
```

State transitions are explicit operations in the DAG:
- `PLAN(goal)` -- updates Goals
- `UMEM(data)` -- updates Beliefs via memory write
- `REFLECT(trace)` -- self-assessment, may update Goals
- `VERIFY(claim, evidence)` -- validates Beliefs

**Unlike BDI**: APXM's AAM is not just conceptual -- it's machine-readable with typed operations and compiler verification.
**Unlike actors**: state is not opaque private data -- it's structured, typed, and inspectable.
**Unlike FSMs**: state transitions are driven by dataflow, not fixed transition tables.

---

## 4. Optimization Separation

### The Lineage

| Model | Optimization Approach | Key Passes |
|-------|---------------------|------------|
| **LLVM/GCC** | SSA-based pass pipeline | CSE, DCE, inlining, loop unrolling, constant folding |
| **JIT (JVM, V8)** | Profile-guided; speculative inlining; deoptimization | Hot path optimization based on runtime feedback |
| **XLA/TVM/Triton** | Graph-level ML optimization | Operator fusion, memory planning, tiling |
| **MapReduce/Spark** | Stage-level optimization | Predicate pushdown, partition pruning |

### How A-PXM Optimizes Agent Workflows

A-PXM uses **MLIR** as its compiler infrastructure, running optimization passes
specific to agent workloads:

| Pass | What It Does | Impact |
|------|-------------|--------|
| **FuseAskOps** | Merges producer-consumer ASK chains into one API call | 1.29x fewer API calls |
| **CSE** | Eliminates duplicate LLM calls with identical inputs | Saves $ and latency |
| **DCE** | Removes operations whose outputs are never consumed | Leaner graphs |
| **Canonicalization** | Normalizes graph patterns for consistent handling | Enables further passes |

**Why this matters more than traditional compilation:**

In traditional compilers, each "instruction" costs nanoseconds. Optimizing away one
instruction saves nanoseconds.

In A-PXM, each "instruction" (LLM call) costs **seconds and dollars**. Optimizing
away one ASK call saves:
- ~1-4 seconds of latency
- ~$0.01-$0.10 in API cost
- Network round-trip overhead

The economic return on optimization is **orders of magnitude higher** than
traditional compilation.

### Pass Pipeline

```
ApxmGraph → AIS MLIR → Canonicalize → CSE → FuseAskOps → DCE → Canonicalize → Verify → .apxmobj
```

The pipeline iterates until convergence (no pass makes changes).

---

## 5. Scheduling Separation

### The Lineage

| Model | How Execution Order Is Determined | Parallelism Discovery |
|-------|----------------------------------|----------------------|
| **Von Neumann** | Program counter increments sequentially | None (branch prediction only) |
| **Dataflow (Manchester)** | Operations fire when all input tokens arrive | Automatic from graph structure |
| **Out-of-Order (Tomasulo)** | Hardware reorders based on data readiness | Automatic but preserves sequential semantics |
| **Task-Based (Cilk, Tokio)** | Work-stealing over task DAGs | Explicit via spawn/async |
| **MapReduce/Spark** | Stage-based with shuffle barriers | Automatic within stages |
| **Petri Nets** | Transitions fire when all input places have tokens | Automatic from net structure |

### How A-PXM Schedules

A-PXM's scheduler is a **token-counting dataflow machine** inspired by the Manchester Machine:

1. Each operation has a `pending` counter = number of input edges
2. When a predecessor completes, it sends a token; counter decrements
3. Counter reaches zero -> operation fires -> dispatched to executor thread
4. **O(1) readiness detection** -- no graph traversal, just `counter == 0`

```
Per-operation overhead: 7.5 microseconds
  Ready set update:       1.8 us (24%)
  Priority queue dequeue: 0.9 us (12%)
  Dependency resolution:  1.4 us (19%)
  Operation dispatch:     2.4 us (32%)
  Token routing:          1.0 us (13%)
```

**Key properties:**
- **Automatic parallelism**: independent operations run concurrently without `async`/`await`
- **Work stealing**: executor threads steal from peers when idle
- **Priority-based**: critical-path operations fire first
- **Session lane guards**: per-session serialization for multi-user safety

**Unlike Petri nets**: APXM operations are typed with operation-specific executors (LLM executor, tool executor, memory executor).
**Unlike task-based (Tokio)**: the DAG structure IS the parallelism specification -- no `spawn` or `await` needed.
**Unlike out-of-order (Tomasulo)**: APXM doesn't preserve sequential semantics -- there's no sequential program to preserve. The graph IS the program.

---

## Summary: Five Separations

```
┌──────────────────────────────────────────────────────────────────┐
│                        A-PXM Stack                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  COMPUTE        AIS operations (ASK, THINK, REASON, INV, ...)   │
│                 Typed nodes in a dataflow graph                   │
│                                                                   │
│  MEMORY         Three-tier hierarchy (STM / LTM / Episodic)     │
│                 First-class QMEM/UMEM/FENCE instructions         │
│                                                                   │
│  STATE          AAM = (Beliefs, Goals, Capabilities)             │
│                 Typed, inspectable, compiler-verifiable           │
│                                                                   │
│  OPTIMIZATION   MLIR-based pass pipeline                         │
│                 FuseAskOps, CSE, DCE, Canonicalization           │
│                                                                   │
│  SCHEDULING     Token-counting dataflow, O(1) readiness          │
│                 Automatic parallelism, work stealing              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

Each concern is isolated, typed, and independently evolvable. This separation is
what enables A-PXM to optimize agent workflows the way LLVM optimizes machine
code -- with the added economic incentive that each "instruction" costs dollars
and seconds, not nanoseconds.

---

## References

### Compute and Architecture

1. J. von Neumann, "First Draft of a Report on the EDVAC," Moore School of Electrical Engineering, University of Pennsylvania, 1945. Reprinted in *IEEE Annals of the History of Computing*, vol. 15, no. 4, pp. 27–75, 1993. DOI: [10.1109/85.238389](https://doi.org/10.1109/85.238389)

2. J. Backus, "Can Programming Be Liberated from the von Neumann Style? A Functional Style and Its Algebra of Programs," *Communications of the ACM*, vol. 21, no. 8, pp. 613–641, 1978. DOI: [10.1145/359576.359579](https://doi.org/10.1145/359576.359579)

3. J. R. Gurd, C. C. Kirkham, and I. Watson, "The Manchester Prototype Dataflow Computer," *Communications of the ACM*, vol. 28, no. 1, pp. 34–52, 1985. DOI: [10.1145/2465.2468](https://doi.org/10.1145/2465.2468)

4. Arvind and D. E. Culler, "Dataflow Architectures," *Annual Reviews in Computer Science*, vol. 1, pp. 225–253, 1986. DOI: [10.1146/annurev.cs.01.060186.001301](https://doi.org/10.1146/annurev.cs.01.060186.001301)

### Compiler Infrastructure

5. C. Lattner and V. Adve, "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation," in *Proc. CGO '04*, pp. 75–86, IEEE, 2004. DOI: [10.1109/CGO.2004.1281665](https://doi.org/10.1109/CGO.2004.1281665)

6. R. Cytron, J. Ferrante, B. K. Rosen, M. N. Wegman, and F. K. Zadeck, "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph," *ACM TOPLAS*, vol. 13, no. 4, pp. 451–490, 1991. DOI: [10.1145/115372.115320](https://doi.org/10.1145/115372.115320)

7. C. Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation," in *Proc. CGO '21*, pp. 2–14, IEEE, 2021. DOI: [10.1109/CGO51591.2021.9370308](https://doi.org/10.1109/CGO51591.2021.9370308)

### Concurrency and Scheduling

8. C. Hewitt, P. Bishop, and R. Steiger, "A Universal Modular ACTOR Formalism for Artificial Intelligence," in *Proc. IJCAI '73*, pp. 235–245, 1973. URL: [https://ijcai.org/Proceedings/73/Papers/027B.pdf](https://ijcai.org/Proceedings/73/Papers/027B.pdf)

9. G. Agha, *Actors: A Model of Concurrent Computation in Distributed Systems*, MIT Press, 1986.

10. R. M. Tomasulo, "An Efficient Algorithm for Exploiting Multiple Arithmetic Units," *IBM Journal of Research and Development*, vol. 11, no. 1, pp. 25–33, 1967. DOI: [10.1147/rd.111.0025](https://doi.org/10.1147/rd.111.0025)

11. R. D. Blumofe and C. E. Leiserson, "Scheduling Multithreaded Computations by Work Stealing," *JACM*, vol. 46, no. 5, pp. 720–748, 1999. DOI: [10.1145/324133.324234](https://doi.org/10.1145/324133.324234)

12. J. Dean and S. Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters," *Communications of the ACM*, vol. 51, no. 1, pp. 107–113, 2008. DOI: [10.1145/1327452.1327492](https://doi.org/10.1145/1327452.1327492)

13. M. Zaharia et al., "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing," in *Proc. NSDI '12*, USENIX, 2012.

14. C. A. Petri, "Kommunikation mit Automaten," PhD thesis, Universität Hamburg, 1962. English translation: Technical Report RADC-TR-65-377, 1966.

### Agent Models

15. A. S. Rao and M. P. Georgeff, "BDI Agents: From Theory to Practice," in *Proc. ICMAS '95*, pp. 312–319, AAAI Press, 1995.

16. G. R. Gao, R. Patel, and T. St. John, "The Codelet Program Execution Model," presented at *WiA, ISCA '13*, Tel-Aviv, Israel, 2013.

### GPU Computing

17. J. Nickolls and W. J. Dally, "The GPU Computing Era," *IEEE Micro*, vol. 30, no. 2, pp. 56–69, 2010. DOI: [10.1109/MM.2010.41](https://doi.org/10.1109/MM.2010.41)

### Related APXM Documentation

- [Compute in PXMs](compute.md) -- deep dive on compute separation
- [Memory in PXMs](memory.md) -- deep dive on memory separation
- [Scheduling in PXMs](scheduling.md) -- deep dive on scheduling separation
- [Motivation](../concepts/motivation.md) -- 5-expert example showing all separations in action
