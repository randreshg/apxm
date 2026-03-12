---
title: "Codelets"
description: "The fundamental unit of AI work in A-PXM -- self-contained operations that fire when their dependencies are satisfied."
---

# Codelets

A **codelet** is the fundamental unit of work in A-PXM. It encapsulates a single atomic operation -- an LLM call, a tool invocation, a memory access, or a composition of these -- along with its typed inputs, outputs, and firing conditions.

## Runtime Hierarchy

At runtime, codelets are structurally nested:

**Agent -> Flows -> Codelets -> Nodes (AIS instructions)**

- An **Agent** owns one or more named flows.
- A **Flow** contains a `CodeletDag` view (optional) and a lowered `ExecutionDag` view.
- A **Codelet** groups one or more execution nodes into a logical unit.
- A **Node** is the concrete executable AIS instruction scheduled by the runtime.

## Codelet as Compilable Unit

A codelet is the **minimum compilable unit of AI work**. It sits at the critical point on the compilation spectrum:

- **Below a codelet** (individual AIS instructions like ASK, INV, QMEM): interpreted by the runtime with no optimization opportunity. The runtime dispatches them one at a time.
- **At codelet level**: the compiler has enough context to analyze dependencies, estimate latencies, assign priorities, and generate an optimized execution plan.
- **Above a codelet** (CodeletDag): more optimization surface — the compiler can reorder, parallelize, and fuse across codelet boundaries.

The compilation spectrum runs: **single instruction → codelet → CodeletDag**. A codelet is where interpretation ends and optimization begins.

## Three Sources of Codelets

Codelets can originate from three distinct sources, all converging on the same compilation and execution pipeline:

1. **Developer-authored** — Written directly using the builder API (`Codelet::new(...)`) or declared in APXM graph definitions. These are static, checked at compile time, and represent the most common path.

2. **User-authored (JSON)** — Submitted as JSON payloads (e.g., from a web UI or API call) and deserialized via `CodeletDag::from_json()`. This enables non-developer users to define workflows without writing code.

3. **Agent-generated (tool-use / PLAN instruction)** — Created at runtime by agents using the `create_codelet` and `compile_codelet_dag` tools. The PLAN instruction's `codelet_dag` field also allows agents to express multi-step plans as codelets.

Regardless of source, all codelets follow the same path inside a flow:
**CodeletDag → compile → ExecutionDag → Runtime (under Agent/Flow ownership)**.

## Definition

A codelet consists of:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `CodeletID` | Unique identifier within the DAG |
| `op` | `AIS Instruction` | The operation to perform (ASK, INV, MERGE, etc.) |
| `inputs` | `List<(EdgeID, Type)>` | Typed input edges with expected token types |
| `outputs` | `List<(EdgeID, Type)>` | Typed output edges with produced token types |
| `pending` | `AtomicInt` | Counter of unresolved input tokens |
| `priority` | `Int` | Scheduling priority (higher = sooner when multiple codelets are ready) |

Runtime metadata on each codelet:

```rust
pub struct CodeletMetadata {
    pub priority: u32,
    pub expected_output_schema: Option<String>,
}
```

## Firing Rule

A codelet fires when **all** its input tokens are available:

```
fire(codelet) ⟺ codelet.pending == 0
```

This is the dataflow firing rule. Independent codelets with no shared edges fire in parallel without any explicit concurrency primitives.

## Codelet Types

### Primitive Codelets

A primitive codelet wraps a single AIS instruction:

```mermaid
graph LR
    subgraph Agent["Agent: Assistant"]
        subgraph Flow["Flow: answer"]
            subgraph Codelet["Codelet: ASK"]
                direction TB
                In1["prompt: String"]
                In2["ctx: Context"]
                OP["ASK (~1s)"]
                Out["result: String"]
                In1 --> OP
                In2 --> OP
                OP --> Out
            end
        end
    end
```

### Composite Codelets

A composite codelet encapsulates a sub-DAG, presenting a single-node interface to the outer graph. This enables hierarchical composition:

```mermaid
graph TD
    subgraph Agent["Agent: Planner"]
        subgraph Flow["Flow: plan"]
            subgraph Outer["Flow DAG"]
                A["Codelet A"]
                subgraph Composite["Composite Codelet: RetrieveAndReason"]
                    Q["QMEM Node"]
                    T["THINK Node"]
                    Q --> T
                end
                B["Codelet B"]
                A --> Composite
                Composite --> B
            end
        end
    end
```

Composite codelets are the compilation target for reusable agent sub-routines. The compiler can inline them for optimization or keep them opaque for modularity.

## CodeletDag

A **CodeletDag** is a **flow-level** execution graph: a directed acyclic graph of codelets connected by typed edges. It is canonicalized into `ApxmGraph`, then compiled into an artifact DAG for runtime execution.

`CodeletDag` supports JSON serialization via `to_json()` and `from_json()` for storage, transmission, and dynamic loading. Inner-plan linking now normalizes codelet DAGs through the graph/compiler pipeline (no direct DAG-lowering bypass).

```mermaid
graph TD
    subgraph Agent["Agent: Recommender"]
        subgraph Flow["Flow: recommend"]
            C1["Codelet C1: QMEM\n(retrieve user history)"]
            C2["Codelet C2: QMEM\n(retrieve product catalog)"]
            C3["Codelet C3: REASON\n(analyze preferences)"]
            C4["Codelet C4: INV search_api\n(find candidates)"]
            C5["Codelet C5: INV price_api\n(check prices)"]
            C6["Codelet C6: THINK\n(rank results)"]
            C7["Codelet C7: ASK\n(format response)"]
            C8["Codelet C8: UMEM\n(store interaction)"]

            C1 --> C3
            C2 --> C3
            C3 --> C4
            C3 --> C5
            C4 --> C6
            C5 --> C6
            C6 --> C7
            C7 --> C8
        end
    end
```

In this DAG:
- **C1** and **C2** fire in parallel (no shared dependency)
- **C4** and **C5** fire in parallel (both depend only on C3)
- **C6** waits for both C4 and C5 (fan-in synchronization)
- The critical path is C1/C2 -> C3 -> C4/C5 -> C6 -> C7 -> C8

## Codelet Boundary Preservation

When a `CodeletDag` is canonicalized and compiled, each generated execution node records the codelet it originated from in its `NodeMetadata`:

```rust
pub struct NodeMetadata {
    pub priority: u32,
    pub estimated_latency: Option<u64>,
    pub codelet_source_id: Option<CodeletId>,  // ← source codelet
}
```

The `codelet_source_id` field is preserved through the wire format (serialization/deserialization) and survives all the way to runtime execution. This enables:

- **Codelet-level tracing** — aggregate execution metrics (latency, token usage) back to the original codelet, regardless of how many execution nodes it was expanded into.
- **Codelet-level error reporting** — when an execution node fails, the error can be attributed to a specific codelet by name and ID, giving meaningful diagnostics to developers and users.
- **Boundary-aware optimization** — the compiler can reason about codelet boundaries when deciding whether to fuse or split execution nodes.

## Scheduling and Priority

When multiple codelets are ready simultaneously, the scheduler uses the `priority` field to determine execution order. Priority is assigned by the compiler based on:

1. **Critical path membership**: codelets on the critical path get higher priority
2. **Latency budget**: high-latency operations (REASON) are dispatched early to maximize overlap
3. **Fan-out degree**: codelets that unblock many downstream operations are prioritized

## Intellectual Heritage

The codelet abstraction draws on two traditions:

### HPC Dataflow

Gao et al.'s work on dataflow architectures (the Manchester Dataflow Machine, MIT Tagged-Token Architecture) established the principle that computation should be driven by data availability, not program counters. A-PXM applies this principle to AI workloads where the "instructions" are LLM calls with seconds of latency rather than ALU operations with nanoseconds.

### Cognitive Science

Baars and Franklin's **Global Workspace Theory** (GWT) models cognition as a collection of specialized processors ("codelets") that compete for access to a shared workspace. Winning codelets broadcast their results, triggering further processing. A-PXM's codelets mirror this structure: independent specialized operations that produce tokens consumed by downstream processors, with the dataflow graph serving as the global workspace.

## Codelet Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Pending: DAG loaded
    Pending --> Ready: All inputs resolved (counter=0)
    Ready --> Running: Scheduler dispatches
    Running --> Complete: Operation succeeds
    Running --> Failed: Operation fails
    Complete --> [*]: Tokens emitted to successors
    Failed --> Recovery: TRY_CATCH scope active
    Failed --> [*]: No recovery scope
    Recovery --> Pending: Retry with recovery subgraph
```

---

## References

1. G. R. Gao, R. Patel, and T. St. John, "The Codelet Program Execution Model," presented at *WiA, ISCA '13*, Tel-Aviv, Israel, 2013.

2. S. Zuckerman, J. Suetterlein, R. Knauerhase, and G. R. Gao, "Using a 'Codelet' Program Execution Model for Exascale Machines," in *Proc. EXADAPT Workshop, ASPLOS '11*, ACM, 2011. DOI: [10.1145/2000417.2000424](https://doi.org/10.1145/2000417.2000424)

3. J. R. Gurd, C. C. Kirkham, and I. Watson, "The Manchester Prototype Dataflow Computer," *Communications of the ACM*, vol. 28, no. 1, pp. 34–52, 1985. DOI: [10.1145/2465.2468](https://doi.org/10.1145/2465.2468)
