---
title: "Compute in Program Execution Models"
description: "How compute is defined and separated across classical and modern PXMs, and how APXM's typed AIS operations formalize agentic computation."
---

# Compute in Program Execution Models

The most fundamental question a Program Execution Model answers is: *what is a unit of compute?* The answer determines what can be optimized, what can be parallelized, and what the programmer must reason about. This document examines how six foundational PXMs define compute, then shows how APXM introduces typed, heterogeneous operations purpose-built for agentic AI.

---

## 1. Von Neumann: Instructions Fetched by Program Counter

### Definition of Compute

In the von Neumann model, a unit of compute is a single **machine instruction** fetched from memory at the address held by the program counter (PC). Instructions operate on registers and memory: load, store, add, branch. The instruction set architecture (ISA) defines a fixed vocabulary of operations, each with deterministic semantics and predictable latency (single-digit nanoseconds).

Instructions compose into programs through sequential execution: the PC increments after each instruction unless a branch redirects it. The entire computational model reduces to:

```
while true:
    instruction = memory[PC]
    execute(instruction)
    PC = next(PC, instruction)
```

### Properties

- **Granularity**: Sub-microsecond. Each instruction performs a single arithmetic, logical, or memory operation.
- **Typing**: Weak. The ISA distinguishes integer, floating-point, and address operations, but the data in registers and memory is largely untyped bytes.
- **Composition**: Sequential by default. Parallelism requires explicit multi-threading or SIMD annotations.
- **Cost uniformity**: Most instructions cost roughly the same (1-5 cycles), except memory accesses which vary with cache behavior.

### Limitation for Agents

Agent "instructions" (LLM calls, tool invocations, memory queries) have latencies measured in **seconds**, not nanoseconds -- six to nine orders of magnitude slower. They are heterogeneous: an LLM reasoning call has fundamentally different resource requirements, failure modes, and cost ($0.01-$0.10 per call) than a tool invocation or a memory read. The von Neumann instruction model -- uniform, fine-grained, sequential -- cannot capture these characteristics.

---

## 2. Dataflow: Operations That Fire on Token Arrival

### Definition of Compute

In the dataflow model (Manchester Machine, MIT Tagged-Token Architecture), a unit of compute is an **operation node** in a dataflow graph. An operation has typed input arcs and output arcs. It fires -- becomes eligible for execution -- when **all input tokens** have arrived. Upon firing, it consumes its input tokens and produces output tokens on its outgoing arcs.

```
operation(inputs) -> outputs
    precondition: all input tokens present
    postcondition: output tokens emitted on all outgoing arcs
```

There is no program counter. Operations do not have addresses in a sequential instruction stream. The graph topology IS the program.

### Properties

- **Granularity**: Operation-level (arithmetic operations in hardware dataflow; coarser in software implementations).
- **Typing**: Tokens carry type tags in tagged-token architectures. Operations can check that input tokens have the expected types.
- **Composition**: Structural. Operations compose by wiring output arcs to input arcs. The resulting graph implicitly encodes both data dependencies and available parallelism.
- **Parallelism**: Automatic. Independent operations (no shared input tokens) fire concurrently without programmer annotation.

### Influence on APXM

APXM adopts the dataflow firing rule directly: operations execute when all input data is available. The critical difference is that APXM operations are **coarse-grained and heterogeneous** (LLM calls, tool invocations, memory accesses), whereas Manchester Machine operations were fine-grained and homogeneous (arithmetic). This coarse granularity makes the scheduling overhead (7.5 microseconds per operation) negligible against operation latencies of milliseconds to seconds.

---

## 3. LLVM IR / SSA: Operations in Static Single Assignment Form

### Definition of Compute

In LLVM IR, a unit of compute is an **SSA instruction** that defines a single value. Each value is defined exactly once (Static Single Assignment), and uses of that value reference the definition directly. This eliminates the need for register allocation at the IR level and makes data flow explicit.

```llvm
%sum = add i32 %a, %b       ; defines %sum
%result = mul i32 %sum, %c   ; uses %sum, defines %result
```

Instructions are organized into **basic blocks** (straight-line sequences with a single entry and exit). Basic blocks are organized into **functions**. Control flow between basic blocks uses `br` (branch) and `phi` (merge) instructions.

### Properties

- **Granularity**: Single operations (add, mul, load, store, call), similar to von Neumann but at a higher abstraction level.
- **Typing**: Strong. Every value has a precise type (`i32`, `float`, `ptr`, `{i32, float}`). Type mismatches are compilation errors.
- **Composition**: SSA def-use chains form a value dependency graph within basic blocks. Across blocks, control flow graph (CFG) and phi nodes connect definitions.
- **Optimization surface**: SSA form enables classical optimizations -- CSE (Common Subexpression Elimination), DCE (Dead Code Elimination), constant folding, inlining -- because each value has a single definition point and all uses are visible.

### Influence on APXM

APXM's MLIR-based compiler IR inherits SSA's optimization properties. AIS operations in MLIR are SSA values: each ASK, THINK, or REASON node defines a single result that downstream operations reference. This enables APXM to apply CSE (eliminating duplicate LLM calls with identical inputs), DCE (removing operations whose outputs are never consumed), and FuseAskOps (merging producer-consumer ASK chains) -- optimizations impossible in frameworks where LLM calls are opaque function calls with no formal data flow.

---

## 4. Actor Model: Message-Driven Compute

### Definition of Compute

In the actor model (Hewitt, 1973; Agha, 1986), a unit of compute is a **message handler** -- the code an actor executes in response to receiving a message. An actor processes one message at a time, sequentially, from its mailbox. In response to a message, an actor may:

1. Send messages to other actors
2. Create new actors
3. Update its own private state
4. Designate the behavior for the next message

```erlang
% Erlang actor (process)
loop(State) ->
    receive
        {request, From, Data} ->
            Result = process(Data, State),
            From ! {response, Result},
            loop(State#{last => Data})
    end.
```

### Properties

- **Granularity**: Message handler execution (variable -- from microseconds to seconds depending on the handler logic).
- **Typing**: Untyped in classical formulations. Messages are arbitrary data. Erlang uses pattern matching for runtime type discrimination. Akka Typed adds compile-time message type checking.
- **Composition**: Actors compose by sending messages. The resulting communication pattern is implicit in the code, not declared as a graph.
- **Concurrency**: Natural. Actors with independent mailboxes execute concurrently. No shared state, no locks.

### Limitation for Agents

The actor model provides excellent concurrency and isolation but lacks two properties critical for agent optimization:

1. **No compile-time dependency graph**: Since actors communicate via runtime messages, no compiler can analyze the full communication pattern statically. APXM's dataflow graph makes all dependencies explicit before execution begins.
2. **Untyped operations**: The actor model treats all computation uniformly as "handle this message." APXM distinguishes between LLM calls (ASK, THINK, REASON), tool invocations (INV), memory operations (QMEM, UMEM), and control flow (BRANCH, SWITCH), enabling operation-type-specific optimization and scheduling.

---

## 5. GPU/CUDA: SIMT Parallel Compute

### Definition of Compute

In CUDA's SIMT (Single Instruction, Multiple Threads) model, a unit of compute is a **kernel invocation** that launches thousands of lightweight threads organized into a grid of blocks. All threads in a warp (32 threads) execute the same instruction simultaneously on different data elements.

```cuda
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// Launch: add<<<numBlocks, blockSize>>>(a, b, c, n);
```

### Properties

- **Granularity**: Two levels -- the kernel (coarse, launched from host) and the thread (fine, executing within the kernel). Thousands to millions of threads per kernel.
- **Typing**: C++ types within kernels. Memory spaces (global, shared, constant, texture) are type-qualified.
- **Composition**: Kernels compose sequentially on the host timeline, or concurrently via CUDA streams. Within a kernel, threads compose via shared memory and synchronization barriers.
- **Parallelism**: Massive data parallelism within kernels. All threads execute the same code on different data.

### Limitation for Agents

SIMT is designed for **homogeneous** data-parallel workloads: the same operation applied to millions of data elements. Agent workflows are **heterogeneous**: a REASON call, a tool invocation, and a memory query have nothing in common except being part of the same workflow. APXM's typed, heterogeneous operation model handles this naturally -- each operation type has its own executor with operation-specific scheduling strategies.

---

## 6. Functional: Pure Expressions

### Definition of Compute

In pure functional models (lambda calculus, Haskell, ML), a unit of compute is an **expression evaluation**. Every computation is the evaluation of an expression to a value. There are no statements, no side effects, and no mutable state. Functions are first-class values. Composition is function application and combination.

```haskell
-- Every computation is an expression
result = synthesize (research topic) (critique topic)
  where
    research t = ask ("Research: " ++ t)
    critique t = ask ("Critique: " ++ t)
    synthesize r c = think ("Synthesize: " ++ r ++ "\n" ++ c)
```

### Properties

- **Granularity**: Expression-level. Lazy evaluation (Haskell) defers computation until the value is needed, enabling demand-driven evaluation.
- **Typing**: Strong, static, with type inference (Hindley-Milner). Types are checked at compile time and carry no runtime overhead.
- **Composition**: Function application (`f x`), function composition (`f . g`), higher-order functions (`map f xs`). Monads sequence effectful computations.
- **Referential transparency**: An expression can be replaced with its value without changing program behavior. This enables equational reasoning and aggressive optimization (deforestation, fusion, specialization).

### Influence on APXM

APXM borrows the functional model's key insight: **making data flow explicit enables optimization**. In Haskell, referential transparency lets the compiler fuse `map f . map g` into `map (f . g)`. In APXM, explicit data edges in the DAG let the compiler fuse `ASK(prompt1) -> ASK(prompt2)` into a single `ASK(prompt1 + prompt2)` via the FuseAskOps pass. Both achieve the same result -- eliminating intermediate computations -- through the same mechanism: making data dependencies first-class and visible to the optimizer.

---

## Comparative Summary

| Property | Von Neumann | Dataflow | LLVM IR | Actor | GPU/CUDA | Functional | **APXM** |
|----------|-------------|----------|---------|-------|----------|------------|----------|
| **Unit of compute** | Instruction | Operation node | SSA instruction | Message handler | Kernel/thread | Expression | **AIS operation** |
| **Granularity** | Nanoseconds | Nanoseconds (HW) | Nanoseconds | Variable | Variable | Variable | **Seconds** |
| **Type system** | Weak (ISA types) | Tagged tokens | Strong (IR types) | Untyped/runtime | C++ types | Strong (HM) | **AIS op categories** |
| **Heterogeneity** | Moderate (int/float/mem) | Low (arithmetic) | Moderate | High (any handler) | Low (SIMT) | High | **High (LLM/tool/mem/ctrl)** |
| **Optimization** | Hardware (OoO, speculation) | Graph structure | SSA passes (CSE, DCE) | None (opaque) | Warp scheduling | Fusion, deforestation | **MLIR passes (Fuse, CSE, DCE)** |
| **Parallelism** | Manual | Structural | Manual (threads) | Structural (actors) | Data-parallel (SIMT) | Implicit (purity) | **Structural (DAG)** |
| **Cost per op** | ~1ns, ~free | ~1ns | ~1ns | ~1us | ~1ns per thread | ~1ns | **~1-10s, $0.01-$0.10** |

---

## APXM: Typed Heterogeneous Operations for Agentic Compute

### The Core Insight

The fundamental observation driving APXM's compute model is that **agent operations are the most expensive "instructions" ever scheduled**. A single ASK call costs seconds of wall-clock time and cents of real money. This inverts the optimization calculus: in traditional compilation, eliminating one instruction saves nanoseconds. In APXM, eliminating one ASK call saves seconds and dollars. The economic return on optimization is orders of magnitude higher.

### AIS Operations as Typed Compute Primitives

APXM defines compute through the **Agent Instruction Set (AIS)** -- a set of typed operations organized into categories:

| Category | Operations | Semantics |
|----------|-----------|-----------|
| **LLM** | ASK, THINK, REASON, PLAN, REFLECT, VERIFY | Typed LLM calls with distinct reasoning characteristics and latency profiles |
| **Tool** | INV | External tool invocation with typed parameter schemas |
| **Memory** | QMEM, UMEM, FENCE | Structured memory access across three tiers |
| **Control** | BRANCH, SWITCH | Conditional routing in the dataflow graph |
| **Sync** | MERGE, WAIT_ALL | Parallel path synchronization |
| **Communication** | COMM, FLOW_CALL | Cross-agent messaging and sub-flow invocation |

Each operation is a node in the dataflow DAG with:
- **Typed inputs and outputs**: not opaque function calls, but values with known types
- **Explicit data edges**: declaring exactly what each operation needs
- **Latency annotations**: ASK ~1s, THINK ~3s, REASON ~10s, enabling priority scheduling
- **Operation-specific semantics**: the runtime dispatches each operation to a category-specific executor

### What Makes This Different

**Unlike von Neumann instructions**: AIS operations are coarse-grained (seconds, not nanoseconds), heterogeneous (LLM calls are fundamentally different from tool invocations), and expensive (dollars, not free). Optimization matters orders of magnitude more.

**Unlike dataflow operations**: AIS operations are typed by category, not uniform arithmetic nodes. The scheduler uses operation type information (latency budgets, resource requirements) to make informed dispatch decisions.

**Unlike actor messages**: AIS operations exist in a statically analyzable DAG. The compiler can see every operation, every dependency, and every opportunity for optimization before execution begins.

**Unlike CUDA kernels**: AIS operations are heterogeneous by design. There is no assumption that operations perform similar work on different data. Each operation type has its own executor with specialized handling.

**Unlike functional expressions**: AIS operations have explicit side effects (memory writes, tool invocations) that are visible in the DAG as typed nodes, not hidden behind monadic wrappers. The compiler reasons about effects through graph structure, not type-level effect tracking.

### The Optimization Payoff

Because AIS operations are typed nodes in an SSA-style dataflow graph, APXM can apply compiler optimizations that no other agent framework supports:

| Optimization | What It Does | Impact |
|-------------|-------------|--------|
| **FuseAskOps** | Merges producer-consumer ASK chains into single API calls | 1.29x fewer API calls |
| **CSE** | Eliminates duplicate LLM calls with identical inputs | Saves cost and latency |
| **DCE** | Removes operations whose outputs are never consumed | Leaner execution graphs |
| **Canonicalization** | Normalizes patterns for consistent optimization | Enables further passes |

The compiler pipeline iterates these passes until convergence:

```
ApxmGraph -> AIS MLIR -> Canonicalize -> CSE -> FuseAskOps -> DCE -> Canonicalize -> Verify -> .apxmobj
```

Each eliminated operation saves seconds and dollars -- not nanoseconds. This is why compute separation matters more for agents than for any previous computing paradigm.

---

## References

References are grouped by the PXM section they support.

### Von Neumann Architecture

1. J. von Neumann, "First Draft of a Report on the EDVAC," Moore School of Electrical Engineering, University of Pennsylvania, 1945. Reprinted in *IEEE Annals of the History of Computing*, vol. 15, no. 4, pp. 27–75, 1993. DOI: [10.1109/85.238389](https://doi.org/10.1109/85.238389)

2. J. Backus, "Can Programming Be Liberated from the von Neumann Style? A Functional Style and Its Algebra of Programs," *Communications of the ACM*, vol. 21, no. 8, pp. 613–641, 1978. DOI: [10.1145/359576.359579](https://doi.org/10.1145/359576.359579)

### Dataflow / Manchester Machine

3. J. R. Gurd, C. C. Kirkham, and I. Watson, "The Manchester Prototype Dataflow Computer," *Communications of the ACM*, vol. 28, no. 1, pp. 34–52, 1985. DOI: [10.1145/2465.2468](https://doi.org/10.1145/2465.2468)

4. Arvind and D. E. Culler, "Dataflow Architectures," *Annual Reviews in Computer Science*, vol. 1, pp. 225–253, 1986. DOI: [10.1146/annurev.cs.01.060186.001301](https://doi.org/10.1146/annurev.cs.01.060186.001301)

5. J. B. Dennis, "First Version of a Data Flow Procedure Language," in *Programming Symposium*, Lecture Notes in Computer Science, vol. 19, pp. 362–376, Springer, 1974. DOI: [10.1007/3-540-06859-7_145](https://doi.org/10.1007/3-540-06859-7_145)

### LLVM IR / SSA Form

6. C. Lattner and V. Adve, "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation," in *Proceedings of the 2004 International Symposium on Code Generation and Optimization (CGO '04)*, pp. 75–86, IEEE, 2004. DOI: [10.1109/CGO.2004.1281665](https://doi.org/10.1109/CGO.2004.1281665)

7. R. Cytron, J. Ferrante, B. K. Rosen, M. N. Wegman, and F. K. Zadeck, "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph," *ACM Transactions on Programming Languages and Systems*, vol. 13, no. 4, pp. 451–490, 1991. DOI: [10.1145/115372.115320](https://doi.org/10.1145/115372.115320)

8. C. Lattner, M. Amini, U. Bondhugula, A. Cohen, A. Davis, J. Pienaar, R. Riddle, T. Shpeisman, N. Vasilache, and O. Zinenko, "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation," in *Proceedings of the 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO '21)*, pp. 2–14, IEEE, 2021. DOI: [10.1109/CGO51591.2021.9370308](https://doi.org/10.1109/CGO51591.2021.9370308)

### Actor Model

9. C. Hewitt, P. Bishop, and R. Steiger, "A Universal Modular ACTOR Formalism for Artificial Intelligence," in *Proceedings of the 3rd International Joint Conference on Artificial Intelligence (IJCAI '73)*, pp. 235–245, Morgan Kaufmann, 1973. URL: [https://ijcai.org/Proceedings/73/Papers/027B.pdf](https://ijcai.org/Proceedings/73/Papers/027B.pdf)

10. G. Agha, *Actors: A Model of Concurrent Computation in Distributed Systems*, MIT Press, 1986. ISBN: 978-0-262-01092-4.

### GPU / CUDA

11. J. Nickolls and W. J. Dally, "The GPU Computing Era," *IEEE Micro*, vol. 30, no. 2, pp. 56–69, 2010. DOI: [10.1109/MM.2010.41](https://doi.org/10.1109/MM.2010.41)

12. NVIDIA Corporation, *CUDA C++ Programming Guide*, v12.x, 2024. URL: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Functional Programming

13. A. Church, *The Calculi of Lambda Conversion*, Annals of Mathematics Studies, no. 6, Princeton University Press, 1941. ISBN: 978-0-691-08394-0.

14. P. Wadler, "Deforestation: Transforming Programs to Eliminate Trees," *Theoretical Computer Science*, vol. 73, no. 2, pp. 231–248, 1990. DOI: [10.1016/0304-3975(90)90147-A](https://doi.org/10.1016/0304-3975(90)90147-A)

15. S. L. Peyton Jones, *The Implementation of Functional Programming Languages*, Prentice Hall, 1987. ISBN: 978-0-13-453325-9. URL: [https://www.microsoft.com/en-us/research/publication/the-implementation-of-functional-programming-languages/](https://www.microsoft.com/en-us/research/publication/the-implementation-of-functional-programming-languages/)

### APXM and the Codelet Model

16. G. R. Gao, R. Patel, and T. St. John, "The Codelet Program Execution Model," presented at *Workshop on Innovative Architecture (WiA), 40th International Symposium on Computer Architecture (ISCA)*, Tel-Aviv, Israel, 2013.

17. APXM Documentation: [Agent Instruction Set (AIS)](../concepts/ais.md) -- full specification of typed AIS operations.

18. APXM Documentation: [Compiler Pipeline](../compiler/overview.md) -- MLIR-based compilation and optimization passes for agentic dataflow graphs.

19. APXM Documentation: [Dataflow Execution](../concepts/dataflow-execution.md) -- how the runtime schedules AIS operations using the dataflow firing rule.
