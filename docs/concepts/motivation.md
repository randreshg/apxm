---
title: "Motivation: Why A-PXM Exists"
description: "A concrete example showing why agent workflows need a formal execution model with clear separation of Compute, Memory, State, Optimization, and Scheduling."
---

# Motivation: Why A-PXM Exists

## The Problem in 30 Seconds

You want 5 expert agents to analyze a business proposal. Here's what happens today:

```python
# Every framework writes this same sequential pipeline
context = retrieve_market_data(proposal)          # 200ms
financial = llm.analyze(context, "financial")     # 4000ms
legal = llm.analyze(context, "legal")             # 3500ms  <-- waits for financial!
technical = llm.analyze(context, "technical")     # 3000ms  <-- waits for legal!
market = llm.analyze(context, "market fit")       # 2500ms  <-- waits for technical!
risk = llm.analyze(context, "risk assessment")    # 3000ms  <-- waits for market!
synthesis = llm.synthesize([financial, legal, technical, market, risk])  # 2000ms
# Total wall time: 18,200ms
```

The five analyses are **completely independent** -- none needs the other's result. But the programming model forces them into a sequential chain. Wall time = sum of all latencies.

## The Same Problem, in A-PXM

### High-Level Intent (What You Want)

```
5 experts analyze a proposal in parallel.
A synthesizer merges their findings.
```

### A-PXM DSL (How You Write It)

```ais
agent ProposalReview {
    @entry flow main(proposal: str) -> str {
        // Retrieve shared context
        ask("Summarize the key points of: " + proposal) -> context

        // 5 independent expert analyses -- APXM runs them in parallel automatically
        ask(backend: "claude", prompt: "Financial analysis: " + context) -> financial
        ask(backend: "claude", prompt: "Legal review: " + context) -> legal
        ask(backend: "claude", prompt: "Technical assessment: " + context) -> technical
        ask(backend: "claude", prompt: "Market fit analysis: " + context) -> market
        ask(backend: "claude", prompt: "Risk assessment: " + context) -> risk

        // Synthesize -- APXM waits for all 5 automatically via dataflow edges
        think(
            prompt: "Synthesize these expert analyses into a decision recommendation:\n"
                    + "Financial: " + financial + "\n"
                    + "Legal: " + legal + "\n"
                    + "Technical: " + technical + "\n"
                    + "Market: " + market + "\n"
                    + "Risk: " + risk,
            budget_tokens: 2000
        ) -> synthesis

        return synthesis
    }
}
```

### What Actually Happens

```
context (4s) ──┬── financial (4s) ──┐
               ├── legal (3.5s)     ├── synthesis (2s)
               ├── technical (3s)   │
               ├── market (2.5s)    │
               └── risk (3s) ──────┘
Total wall time: 4s + max(4, 3.5, 3, 2.5, 3) + 2s = 10s
```

**10 seconds vs 18.2 seconds** -- a 1.82x speedup with zero concurrency code.
The developer wrote sequential-looking code; the compiler extracted the parallelism.

## The Five Separations

A-PXM achieves this by cleanly separating five concerns that current frameworks entangle:

### 1. Compute

**What operations run.** In APXM, compute = AIS operations (ASK, THINK, REASON, INV, etc.). Each is a typed node in a dataflow graph. The developer declares *what* to compute; the system decides *when* and *where*.

| Current Frameworks | A-PXM |
|---|---|
| Compute = Python function calls | Compute = typed AIS instruction nodes |
| Sequencing implicit in code order | Sequencing explicit in graph edges |
| No optimization possible | Compiler fuses, deduplicates, eliminates |

### 2. Memory

**Where knowledge lives.** A-PXM provides a three-tier hierarchy instead of scattered Python variables:

| Tier | Purpose | Backing | Latency |
|------|---------|---------|---------|
| STM | Working memory (current session) | In-memory KV | ~us |
| LTM | Persistent knowledge (facts, preferences) | SQLite + FTS5 | ~ms |
| Episodic | Execution history (for reflection) | Append-only log | ~ms |

Memory access is through typed instructions (QMEM, UMEM), not arbitrary `self.state["key"]` mutations.

### 3. State

**What the agent knows, wants, and can do.** The Agent Abstract Machine (AAM) formalizes state as:

```
AAM = (Beliefs, Goals, Capabilities)
```

- **Beliefs** (B): Typed key-value knowledge (`"location": String("Tokyo")`)
- **Goals** (G): Prioritized objectives (`Goal("answer_query", priority=1)`)
- **Capabilities** (C): What tools are available with typed signatures

Every AIS instruction is a state transition: `d(AAM, Instr) -> AAM'`. No hidden mutations.

### 4. Optimization

**Making it faster and cheaper.** Because operations are in a typed graph (not opaque Python), the compiler can optimize:

| Pass | What It Does | Impact |
|------|-------------|--------|
| FuseAskOps | Merges sequential ASK chains into one API call | 1.29x fewer calls |
| CSE | Eliminates duplicate identical LLM calls | Saves $ and time |
| DCE | Removes operations whose outputs are unused | Leaner graphs |

These are impossible in imperative frameworks where the runtime can't see dependencies.

### 5. Scheduling

**When and where operations execute.** Token-based dataflow scheduling:

1. Each node has a counter = number of input edges
2. When a predecessor completes, it sends a token; counter decrements
3. Counter reaches zero -> node fires
4. O(1) readiness detection, no graph traversal

Independent operations run in parallel *automatically*. No `async`, no `await`, no `Promise.all`.

## The Compiler Analogy

The relationship between agent frameworks and A-PXM mirrors the relationship between programming languages and LLVM:

```
Source Language    Compiler Frontend    IR          Optimizer       Backend
─────────────    ─────────────────    ──          ─────────       ───────
C                Clang                LLVM IR     LLVM passes     x86/ARM
Fortran          Flang                LLVM IR     LLVM passes     x86/ARM
Rust             rustc                LLVM IR     LLVM passes     x86/ARM

Python API       agentmate-py         ApxmGraph   AIS passes      APXM Runtime
Rust API         agentmate-rs         ApxmGraph   AIS passes      APXM Runtime
AIS DSL          AIS parser           ApxmGraph   AIS passes      APXM Runtime
```

Multiple frontends, one optimizing backend. Framework authors focus on developer experience; the runtime handles scheduling, parallelism, and fault recovery.

## Key Results

Across 10 evaluation workloads:

| Metric | Improvement |
|--------|------------|
| Multi-agent latency | Up to **10.37x** reduction |
| Conditional routing | **5.18x** improvement |
| API call reduction | **1.29x** via fusion |
| Lines of code | **7.3x** fewer than LangGraph |
| Error detection | **49x** faster (compile-time vs runtime) |

## Next Steps

- [The Agentic von Neumann Bottleneck](the-problem.md) -- The full analysis of what's broken
- [Architecture](architecture.md) -- How the pieces fit together
- [Agent Abstract Machine](aam.md) -- Formal state model
- [Agent Instruction Set](ais.md) -- The 17 typed operations
- [Dataflow Execution](dataflow-execution.md) -- Token-based scheduling
- [Strategic Analysis](strategic-analysis.md) -- Where A-PXM wins vs alternatives
- [PXM Foundations](../pxm/foundations.md) -- Research lineage behind the five separations

## References

1. J. Backus, "Can Programming Be Liberated from the von Neumann Style?," *Communications of the ACM*, vol. 21, no. 8, pp. 613–641, 1978. DOI: [10.1145/359576.359579](https://doi.org/10.1145/359576.359579)

2. J. R. Gurd, C. C. Kirkham, and I. Watson, "The Manchester Prototype Dataflow Computer," *Communications of the ACM*, vol. 28, no. 1, pp. 34–52, 1985. DOI: [10.1145/2465.2468](https://doi.org/10.1145/2465.2468)

3. C. Lattner and V. Adve, "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation," in *Proc. CGO '04*, IEEE, 2004. DOI: [10.1109/CGO.2004.1281665](https://doi.org/10.1109/CGO.2004.1281665)

4. C. Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation," in *Proc. CGO '21*, IEEE, 2021. DOI: [10.1109/CGO51591.2021.9370308](https://doi.org/10.1109/CGO51591.2021.9370308)

5. G. R. Gao, R. Patel, and T. St. John, "The Codelet Program Execution Model," presented at *WiA, ISCA '13*, Tel-Aviv, Israel, 2013.

6. A. S. Rao and M. P. Georgeff, "BDI Agents: From Theory to Practice," in *Proc. ICMAS '95*, pp. 312–319, AAAI Press, 1995.

7. R. D. Blumofe and C. E. Leiserson, "Scheduling Multithreaded Computations by Work Stealing," *JACM*, vol. 46, no. 5, pp. 720–748, 1999. DOI: [10.1145/324133.324234](https://doi.org/10.1145/324133.324234)

8. R. C. Atkinson and R. M. Shiffrin, "Human Memory: A Proposed System and Its Control Processes," in *The Psychology of Learning and Motivation*, vol. 2, pp. 89–195, Academic Press, 1968.
