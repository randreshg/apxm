# Workload 9: Conditional Routing

## Purpose

Demonstrate A-PXM's native switch/case control flow for LLM-driven routing. Only the selected branch executes, saving LLM calls compared to preparing all branches.

## What We're Demonstrating

**A-PXM Property**: Dataflow semantics with SWITCH/CASE control flow

A-PXM specifies native control flow operations that integrate with dataflow scheduling. The switch/case construct routes execution based on LLM classification, with compile-time validation of all branches.

```
                 CONDITIONAL ROUTING FLOW
+----------------------------------------------------------------+
|                                                                |
|  +-----------------+                                           |
|  |  INPUT: query   |                                           |
|  +--------+--------+                                           |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | ASK: Classify input     |  <- LLM determines category       |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  |      SWITCH/CASE        |  <- Route based on classification |
|  +--------+----------------+                                   |
|           |                                                    |
|     +-----+-----+-----+                                        |
|     |     |     |     |                                        |
|     v     v     v     v                                        |
|  +-----+ +-----+ +-----+ +---------+                          |
|  |TECH | |CREA | |FACT | | DEFAULT |  <- Only ONE executes     |
|  +--+--+ +--+--+ +--+--+ +----+----+                          |
|     |       |       |         |                                |
|     +-------+-------+---------+                                |
|                 |                                              |
|                 v                                              |
|  +-------------------------+                                   |
|  | ASK: Refine response    |  <- Result flows from branch      |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  |  OUTPUT: response       |                                   |
|  +-------------------------+                                   |
|                                                                |
+----------------------------------------------------------------+
```

### A-PXM Code (workflow.ais)

```
agent RouterAgent {
    @entry flow main(input: str) -> str {
        // 1. Classify the input
        ask("Classify this input into exactly one word: technical, creative, or factual. Input: " + input) -> category

        // 2. Route based on classification (only selected branch runs)
        switch category {
            case "technical" => ask("Provide a detailed technical explanation for: " + input)
            case "creative" => ask("Provide a creative, imaginative response for: " + input)
            case "factual" => ask("Provide accurate factual information for: " + input)
            default => ask("Provide a helpful general response for: " + input)
        } -> routed_response

        // 3. Refine the response with context
        ask("Refine and summarize this response: ", routed_response) -> output
        print("=== Routed Response ===")
        print(output)
        return output
    }
}
```

### LangGraph Comparison

LangGraph uses runtime conditional edges:
- Routing logic in Python functions
- No compile-time validation of branches
- Default case is optional (potential runtime errors)

---

## How to Run

### Quick Run (Compile + Execute)

```bash
cd papers/cf26/benchmarks/workloads/9_conditional_routing

# Compile and execute
apxm execute --emit-metrics metrics.json workflow.ais "How-do-quantum-computers-work"
```

### Compile Only

```bash
# Compile with diagnostics
apxm compile workflow.ais -o workflow.apxmobj --emit-diagnostics diagnostics.json -O1
```

### Run Pre-compiled Artifact

```bash
# Run with metrics export
apxm run --emit-metrics metrics.json workflow.apxmobj "query"
```

### Run Full Benchmark

```bash
# From repo root
apxm workloads run 9_conditional_routing

# With JSON output
apxm workloads run 9_conditional_routing --json
```

---

## Results

### Measured Values

| Metric | Value | Notes |
|--------|-------|-------|
| Total Duration | 6,253ms | 3 LLM calls (classify + branch + refine) |
| Nodes Executed | 11 | Including spliced sub-DAG nodes |
| LLM Calls | 3 | NOT 5 (if all branches executed) |
| Branches Skipped | 3 | Only 1 of 4 branches runs |
| Avg LLM Latency | ~2.1s | Per-call average |
| Optimization Level | O1 | Standard optimization |

### A-PXM vs LangGraph Comparison

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Route definition | **Compile-time switch** | Runtime conditional | Static analysis |
| Route validation | **Static checking** | Runtime errors | Type-checked branches |
| Branch handling | **Native control flow** | Conditional edges | Sub-DAG splicing |
| Default case | **Compiler-enforced** | Optional | Exhaustive matching |
| LLM efficiency | **3 calls** | Same (if optimized) | Only selected branch |
| Output routing | **Token delegation** | Manual state | Dataflow-managed |

---

## Analysis

### Observations

1. **Single branch execution works**: Only the matched case branch executes, demonstrated by 3 LLM calls (classify + 1 branch + refine) instead of 5+ if all branches ran.

2. **Output flows correctly**: The switch output token delegation mechanism ensures the branch result flows to downstream operations (the refine step).

3. **Sub-DAG splicing**: The switch handler dynamically splices the selected case's sub-DAG into the execution graph at runtime.

4. **Compile-time validation**: All branches are type-checked during compilation, ensuring valid operation sequences in each case.

### Fix Applied

The switch handler now uses token delegation to ensure outputs flow correctly:
- Switch marks its output tokens as "delegated by" its node ID
- When switch returns Null, `publish_outputs` skips tokens delegated by that node
- The sub-DAG's inner ask produces the actual value on those tokens
- Downstream operations receive the real result, not Null

### Key Insight

This workload demonstrates that A-PXM's control flow is part of the formal specification, not ad-hoc Python functions. The switch/case construct:

- **Saves LLM calls**: Only the selected branch executes (3 calls vs potential 5)
- **Compile-time safety**: All branches validated before execution
- **Native integration**: Control flow integrates with dataflow scheduling
- **Exhaustive matching**: Compiler enforces default case requirement
- **Correct output routing**: Token delegation ensures results flow to downstream ops
