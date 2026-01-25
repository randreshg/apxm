# Workload 8: Planning

## Purpose

Demonstrate A-PXM's native PLAN operation for task decomposition. Planning is a first-class operation that integrates with automatic parallelism for step execution.

## What We're Demonstrating

**A-PXM Property**: AIS PLAN operation for native planning

A-PXM specifies an explicit `PLAN` operation that decomposes goals into executable steps. The plan result is a `goal` type that can be passed as context to subsequent operations. The compiler can analyze step dependencies and parallelize independent steps automatically.

```
                      PLANNING FLOW
+----------------------------------------------------------------+
|                                                                |
|  +-----------------+                                           |
|  |  INPUT: goal    |                                           |
|  +--------+--------+                                           |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | PLAN: Decompose goal    |  <- Native planning operation     |
|  |       into steps        |     Returns !ais.goal type        |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | ASK: Execute step 1     |                                   |
|  | ASK: Execute step 2     |  <- Parallel (3 concurrent)       |
|  | ASK: Execute step 3     |     Plan passed as context        |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | ASK: Synthesize results |  <- Combine step outputs          |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  |  OUTPUT: final_result   |                                   |
|  +-------------------------+                                   |
|                                                                |
+----------------------------------------------------------------+
```

### A-PXM Code (workflow.ais)

```
agent PlannerAgent {
    @entry flow main(objective: str) -> str {
        // 1. Generate structured plan using native PLAN operation
        // PLAN returns a goal type with structured steps and priorities
        plan(objective) -> plan_result

        // 2. Execute steps in parallel, using plan as context
        // Goal types can be passed as variadic context to ask operations
        ask("Execute step 1 - create detailed design:", plan_result) -> step1
        ask("Execute step 2 - implement core features:", plan_result) -> step2
        ask("Execute step 3 - testing and refinement:", plan_result) -> step3

        // 3. Merge step results (tokens, not goals)
        merge(step1, step2, step3) -> combined

        // 4. Synthesize final result
        ask("Synthesize these step results into a final deliverable:", combined) -> final_result
        print("=== Planning Result ===")
        print(final_result)
        return final_result
    }
}
```

### LangGraph Comparison

LangGraph requires custom chain-of-thought prompting:
- No native planning operation
- Step extraction requires text parsing
- Manual orchestration of step execution

---

## How to Run

### Quick Run (Compile + Execute)

```bash
cd papers/cf26/benchmarks/workloads/8_planning

# Compile and execute
apxm execute --emit-metrics metrics.json workflow.ais "Build-a-web-scraper"
```

### Compile Only

```bash
# Compile with diagnostics
apxm compile workflow.ais -o workflow.apxmobj --emit-diagnostics diagnostics.json -O1
```

### Run Pre-compiled Artifact

```bash
# Run with metrics export
apxm run --emit-metrics metrics.json workflow.apxmobj "goal"
```

### Run Full Benchmark

```bash
# From repo root
apxm workloads run 8_planning

# With JSON output
apxm workloads run 8_planning --json
```

---

## Results

### Measured Values

| Metric | Value | Notes |
|--------|-------|-------|
| Total Duration | 51,353ms | 5 LLM calls (plan + 3 steps + synthesis) |
| Nodes Executed | 9 | plan, ask×4, merge, print×2, return |
| LLM Calls | 5 | plan + step1 + step2 + step3 + synthesis |
| Max Parallelism | 3 | Steps 1, 2, 3 execute concurrently |
| Avg Parallelism | 2.0 | Sustained parallel execution |
| Total Tokens | 13,467 | 5,984 input + 7,483 output |
| Avg LLM Latency | ~14.9s | Per-call average (large responses) |
| Optimization Level | O1 | Standard optimization |

### A-PXM vs LangGraph Comparison

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Planning | **Native PLAN op** | Custom CoT prompting | First-class operation |
| Step execution | **Automatic parallelism** | Manual orchestration | Dataflow scheduling |
| Step format | **Goal type** | Text parsing required | Type-safe context passing |
| Context passing | **Variadic args** | Manual state | Plan as context to ask() |
| Parallelism | **Compiler-detected** | Manual async | 3 concurrent LLM calls |

---

## Analysis

### Observations

1. **Plan-as-context works end-to-end**: The PLAN operation returns a goal type that is passed as variadic context to the ask operations, enabling structured step execution.

2. **3-way parallelism achieved**: Steps 1, 2, and 3 execute concurrently (max_parallelism: 3), demonstrating automatic parallelization without explicit coordination.

3. **High-quality output**: The synthesis phase combines detailed outputs from all three steps into a comprehensive deliverable (web scraper architecture + code skeletons + configurations).

4. **LLM dominates latency**: Total execution (~51s) is dominated by LLM calls. Each step generates substantial content (avg 1,500+ tokens output per call).

### Fixes Made During This Workload

1. **ArtifactEmitter.cpp**: Removed PlanOp from `inner_plan_supported` condition - PlanOp has its own handler that manages planning differently from ReasonOp.

2. **workflow.ais**: Updated to pass `plan_result` as variadic context to ask operations, properly utilizing the goal type system.

### Key Insight

This workload demonstrates that A-PXM integrates planning with execution through its type system. The PLAN operation produces a goal type that flows as context to subsequent operations, enabling:

- **Type-safe context**: Plan results are passed with proper typing, not string concatenation
- **Automatic parallelism**: Independent steps execute concurrently (3 parallel LLM calls)
- **Structured output**: Plan contains steps with descriptions and priorities in JSON format
- **AAM integration**: Goals are tracked in the agent's state machine for monitoring/debugging
