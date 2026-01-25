# Workload 6: Tool Invocation

## Purpose

Demonstrate A-PXM's native INV operations with compile-time capability validation. Tools are registered capabilities with typed signatures, not runtime-bound functions.

## What We're Demonstrating

**A-PXM Property**: AIS INV operation with typed tool calls

A-PXM specifies an explicit `INV` (invoke) operation for tool calls. The capability registry validates tool availability and signatures at compile time, preventing runtime "tool not found" errors.

```
                     TOOL INVOCATION FLOW
+----------------------------------------------------------------+
|                                                                |
|  +-----------------+                                           |
|  |  INPUT: query   |                                           |
|  +--------+--------+                                           |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | ASK: Decide which tool  |  <- LLM reasons about tool choice |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | INV: Execute tool       |  <- Invoke registered capability  |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | ASK: Synthesize answer  |  <- LLM reasons about results     |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  |  OUTPUT: final_answer   |                                   |
|  +-------------------------+                                   |
|                                                                |
+----------------------------------------------------------------+
```

### A-PXM Code (workflow.ais)

```
agent ToolAgent {
    @entry flow main(query: str) -> str {
        // 1. Invoke registered capability (mock search tool)
        // The runtime's capability registry validates this at compile time
        inv("search", "{\"query\": \"quantum computing\"}") -> search_results

        // 2. Reason about results to formulate answer
        ask("Given search results: " + search_results + ", answer: " + query) -> answer

        print("Tool Invocation Result")
        print(answer)
        return answer
    }
}
```

### LangGraph Comparison

LangGraph uses runtime tool binding:
- Tools are Python functions decorated with `@tool`
- Validation happens at runtime when tools are called
- Tool discovery requires dynamic lookup

---

## How to Run

### Quick Run (Compile + Execute)

```bash
cd papers/cf26/benchmarks/workloads/6_tool_invocation

# Compile and execute
apxm execute workflow.ais "Search for quantum computing news"
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

### Run LangGraph Comparison

```bash
python workflow.py
```

### Run Full Benchmark

```bash
# From repo root
apxm workloads run 6_tool_invocation

# With JSON output
apxm workloads run 6_tool_invocation --json
```

---

## Results

### Measured Values

| Metric | Value | Notes |
|--------|-------|-------|
| Total Duration | 2,318ms | Includes INV + LLM |
| Nodes Executed | 8 | inv, ask, print×2, const×2, merge, return |
| LLM Calls | 1 | Single ask operation |
| INV Latency | <1ms | MockSearchCapability direct dispatch |
| Optimization Level | O1 | Standard optimization |

### A-PXM vs LangGraph Comparison

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Tool registration | **Compile-time** | Runtime | Key differentiator |
| Validation | **Static type checking** | Runtime errors | Catches invalid tools early |
| Invocation overhead | **~µs** | ~ms | Direct dispatch vs reflection |
| Tool discovery | **Capability registry** | Dynamic lookup | AAM-integrated |
| Error handling | **Typed errors** | Exceptions | Predictable failure modes |

---

## Analysis

### Observations

1. **Capability invocation works end-to-end**: INV operation successfully invokes MockSearchCapability, which returns simulated search results that are then passed to the ask operation.

2. **LLM dominates latency**: Total execution time (~2.3s) is dominated by the LLM call. INV overhead is negligible (<1ms for capability dispatch).

3. **params_json parsing**: Runtime now parses JSON parameters from InvOp and extracts named arguments for capability execution.

### Key Insight

This workload demonstrates that A-PXM treats tool invocation as a first-class operation with typed semantics. The capability registry is part of the AAM specification, enabling:

- **Static validation**: Invalid capability names caught at compile time
- **Typed interfaces**: Capability schemas define expected inputs/outputs
- **AAM integration**: Capability invocations are tracked in the agent's state machine
