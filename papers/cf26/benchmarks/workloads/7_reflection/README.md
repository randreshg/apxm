# Workload 7: Reflection

## Purpose

Demonstrate A-PXM's native REFL operation for structured self-analysis. Reflection is a first-class operation, not a prompting pattern.

## What We're Demonstrating

**A-PXM Property**: AIS REFL operation for native reflection

A-PXM specifies an explicit `REFL` (reflect) operation that produces structured self-critique. The output format is defined by the operation semantics, enabling compiler optimizations like fusing reflection with subsequent improvements.

```
                     REFLECTION FLOW
+----------------------------------------------------------------+
|                                                                |
|  +-----------------+                                           |
|  |  INPUT: task    |                                           |
|  +--------+--------+                                           |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | ASK: Initial attempt    |  <- First solution attempt        |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | REFL: Self-critique     |  <- Native reflection operation   |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | ASK: Improve answer     |  <- Apply reflection feedback     |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  |  OUTPUT: improved       |                                   |
|  +-------------------------+                                   |
|                                                                |
+----------------------------------------------------------------+
```

### A-PXM Code (workflow.ais)

```
agent ReflectiveAgent {
    @entry flow main(task: str) -> str {
        // 1. Initial attempt at solving the task
        ask("Solve this task: " + task) -> initial_answer

        // 2. Reflect on the answer (native reflect operation)
        reflect(initial_answer) -> reflection

        // 3. Improve based on reflection feedback
        ask("Given this feedback: " + reflection + ", improve the answer to: " + task) -> improved_answer
        return improved_answer
    }
}
```

### LangGraph Comparison

LangGraph requires custom prompting for reflection:
- No native reflection operation
- Output format varies by prompt design
- Manual parsing of reflection results

---

## How to Run

### Prerequisites

```bash
# Start Ollama (local LLM backend)
ollama serve
ollama pull phi3:mini

# Install Python dependencies
pip install langgraph langchain-ollama

# Build A-PXM compiler (from repo root)
apxm build
```

### Quick Run (Compile + Execute)

```bash
cd papers/cf26/benchmarks/workloads/7_reflection

# Compile and execute
apxm execute workflow.ais "Write a sorting algorithm"
```

### Compile Only

```bash
# Compile with diagnostics
apxm compile workflow.ais -o workflow.apxmobj --emit-diagnostics diagnostics.json -O1
```

### Run Pre-compiled Artifact

```bash
# Run with metrics export
apxm run --emit-metrics metrics.json workflow.apxmobj "task"
```

### Run LangGraph Comparison

```bash
python workflow.py
```

### Run Full Benchmark

```bash
# From repo root
apxm workloads run 7_reflection

# With JSON output
apxm workloads run 7_reflection --json
```

---

## Results

### Measured Values

| Metric | Value | Notes |
|--------|-------|-------|
| Total Duration | 7,701ms | 3 LLM calls (ask + reflect + ask) |
| Nodes Executed | 11 | ask×2, reflect, print×2, const×3, merge×2, return |
| LLM Calls | 3 | Initial attempt, reflection, improvement |
| Avg LLM Latency | ~2.5s | Per-call average |
| Optimization Level | O1 | Standard optimization |

### A-PXM vs LangGraph Comparison

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Reflection support | **Native REFL op** | Custom prompting | First-class operation |
| Structured output | **Built-in format** | Manual parsing | JSON schema defined |
| Context handling | **Input operands** | Manual state | Inputs flow through DAG |
| Iteration control | Compiler-managed | Runtime loops | |
| Cost optimization | Fusible prompts | Separate calls | Optimization potential |

---

## Analysis

### Observations

1. **Reflection works end-to-end**: The REFLECT operation successfully analyzes the initial answer and provides structured feedback that improves the final output.

2. **Three LLM calls as expected**: The workflow executes ask→reflect→ask, with each LLM call taking ~2.5s average.

3. **Input-based context**: The fix enables reflect() to use its input operands as the content to analyze, rather than requiring a static prompt attribute.

### Fix Made During This Workload

1. **reflect.rs**: Updated REFLECT handler to:
   - Accept input operands as the content to reflect on
   - Fall back to `trace_id` or `prompt` attributes
   - Generate a sensible default prompt when no explicit prompt is provided

### Key Insight

This workload demonstrates that A-PXM elevates common agent patterns (like reflection) to first-class operations. The REFLECT operation:

- **Native semantics**: Defined operation type with specific behavior
- **Structured output**: Returns JSON with insights, patterns, recommendations
- **Dataflow integration**: Inputs flow naturally through the DAG
- **Compiler visibility**: Enables future optimizations (e.g., fusing reflect+ask)
