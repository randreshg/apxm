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
|  | RSN: Initial attempt    |  <- First solution attempt        |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | REFL: Self-critique     |  <- Native reflection operation   |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | RSN: Improve answer     |  <- Apply reflection feedback     |
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
        rsn("Solve this task: " + task) -> initial_answer

        // 2. Reflect on the answer (native reflect operation)
        reflect(initial_answer) -> reflection

        // 3. Improve based on reflection feedback
        rsn("Given this feedback: " + reflection + ", improve the answer to: " + task) -> improved_answer
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
ollama pull gpt-oss:20b-cloud

# Install Python dependencies
pip install langgraph langchain-ollama

# Build A-PXM compiler (from repo root)
apxm compiler build
```

### Run A-PXM Version

```bash
cd papers/CF26/benchmarks/workloads/7_reflection

# Compile and run
apxm compiler run workflow.ais -O1
```

### Run LangGraph Comparison

```bash
cd papers/CF26/benchmarks/workloads/7_reflection
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 7_reflection

# With JSON output
apxm workloads run 7_reflection --json
```

---

## Results

*To be filled after benchmark execution*

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Reflection support | Native REFL op | Custom prompting | |
| Structured output | Built-in format | Manual parsing | |
| Iteration control | Compiler-managed | Runtime loops | |
| Cost optimization | Fusible prompts | Separate calls | |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Native operation**: REFL is a first-class AIS operation with defined semantics.

2. **Structured output**: Reflection produces a consistent format that subsequent operations can consume.

3. **Optimization potential**: The compiler can fuse REFL with subsequent RSN operations.

### Key Insight

This workload demonstrates that A-PXM elevates common agent patterns (like reflection) to first-class operations. This enables compiler reasoning and optimization that's impossible when patterns are hidden in prompts.
