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
|  | RSN: Classify input     |  <- LLM determines category       |
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
|  |TECH | |CREA | |FACT | | DEFAULT |                          |
|  +--+--+ +--+--+ +--+--+ +----+----+                          |
|     |       |       |         |                                |
|     +-------+-------+---------+                                |
|                 |                                              |
|                 v                                              |
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
        rsn("Classify this input into: technical, creative, or factual. Input: " + input) -> category

        // 2. Route based on classification (only selected branch runs)
        switch category {
            case "technical" => rsn("Provide a detailed technical explanation for: " + input)
            case "creative" => rsn("Provide a creative, imaginative response for: " + input)
            case "factual" => rsn("Provide accurate factual information for: " + input)
            default => rsn("Provide a helpful general response for: " + input)
        }

        // 3. Emit a final response that reflects the chosen category
        rsn("Provide a response for: " + input + " (category: " + category + ")") -> output
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
cd papers/CF26/benchmarks/workloads/9_conditional_routing

# Compile and run
apxm compiler run workflow.ais -O1
```

### Run LangGraph Comparison

```bash
cd papers/CF26/benchmarks/workloads/9_conditional_routing
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 9_conditional_routing

# With JSON output
apxm workloads run 9_conditional_routing --json
```

---

## Results

*To be filled after benchmark execution*

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Route definition | Compile-time switch | Runtime conditional | |
| Route validation | Static checking | Runtime errors | |
| Branch handling | Native control flow | Conditional edges | |
| Default case | Compiler-enforced | Optional | |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Single branch execution**: Only the matched branch executes, no wasted LLM calls.

2. **Compile-time validation**: All branches are type-checked before execution.

3. **Default case required**: Compiler enforces exhaustive pattern matching.

### Key Insight

This workload demonstrates that A-PXM's control flow is part of the formal specification, not ad-hoc Python functions. The switch/case construct integrates with dataflow semantics, enabling compiler analysis of all possible execution paths.
