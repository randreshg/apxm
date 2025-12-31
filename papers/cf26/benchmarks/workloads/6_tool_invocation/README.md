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
|  | RSN: Decide which tool  |  <- LLM reasons about tool choice |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | INV: Execute tool       |  <- Invoke registered capability  |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | RSN: Synthesize answer  |  <- LLM reasons about results     |
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
        // 1. Invoke registered capability (search tool)
        // The runtime's capability registry validates this at compile time
        inv("search", "{}") -> search_results

        // 2. Reason about results to formulate answer
        rsn("Given search results: " + search_results + ", answer: " + query) -> answer

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
cd papers/CF26/benchmarks/workloads/6_tool_invocation

# Note: workflow.ais is currently disabled pending capability runtime
# Compile and run (when enabled)
apxm compiler run workflow.ais -O1
```

### Run LangGraph Comparison

```bash
cd papers/CF26/benchmarks/workloads/6_tool_invocation
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 6_tool_invocation

# With JSON output
apxm workloads run 6_tool_invocation --json
```

---

## Results

*To be filled after benchmark execution*

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Tool registration | Compile-time | Runtime | Key differentiator |
| Validation | Static type checking | Runtime errors | |
| Invocation overhead | Microseconds | Milliseconds | |
| Tool discovery | Capability registry | Dynamic lookup | |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Compile-time validation**: Invalid tool names or signatures are caught before execution.

2. **Typed tool signatures**: Tool inputs and outputs have explicit types in the capability registry.

3. **Low invocation overhead**: Direct capability dispatch vs runtime reflection.

### Key Insight

This workload demonstrates that A-PXM treats tool invocation as a first-class operation with typed semantics. The capability registry is part of the AAM specification, enabling formal reasoning about agent capabilities.
