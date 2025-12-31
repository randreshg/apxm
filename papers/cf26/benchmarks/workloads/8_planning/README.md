# Workload 8: Planning

## Purpose

Demonstrate A-PXM's native PLAN operation for task decomposition. Planning is a first-class operation that integrates with automatic parallelism for step execution.

## What We're Demonstrating

**A-PXM Property**: AIS PLAN operation for native planning

A-PXM specifies an explicit `PLAN` operation that decomposes goals into executable steps. The compiler can analyze step dependencies and parallelize independent steps automatically.

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
|  |       into steps        |                                   |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | RSN: Execute step 1     |                                   |
|  | RSN: Execute step 2     |  <- Parallel where possible       |
|  | RSN: Execute step 3     |                                   |
|  +--------+----------------+                                   |
|           |                                                    |
|           v                                                    |
|  +-------------------------+                                   |
|  | RSN: Synthesize results |  <- Combine step outputs          |
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
    @entry flow main(goal: str) -> str {
        // 1. Decompose goal into executable steps
        plan(goal) -> steps

        // 2. Execute each step (compiler detects parallelism opportunities)
        rsn("Execute step 1: " + steps) -> step1_result
        rsn("Execute step 2: " + steps) -> step2_result
        rsn("Execute step 3: " + steps) -> step3_result

        // 3. Merge step results
        merge(step1_result, step2_result, step3_result) -> combined

        // 4. Synthesize final result
        rsn("Synthesize all step results: " + combined) -> final_result
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
cd papers/CF26/benchmarks/workloads/8_planning

# Compile and run
apxm compiler run workflow.ais -O1
```

### Run LangGraph Comparison

```bash
cd papers/CF26/benchmarks/workloads/8_planning
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 8_planning

# With JSON output
apxm workloads run 8_planning --json
```

---

## Results

*To be filled after benchmark execution*

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Planning | Native PLAN op | Custom CoT prompting | |
| Step execution | Automatic parallelism | Manual orchestration | |
| Step format | Structured output | Text parsing required | |
| Replanning | Built-in support | Custom implementation | |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Integrated planning**: PLAN produces structured steps that feed directly into execution.

2. **Automatic parallelism**: Independent steps run concurrently without manual coordination.

3. **Replanning support**: The operation semantics support dynamic replanning when steps fail.

### Key Insight

This workload demonstrates that A-PXM integrates planning with execution. The PLAN operation produces steps, and dataflow semantics automatically parallelize independent step execution. This is the kind of integration impossible when planning is just a prompting pattern.
