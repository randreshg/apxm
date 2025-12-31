# Workload 2: Chain Fusion

## Purpose

Demonstrate that A-PXM's typed IR enables compiler optimizations impossible in opaque orchestration frameworks. The FuseReasoning pass batches dependent RSN chains into a single LLM call.

## What We're Demonstrating

**A-PXM Property**: Typed IR enables FuseReasoning optimization

Because A-PXM represents operations in a typed intermediate representation, the compiler can analyze dependency chains and transform them. FuseReasoning identifies sequential RSN operations where each depends on the previous output, then batches them into a single prompt.

```
WITHOUT FUSION (5 API calls):
[RSN 1] --2s--> [RSN 2] --2s--> [RSN 3] --2s--> [RSN 4] --2s--> [RSN 5]
Total: ~10s, 5 LLM calls

WITH FUSION (1 API call):
+--------------------------------------------------+
| FUSED RSN (single batched prompt)                |
| "Define quantum computing                        |
| ---                                              |
| Using the above, explain qubits                  |
| ---                                              |
| Using the above, explain superposition           |
| ---                                              |
| Using the above, explain entanglement            |
| ---                                              |
| Summarize all concepts above"                    |
+--------------------------------------------------+
Total: ~2s, 1 LLM call
SPEEDUP: 5x
```

### A-PXM Code (workflow.ais)

```
agent ChainFusion {
    @entry flow main() -> str {
        // These 5 RSN ops form a dependency chain
        // FuseReasoning pass will batch them into a single prompt
        rsn("Define quantum computing") -> step1
        rsn("Using: " + step1 + ", explain qubits") -> step2
        rsn("Using: " + step2 + ", explain superposition") -> step3
        rsn("Using: " + step3 + ", explain entanglement") -> step4
        rsn("Summarize all concepts above: " + step4) -> summary
        return summary
    }
}
```

### LangGraph Comparison

LangGraph has no visibility into prompt structure:
- Each node is an opaque Python function
- No way to analyze dependencies between prompts
- Each LLM call is a separate API request

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
cd papers/CF26/benchmarks/workloads/2_chain_fusion

# Compile without fusion (O0)
apxm compiler compile workflow.ais -O0

# Compile with fusion (O1)
apxm compiler compile workflow.ais -O1

# Compare both optimization levels
apxm compiler run workflow.ais -O0
apxm compiler run workflow.ais -O1
```

### Run LangGraph Comparison

```bash
cd papers/CF26/benchmarks/workloads/2_chain_fusion
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 2_chain_fusion

# With JSON output
apxm workloads run 2_chain_fusion --json
```

---

## Results

*To be filled after benchmark execution*

| Metric | LangGraph | A-PXM (O0) | A-PXM (O1) | Notes |
|--------|-----------|------------|------------|-------|
| LLM Calls | 5 | 5 | 1 | Fusion reduces calls |
| Mean latency (ms) | - | - | - | |
| API cost estimate | ~$0.05 | ~$0.05 | ~$0.01 | |
| Speedup | 1x | 1x | ~5x | |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Fusion mechanism works**: The compiler successfully identifies the dependency chain and batches operations.

2. **Latency reduction**: With 5 sequential calls at ~2s each, unfused takes ~10s. Fused should take ~2-3s.

3. **Cost savings**: Fewer API calls means lower cost, especially with per-call overhead.

### Research Insights from EVALUATION_DISCUSSION.md

The evaluation revealed that **naive fusion isn't always optimal**:
- FuseReasoning reduced LLM calls from 5 to 1
- But fused prompts can be harder for LLMs to process
- This demonstrates the IR enables the optimization, and opens research into cost-aware fusion heuristics

### Key Insight

This workload demonstrates that A-PXM's typed IR enables compiler transformations impossible in frameworks where operations are opaque functions. The IR exposes structure needed for optimization research.
