# Workload 2: Chain Fusion

## Purpose

Demonstrate that A-PXM's typed IR enables compiler optimizations impossible in opaque orchestration frameworks. The FuseAskOps pass batches dependent ask chains into a single LLM call.

## What We're Demonstrating

**A-PXM Property**: Typed IR enables FuseAskOps optimization

Because A-PXM represents operations in a typed intermediate representation, the compiler can analyze dependency chains and transform them. FuseAskOps identifies sequential ask operations where each depends on the previous output, then batches them into a single prompt.

```
WITHOUT FUSION (5 API calls):
[ASK 1] --2s--> [ASK 2] --2s--> [ASK 3] --2s--> [ASK 4] --2s--> [ASK 5]
Total: ~10s, 5 LLM calls

WITH FUSION (1 API call):
+--------------------------------------------------+
| FUSED ASK (single batched prompt)                |
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
        // These 5 ask ops form a dependency chain
        // FuseAskOps pass will batch them into a single prompt
        ask("Define quantum computing") -> step1
        ask("Using: " + step1 + ", explain qubits") -> step2
        ask("Using: " + step2 + ", explain superposition") -> step3
        ask("Using: " + step3 + ", explain entanglement") -> step4
        ask("Summarize all concepts above: " + step4) -> summary
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

### Quick Run (Compile + Execute)

```bash
cd papers/cf26/benchmarks/workloads/2_chain_fusion

# Execute without fusion (O0)
apxm execute -O0 workflow.ais

# Execute with fusion (O1) - should be faster
apxm execute -O1 workflow.ais
```

### Compile Only

```bash
# Compile without fusion
apxm compile workflow.ais -o workflow_O0.apxmobj -O0 --emit-diagnostics diagnostics_O0.json

# Compile with fusion
apxm compile workflow.ais -o workflow_O1.apxmobj -O1 --emit-diagnostics diagnostics_O1.json
```

### Run Pre-compiled Artifact

```bash
# Run with metrics export
apxm run --emit-metrics metrics.json workflow_O1.apxmobj
```

### Run LangGraph Comparison

```bash
python workflow.py
```

### Run Full Benchmark

```bash
# From repo root
apxm workloads run 2_chain_fusion

# With JSON output
apxm workloads run 2_chain_fusion --json
```

---

## Results

### Measured Values

| Metric | O0 (No Fusion) | O1 (With Fusion) | Improvement |
|--------|----------------|------------------|-------------|
| Duration (ms) | 11276 | 5062 | **2.2x faster** |
| LLM Calls | 5 | 1 | **5x reduction** |
| DAG Nodes | 20 | 2 | 10x simpler graph |
| DAG Edges | 19 | 1 | Minimal dependencies |
| Input Tokens | 743 | 141 | 5x reduction |
| Output Tokens | 1075 | 589 | Fewer total tokens |

### Compiler Passes Applied

| O0 | O1 |
|----|-----|
| lower-to-async | normalize, scheduling, **fuse-ask-ops**, canonicalizer, cse, symbol-dce, lower-to-async |

## Analysis

### Observations

1. **Significant latency reduction**: 5 sequential LLM calls at ~2s each = ~11s unfused. Fusion reduces to 1 call at ~5s = **2.2x speedup**.

2. **Cost savings**: Fewer API calls + fewer total tokens (743 → 141 input) = significant cost reduction.

3. **Framework overhead negligible**: Scheduling overhead (~2µs/op) is 6 orders of magnitude below LLM latency.

### Key Insight

This workload demonstrates that A-PXM's **typed IR enables compiler transformations impossible in opaque frameworks**. LangGraph/CrewAI cannot see inside Python functions to optimize prompt chains. A-PXM's IR exposes the structure needed for automatic optimization.

---

## Future Work: Fusion Heuristics

The current FuseAskOps pass eagerly fuses all valid ask chains. Production deployment requires heuristics to identify cases where fusion may harm output quality. The pass infrastructure is defined in `Passes.td` with the following options:

### Configurable Options (Defined, Not Yet Implemented)

| Option | Default | Description |
|--------|---------|-------------|
| `max-fusion-depth` | 5 | Maximum operations to fuse (0 = unlimited) |
| `max-template-tokens` | 2000 | Maximum estimated tokens in fused template |
| `fusion-mode` | "auto" | Strategy: `eager`, `conservative`, `auto` |

### Planned Heuristics

1. **Chain Length Limits**: Long chains (>5 ops) may overwhelm the model's ability to produce coherent multi-part responses. Track fusion depth during chain tracing.

2. **Token Budget**: Estimate combined template token count. Very long prompts degrade response quality and may hit context limits.

3. **Fusion Modes**:
   - `eager`: Always fuse (current behavior, good for latency-critical workloads)
   - `conservative`: Only fuse short chains (2-3 ops)
   - `auto`: Apply quality-preservation heuristics based on task type

4. **Per-Operation Opt-Out**: Support `@[no_fuse]` attribute on individual operations where developers know fusion would harm quality (e.g., complex reasoning steps).

5. **Task-Type Awareness**: Different tasks tolerate fusion differently:
   - **Classification/Extraction**: Highly fusible (structured output)
   - **Creative/Reasoning**: Less fusible (model needs separation)

### Research Questions

- What is the quality degradation curve as fusion depth increases?
- Can we detect task type from template content to auto-select fusion strategy?
- Should we fuse only within "homogeneous" chains (same task type)?
