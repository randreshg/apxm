# Workload 14: Token Estimation

## Purpose

Demonstrate that A-PXM's typed IR enables static token cost estimation. This allows predicting fusion effectiveness before execution.

## What We're Demonstrating

**A-PXM Property**: Typed IR enables cost estimation

Because A-PXM represents prompts in a typed IR, the compiler can estimate token counts before execution. This enables:
- Predicting fusion cost savings
- Deciding when fusion is beneficial
- Budget estimation for workflows

```
Token Estimation:
+----------------+------------------+------------------+
| Workflow       | O0 Tokens        | O1 Tokens        |
+----------------+------------------+------------------+
| Simple Chain   | 3 x system_prompt| 1 x system_prompt|
|                | + 3 x user       | + 1 x user       |
+----------------+------------------+------------------+
| Sequential     | 5 x system_prompt| 1 x system_prompt|
| Reasoning      | + 5 x user       | + 1 x user       |
+----------------+------------------+------------------+
| Parallel       | 3 x system_prompt| 1 x system_prompt|
| Research       | + 3 x user       | + 1 x user       |
+----------------+------------------+------------------+

System prompt amortization:
- Each LLM call includes a system prompt (~100-500 tokens)
- Fusion reduces calls, amortizing system prompt cost
```

### Test Workflows

**simple_chain.ais**: 3-step reasoning with explicit dependencies
```
agent SimpleChain {
    @entry flow main() -> str {
        rsn("Analyze this problem: How can we improve software testing?") -> step1
        rsn("Based on this analysis: {{step1}}. Generate 3 potential solutions.") -> step2
        rsn("Given these solutions: {{step2}}. Evaluate and select the best one.") -> step3
        return step3
    }
}
```

**sequential_reasoning.ais**: 5-step deep reasoning chain

**parallel_research.ais**: 3 parallel research queries + merge

---

## How to Run

### Prerequisites

```bash
# Build A-PXM compiler (from repo root)
apxm compiler build
```

### Run Token Estimation Benchmark

```bash
cd papers/CF26/benchmarks/workloads/14_token_estimation

# Run estimation
python run.py

# With JSON output
python run.py --json
```

### Run via CLI

```bash
# From repo root
apxm workloads run 14_token_estimation --json
```

---

## Results

*To be filled after benchmark execution*

| Workflow | O0 Calls | O1 Calls | Est. Token Savings | API Savings |
|----------|----------|----------|-------------------|-------------|
| Simple Chain | 3 | 1 | - | ~66% |
| Sequential | 5 | 1 | - | ~80% |
| Parallel | 3 | 1 | - | ~66% |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **System prompt amortization**: Fusing N calls into 1 saves (N-1) system prompt copies.

2. **Predictable savings**: Token reduction can be estimated statically from the IR.

3. **Cost model foundation**: These estimates enable automated fusion decisions.

### Key Insight

This workload demonstrates that A-PXM's typed IR enables cost analysis before execution. Unlike opaque frameworks where prompts are runtime strings, A-PXM can reason about token costs statically. This is the foundation for cost-aware optimization passes.
