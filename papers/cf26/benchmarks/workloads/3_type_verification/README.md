# Workload 3: Type Verification

## Purpose

Demonstrate that A-PXM's typed operations catch errors at compile time, before any LLM calls are made. This prevents wasted compute and cost from runtime failures.

## What We're Demonstrating

**A-PXM Property**: Typed operations enable compile-time verification

A-PXM's verifier analyzes the dataflow graph and catches undefined variables, type mismatches, and invalid operations before execution. In dynamic frameworks, these errors only surface at runtime after LLM calls have already been made.

```
A-PXM (Compile-Time):                    LangGraph (Runtime):
----------------------                   ---------------------

$ apxm compile workflow.ais              $ python workflow.py

error[E0425]: undefined variable         Traceback (most recent call last):
 --> workflow.ais:5:20                     File "workflow.py", line 47
  |                                        in invoke
5 |     rsn(undefined_var) -> output        KeyError: 'missing_key'
  |         ^^^^^^^^^^^^
  |         not defined                  # After LLM call already made!
                                         # Tokens wasted, cost incurred
COST: $0.00 (caught before LLM)
TIME: ~80ms (compile only)               COST: $0.01+ (failed after LLM)
                                         TIME: ~1000ms (LLM + failure)
```

### A-PXM Code (workflow.ais) - INTENTIONALLY BROKEN

```
agent BrokenAgent {
    flow main {
        // First operation succeeds
        rsn("Do something useful") -> result

        // ERROR: undefined_var is not defined anywhere
        // A-PXM compiler will catch this BEFORE execution
        rsn("Use this: " + undefined_var) -> output
    }
}
```

### Why This Matters at Scale

```
Scenario: 1000 workflow executions with a bug

LangGraph: 1000 x $0.01 = $10+ wasted before bug found
A-PXM:     0 x $0.01 = $0 wasted, bug caught immediately

For expensive models (GPT-4, Claude):
LangGraph: 1000 x $0.10 = $100+ wasted
A-PXM:     $0 wasted
```

---

## How to Run

### Prerequisites

```bash
# Install Python dependencies
pip install langgraph langchain-ollama

# Build A-PXM compiler (from repo root)
apxm compiler build
```

### Run A-PXM Version (Compile-Time Error)

```bash
cd papers/CF26/benchmarks/workloads/3_type_verification

# This SHOULD fail at compile time with a clear error message
apxm compiler compile workflow.ais
```

### Run LangGraph Comparison (Runtime Error)

```bash
cd papers/CF26/benchmarks/workloads/3_type_verification

# This will fail at RUNTIME after making an LLM call
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 3_type_verification

# With JSON output
apxm workloads run 3_type_verification --json
```

---

## Results

*To be filled after benchmark execution*

| Metric | LangGraph | A-PXM | Notes |
|--------|-----------|-------|-------|
| Error Detection | Runtime | Compile-time | Key differentiator |
| Time to Error | ~1000ms | ~80ms | 12x faster |
| LLM Calls Wasted | 1 | 0 | Cost savings |
| Cost Wasted | $0.01+ | $0.00 | Real money saved |
| Error Quality | Stack trace | Precise location | Better DX |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Zero LLM calls wasted**: A-PXM catches the undefined variable during compilation, before any runtime execution.

2. **Faster feedback**: Compile-time errors appear in ~80ms vs ~1000ms for runtime discovery.

3. **Better error messages**: A-PXM provides precise source location vs Python stack traces.

### Key Insight

This workload demonstrates that A-PXM's type system provides real cost savings. Every error caught at compile time is an LLM call saved. For production workflows running thousands of times, this adds up to significant cost and time savings.
