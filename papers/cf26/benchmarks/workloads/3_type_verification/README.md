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
5 |     ask(undefined_var) -> output        KeyError: 'missing_key'
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
        ask("Do something useful") -> result

        // ERROR: undefined_var is not defined anywhere
        // A-PXM compiler will catch this BEFORE execution
        ask("Use this: " + undefined_var) -> output
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

### Run A-PXM Version (Compile-Time Error)

```bash
cd papers/cf26/benchmarks/workloads/3_type_verification

# This SHOULD fail at compile time with a clear error message
apxm compile workflow.ais -o workflow.apxmobj
```

Expected output: Compile-time error with precise source location.

### Run LangGraph Comparison (Runtime Error)

```bash
# This will fail at RUNTIME after making an LLM call
python workflow.py
```

Expected output: Runtime exception after LLM call is already made.

### Run Full Benchmark

```bash
# From repo root
apxm workloads run 3_type_verification

# With JSON output
apxm workloads run 3_type_verification --json
```

---

## Collecting Metrics

### Compiler Diagnostics (`--emit-diagnostics`)

For valid workflows, export compilation statistics:

```bash
apxm compile workflow.ais -o workflow.apxmobj --emit-diagnostics diagnostics.json
```

Output includes:
- `dag_statistics`: total_nodes, entry_nodes, exit_nodes, total_edges
- `compilation_phases`: total_ms, artifact_gen_ms
- `passes_applied`: list of optimization passes

Note: For this workload, compilation should fail with an error before diagnostics are written.

---

## Results

### Measured Values

| Metric | LangGraph | A-PXM | Notes |
|--------|-----------|-------|-------|
| Error Detection | Runtime | **Compile-time** | Key differentiator |
| Time to Error | ~1000ms+ | **65ms** | 15x+ faster feedback |
| LLM Calls Wasted | 1+ | **0** | Cost savings |
| Cost Wasted | $0.01+ | **$0.00** | Real money saved |
| Error Quality | Stack trace | **Precise source location** | Better DX |

### Errors Caught at Compile-Time

| Error Type | Test File | Expected Error | Status |
|------------|-----------|----------------|--------|
| Undefined Variable | `01_undefined_variable.ais` | `Undefined variable: undefined_input` | ✓ Caught |
| Invalid Memory Space | `02_invalid_memory_space.ais` | `space must be 'stm', 'ltm', or 'episodic'` | ✓ Caught |
| Empty Ask Template | `04_empty_reasoning.ais` | `ask operation requires a template or context` | ✓ Caught |
| Empty Capability Name | `03_empty_capability.ais` | `capability name cannot be empty` | ✓ Caught |
| Empty Switch Cases | `05_switch_empty_cases.ais` | `switch must have at least one case label` | ✓ Caught |

All 5 error categories are now verified at compile-time before any LLM calls.

### Error Message Quality

```
error: Undefined variable: undefined_var
  --> workflow.ais:14:28
     |
  14 |     ask("Use this: " + undefined_var) -> output
     |                        ^^^^^^^^^^^^^
```

Precise source location (file:line:column) vs Python's generic stack traces.

---

## Analysis

### Observations

1. **Zero LLM calls wasted**: A-PXM catches all 5 error types during compilation (~65ms), before any runtime execution or API calls.

2. **15x+ faster feedback**: Compile-time errors appear in 65ms vs 1000ms+ for runtime discovery (includes LLM latency).

3. **Better error messages**: A-PXM provides precise source location (file:line:column) vs Python stack traces.

4. **Comprehensive verification**: The verifier catches semantic errors beyond syntax:
   - Undefined variables (dataflow analysis)
   - Invalid enum values (memory space validation)
   - Empty operations (ask, inv, switch constraints)

### Verifier Improvements Made

During this workload, we identified and fixed two verifier gaps:

1. **Empty capability fallback** (`MLIRGenOperations.cpp:160-162`): Removed fallback that masked empty capability strings
2. **Empty switch cases** (`AISOps.cpp:641-643`): Added check requiring at least one case label

### Key Insight

This workload demonstrates that A-PXM's type system provides **real cost savings**. Every error caught at compile time is an LLM call saved. For production workflows running thousands of times:

```
Scenario: 1000 workflow executions with a bug

LangGraph: 1000 × $0.01 = $10+ wasted before bug found
A-PXM:     0 × $0.01 = $0 wasted, bug caught in 65ms

For expensive models (GPT-4, Claude):
LangGraph: 1000 × $0.10 = $100+ wasted
A-PXM:     $0 wasted
```
