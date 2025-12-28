# A-PXM Benchmark Suite

## Quick Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BENCHMARK OVERVIEW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  COMPARISON MODEL: AIS DSL vs LangGraph DSL (NOT Rust vs Python)            │
│                                                                              │
│  VERIFIED OPTIMIZATIONS (safe to benchmark):                                │
│    ✓ FuseReasoning      - Batches RSN chains (KILLER DEMO: Nx speedup)      │
│    ✓ CapabilityScheduling - Adds static annotations to IR                   │
│    ✓ Type Verification  - 50+ compile-time error checks                     │
│    ✓ Runtime Parallelism - ~8μs/op overhead, measured                       │
│                                                                              │
│  OUTPUT FORMAT: JSON (machine-readable, reproducible)                       │
│                                                                              │
│  STRUCTURE:                                                                  │
│    benchmarks/workloads/  - Test workloads in both DSLs                     │
│    benchmarks/results/    - JSON output files                               │
│    benchmarks/analysis/   - Comparison scripts                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Fair Comparison Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FAIR COMPARISON MODEL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  A-PXM (Custom DSL)                    LangGraph (Python Embedded DSL)       │
│  ─────────────────                     ────────────────────────────────      │
│                                                                              │
│  ┌──────────────────┐                  ┌──────────────────────────────┐      │
│  │  workflow.ais    │                  │  workflow.py                 │      │
│  │                  │                  │                              │      │
│  │  agent Research {│                  │  graph = StateGraph(State)   │      │
│  │    flow main {   │                  │  graph.add_node("a", fn)     │      │
│  │      rsn "..." →r│                  │  graph.add_node("b", fn)     │      │
│  │      rsn "..." →s│                  │  graph.add_edge(START, "a")  │      │
│  │      merge → out │                  │  ...                         │      │
│  │    }             │                  │  app = graph.compile()       │      │
│  │  }               │                  │  app.invoke(state)           │      │
│  └────────┬─────────┘                  └──────────────┬───────────────┘      │
│           │                                           │                      │
│           ▼                                           ▼                      │
│  ┌──────────────────┐                  ┌──────────────────────────────┐      │
│  │  apxm compile    │                  │  Python interpreter          │      │
│  │  (MLIR → Binary) │                  │  (runtime graph build)       │      │
│  └────────┬─────────┘                  └──────────────┬───────────────┘      │
│           │                                           │                      │
│           ▼                                           ▼                      │
│  ┌──────────────────┐                  ┌──────────────────────────────┐      │
│  │  Execute Binary  │                  │  Execute Graph               │      │
│  │  (Rust runtime)  │                  │  (Python runtime)            │      │
│  └──────────────────┘                  └──────────────────────────────┘      │
│                                                                              │
│  MEASURE: Total time from source to result                                   │
│  COMPARE: Developer experience, lines of code, error handling                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this comparison is fair:**
- Both are DSLs for defining agent workflows
- Measures what developers actually experience
- Compares equivalent workflows, not artificial micro-benchmarks

---

## Testing Methodology

### Philosophy

1. **Measure what developers experience**: End-to-end from source code to result
2. **Fair comparison**: Equivalent workflows, not artificial micro-benchmarks
3. **Reproducible**: JSON output, versioned configs, documented environment
4. **General-purpose**: Benchmarks useful beyond any single paper

### What We Measure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEASUREMENT DIMENSIONS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. COMPILATION/SETUP TIME                                                   │
│     ├── A-PXM: Time to compile .ais → binary                                │
│     └── LangGraph: Time to build StateGraph and compile()                   │
│                                                                              │
│  2. EXECUTION TIME                                                           │
│     ├── Cold start: First execution (includes JIT, cache misses)            │
│     ├── Warm: Subsequent executions (steady state)                          │
│     └── Per-operation overhead: (total - LLM time) / num_operations         │
│                                                                              │
│  3. PARALLELISM EFFICIENCY                                                   │
│     ├── Theoretical speedup: N parallel ops → Nx faster                     │
│     ├── Actual speedup: Measured wall time                                  │
│     └── Efficiency: actual / theoretical × 100%                             │
│                                                                              │
│  4. DEVELOPER EXPERIENCE                                                     │
│     ├── Lines of code: For equivalent workflow                              │
│     ├── Error handling: Compile-time vs runtime errors                      │
│     └── Debugging: State inspection latency                                 │
│                                                                              │
│  5. RESOURCE USAGE                                                           │
│     ├── Memory: Peak RSS during execution                                   │
│     ├── CPU: Utilization during parallel ops                                │
│     └── Tokens: LLM API consumption                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Statistical Rigor

Following "AI Agents That Matter" (arXiv:2407.01502):

- **Iterations**: 100 runs minimum for synthetic, 10+ for LLM workloads
- **Warmup**: 3-5 iterations discarded
- **Metrics**: Mean, std dev, 95% CI, P50, P99
- **Variance**: Report coefficient of variation (CV = std/mean)

---

## Benchmark Workloads

### Workload 1: Parallel Research (Primary)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PARALLEL RESEARCH WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        ┌─────────────────────┐                              │
│                        │  INPUT: topic       │                              │
│                        └──────────┬──────────┘                              │
│                                   │                                          │
│                    ┌──────────────┼──────────────┐                          │
│                    │              │              │                          │
│                    ▼              ▼              ▼                          │
│          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                    │
│          │ RSN: Domain │ │ RSN: Recent │ │ RSN: Impact │  ← PARALLEL        │
│          │ background  │ │ advances    │ │ assessment  │                    │
│          └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                    │
│                 │               │               │                            │
│                 └───────────────┼───────────────┘                            │
│                                 │                                            │
│                                 ▼                                            │
│                        ┌─────────────────┐                                  │
│                        │ MERGE: Combine  │                                  │
│                        └────────┬────────┘                                  │
│                                 │                                            │
│                                 ▼                                            │
│                        ┌─────────────────┐                                  │
│                        │ RSN: Synthesize │  ← SEQUENTIAL                    │
│                        └────────┬────────┘                                  │
│                                 │                                            │
│                                 ▼                                            │
│                        ┌─────────────────┐                                  │
│                        │  OUTPUT: report │                                  │
│                        └─────────────────┘                                  │
│                                                                              │
│  Measures: Parallelism efficiency, total latency, per-op overhead           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Files:**
- `workloads/parallel_research.ais` - A-PXM version (~10 lines)
- `workloads/parallel_research.py` - LangGraph version (~40 lines)

### Workload 2: Scalability Curve

Test parallelism at N = 1, 2, 4, 8 concurrent operations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXPECTED RESULTS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Speedup                                                                     │
│     ▲                                                                        │
│   8 ┤                                    ╱ Theoretical (y=x)                 │
│     │                                  ╱                                     │
│   6 ┤                               ╱                                        │
│     │                            ╱   ●                                       │
│   4 ┤                         ╱    ●   Actual A-PXM                          │
│     │                      ╱     ●                                           │
│   2 ┤                   ╱      ●                                             │
│     │                ╱       ●                                               │
│   1 ┼──────────────●─────────────────────────────────────▶ N                 │
│     1        2        4        8       16                                    │
│                                                                              │
│  Efficiency (actual/theoretical):                                            │
│  N=1: 100%  N=2: ~85%  N=4: ~75%  N=8: ~60%                                 │
│                                                                              │
│  Bottleneck: LLM API rate limiting, not scheduler                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Workload 3: Reasoning Chain Fusion (COMPILER POWER)

**This is the killer compiler optimization - Nx latency savings!**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REASONING CHAIN FUSION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SOURCE CODE (unfused):                                                      │
│  ───────────────────────                                                     │
│  rsn "What is quantum computing?" -> definition                              │
│  rsn "Based on: " + definition + ", explain qubits" -> qubits               │
│  rsn "Summarize: " + qubits -> summary                                       │
│                                                                              │
│  WITHOUT FUSION (LangGraph behavior):                                        │
│  ────────────────────────────────────                                        │
│  ┌─────────┐   2s    ┌─────────┐   2s    ┌─────────┐   2s    ┌─────────┐    │
│  │ RSN #1  │ ──────▶ │ RSN #2  │ ──────▶ │ RSN #3  │ ──────▶ │ RESULT  │    │
│  └─────────┘         └─────────┘         └─────────┘         └─────────┘    │
│                                                                              │
│  Total: 3 × 2s = 6 seconds                                                  │
│  LLM API calls: 3                                                           │
│                                                                              │
│  WITH FUSION (A-PXM FuseReasoning pass):                                    │
│  ─────────────────────────────────────────                                   │
│  ┌──────────────────────────────────────┐   2s    ┌─────────┐               │
│  │ FUSED RSN (batched template)         │ ──────▶ │ RESULT  │               │
│  │ "What is quantum computing?          │         └─────────┘               │
│  │ ---                                  │                                   │
│  │ Based on the above, explain qubits   │                                   │
│  │ ---                                  │                                   │
│  │ Summarize the above"                 │                                   │
│  └──────────────────────────────────────┘                                   │
│                                                                              │
│  Total: 1 × 2s = 2 seconds                                                  │
│  LLM API calls: 1                                                           │
│  SPEEDUP: 3x (more chains = more savings)                                   │
│                                                                              │
│  5-chain example: 5 × 2s = 10s → 1 × 2s = 2s = 5x SPEEDUP                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Files:**
- `workloads/chain_fusion.ais` - A-PXM version (compiler fuses automatically)
- `workloads/chain_fusion.py` - LangGraph version (cannot fuse)

### Workload 4: Compile-Time Type Verification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ERROR DETECTION COMPARISON                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  A-PXM (Compile-Time)                    LangGraph (Runtime)                 │
│  ────────────────────                    ──────────────────                  │
│                                                                              │
│  $ apxm compile broken.ais               $ python broken.py                  │
│                                                                              │
│  error[E0308]: undefined variable        Traceback (most recent call last):  │
│    --> broken.ais:5:12                     File "broken.py", line 47         │
│    │                                       in invoke                         │
│  5 │     rsn undefined_var -> result       KeyError: 'missing_key'           │
│    │         ^^^^^^^^^^^^                                                    │
│    │         not defined                 # After LLM call already made!      │
│                                          # Tokens wasted, cost incurred      │
│  COST: $0.00 (caught before LLM)         COST: $0.15 (failed after LLM)     │
│  TIME: 50ms (compile only)               TIME: 3.2s (LLM + failure)         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Files:**
- `workloads/type_error.ais` - A-PXM version (compile-time error)
- `workloads/type_error.py` - LangGraph version (runtime error)

### Workload 5: Static Parallelism Annotation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STATIC PARALLELISM DETECTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  A-PXM COMPILER ANALYSIS (CapabilityScheduling pass):                       │
│  ─────────────────────────────────────────────────────                       │
│  $ apxm compile --emit=ir parallel.ais                                      │
│                                                                              │
│  %a = ais.rsn "Research A" {ais.parallel_safe = true,                       │
│                              ais.estimated_cost = 5,                         │
│                              ais.tier = "reasoning"}                         │
│  %b = ais.rsn "Research B" {ais.parallel_safe = true, ...}                  │
│  %c = ais.rsn "Research C" {ais.parallel_safe = true, ...}                  │
│                                                                              │
│  BENEFITS:                                                                   │
│  - Scheduler KNOWS operations are parallel-safe at load time                │
│  - No runtime dependency analysis needed                                     │
│  - Cost estimates enable intelligent scheduling                              │
│                                                                              │
│  LANGGRAPH:                                                                  │
│  ──────────                                                                  │
│  # Must explicitly use Send API                                             │
│  # No cost model                                                            │
│  # Runtime graph analysis on every invocation                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Verified Compiler Optimizations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              VERIFIED COMPILER OPTIMIZATIONS (Dec 2025 Audit)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STATUS KEY:  ✓ VERIFIED  ⚠ PARTIAL  ✗ MISSING                              │
│                                                                              │
│  OPTIMIZATION           │ STATUS   │ DETAILS                           │    │
│  ───────────────────────┼──────────┼───────────────────────────────────│    │
│  Reasoning Chain Fusion │ ✓ WORKS  │ FuseReasoning.cpp (160 lines)     │    │
│    → N RSN calls → 1    │          │ Batches dependent RSN chains      │    │
│                         │          │                                   │    │
│  Static Annotations     │ ✓ WORKS  │ CapabilityScheduling.cpp          │    │
│    → parallel_safe      │          │ Adds: ais.parallel_safe,          │    │
│    → estimated_cost     │          │       ais.estimated_cost,         │    │
│    → tier, intent       │          │       ais.tier, ais.intent        │    │
│                         │          │                                   │    │
│  Type Verification      │ ✓ WORKS  │ AISOps.cpp (50+ error checks)     │    │
│    → Compile-time errs  │          │ Validates tokens, types, names    │    │
│                         │          │                                   │    │
│  Symbol DCE             │ ⚠ PARTIAL│ MLIR built-in, only removes       │    │
│    → Unused symbols     │          │ unused SYMBOLS (funcs/globals)    │    │
│    → NOT operation DCE  │          │ Does NOT remove unused RSN ops    │    │
│                         │          │                                   │    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Format: JSON

All benchmarks output JSON for analysis:

```json
{
  "meta": {
    "timestamp": "2025-12-26T10:30:00Z",
    "machine": {
      "os": "Darwin 25.1.0",
      "cpu": "Apple M2 Pro",
      "memory_gb": 32
    },
    "versions": {
      "apxm": "0.1.0",
      "langgraph": "0.2.53",
      "python": "3.12.0",
      "ollama_model": "phi3:mini"
    }
  },
  "workload": "parallel_research",
  "config": {
    "parallel_ops": 3,
    "iterations": 100,
    "warmup": 5
  },
  "results": {
    "apxm": {
      "compile_time_ms": 42.5,
      "execution": {
        "mean_ms": 3250.2,
        "std_ms": 145.3,
        "ci_95_ms": 28.4,
        "p50_ms": 3180.0,
        "p99_ms": 3520.1,
        "samples": [3150, 3200, "..."]
      },
      "per_op_overhead_us": 8.4,
      "parallelism": {
        "theoretical": 3.0,
        "actual": 2.55,
        "efficiency_pct": 85.0
      }
    },
    "langgraph": {
      "graph_build_time_ms": 12.3,
      "execution": {
        "mean_ms": 4850.7,
        "std_ms": 203.1,
        "ci_95_ms": 39.8,
        "p50_ms": 4720.0,
        "p99_ms": 5280.3,
        "samples": [4650, 4780, "..."]
      },
      "per_op_overhead_us": 14200.0,
      "parallelism": {
        "theoretical": 3.0,
        "actual": 1.92,
        "efficiency_pct": 64.0
      }
    }
  },
  "comparison": {
    "overhead_ratio": 1690.5,
    "parallelism_efficiency_delta_pct": 21.0,
    "total_time_speedup": 1.49
  }
}
```

---

## Running Benchmarks

```bash
# Install dependencies
pip install -r requirements.txt

# Run all benchmarks
python runner.py

# Run specific workload
python runner.py --workload parallel_research

# Run with custom iterations
python runner.py --iterations 50 --warmup 3

# Output to specific file
python runner.py --output results/my_run.json
```

---

## Directory Structure

```
benchmarks/
├── README.md              # This file
├── runner.py              # Unified benchmark runner
├── config.json            # Benchmark configuration
├── requirements.txt       # Python dependencies
├── workloads/             # Test workloads in both DSLs
│   ├── parallel_research.ais
│   ├── parallel_research.py
│   ├── chain_fusion.ais
│   ├── chain_fusion.py
│   ├── type_error.ais
│   └── type_error.py
├── results/               # JSON output files
│   └── .gitkeep
└── analysis/              # Analysis scripts
    └── compare.py         # Generate comparison tables
```
