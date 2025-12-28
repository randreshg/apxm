# A-PXM Paper Testing Plan

## Research-Informed Evaluation Methodology

Based on analysis of how other PXMs and agent frameworks were evaluated:

### Established Patterns from PXM Literature

1. **Codelet Model Evaluation** (EXADAPT 2014)
   - 3-tier hierarchy: microbenchmarks → synthetic kernels → real workloads
   - Overhead characterization through targeted micro-benchmarks
   - Speedup = S/Tp, Efficiency = Speedup/p
   - DGEMM: 1.40x avg speedup; Graph 500: 1.15-2.38x speedup

2. **MLIR Evaluation Philosophy** (CGO 2021)
   - Multi-dimensional: performance + adoption + developer experience
   - "Optimizations should pay for themselves"
   - Code reduction metrics matter for DSLs

3. **LLMCompiler Evaluation** (ICML 2024)
   - 3.7x latency speedup, 6.7x cost savings, 9% accuracy improvements
   - Quality preservation is critical for fusion claims
   - Speedup must not sacrifice correctness

4. **Multi-Agent Framework Metrics**
   - Node F1, Structural Similarity Index, Graph Edit Distance
   - Trajectory-level evaluation, not just final accuracy
   - Scaling laws with explicit R² reporting

---

## Paper Claims to Validate

| #  | Claim                                                           | Source     | Benchmark(s)                        |
|----|-----------------------------------------------------------------|------------|-------------------------------------|
| C1 | AAM provides formal agent state model (B,G,C) + 3-tier memory   | Theory     | 5_memory_augmented                  |
| C2 | AIS provides 19 typed operations with compile-time verification | Theory     | 3_type_verification                 |
| C3 | Scheduler overhead ~7.5μs (0.0004% of LLM latency)              | Evaluation | Runtime microbenchmark              |
| C4 | Dataflow enables automatic parallelism (4x+ speedup)            | Evaluation | 1_parallel_research, 4_scalability |
| C5 | FuseReasoning reduces LLM calls by 5x                           | Evaluation | 2_chain_fusion                      |
| C6 | 12.6x fewer lines vs LangGraph                                  | Comparison | All 10 workloads                    |
| C7 | 64x faster error detection (compile-time vs runtime)            | Comparison | 3_type_verification                 |

---

## Experiment Groups

### Group A: Substrate Properties (AAM/AIS)

**Purpose**: Validate the formal model, not performance

| Experiment            | What it Shows                     | Metric                           |
|-----------------------|-----------------------------------|----------------------------------|
| A1: Type Safety       | Compile-time error detection      | Time-to-error, LLM calls wasted  |
| A2: Memory Tiers      | STM/LTM/Episodic access patterns  | Operations per tier, correctness |
| A3: Operation Coverage| All 19 AIS operations work        | Pass/fail per operation          |

**Benchmarks**: 3_type_verification, 5_memory_augmented

### Group B: Runtime Overhead

**Purpose**: Show overhead is negligible vs LLM latency

| Experiment           | What it Shows               | Metric       |
|----------------------|-----------------------------|--------------|
| B1: Scheduler μ-bench| Per-node dispatch overhead  | μs/node      |
| B2: Memory Access    | Tier access latency         | μs/operation |
| B3: Token Passing    | Dataflow token overhead     | μs/token     |

**Target**: Overhead < 0.1% of LLM latency (~2000ms)

### Group C: Parallelism Efficiency

**Purpose**: Validate automatic parallelism from dataflow

| Experiment           | What it Shows            | Metric                      |
|----------------------|--------------------------|-----------------------------|
| C1: N-way Parallel   | Speedup at N=2,4,8       | Actual/Theoretical speedup  |
| C2: Research Pattern | 3-way parallel research  | End-to-end latency          |
| C3: Efficiency Curve | Scaling behavior         | Efficiency %                |

**Benchmarks**: 1_parallel_research, 4_scalability

### Group D: FuseReasoning Optimization

**Purpose**: Validate compiler-level LLM call reduction

| Experiment              | What it Shows          | Metric                    |
|-------------------------|------------------------|---------------------------|
| D1: Call Reduction      | N RSN → 1 LLM          | LLM calls O0 vs O1        |
| D2: Latency Speedup     | End-to-end improvement | Speedup factor            |
| D3: Quality Preservation| Output equivalence     | Semantic similarity score |
| D4: Task Type Impact    | Where fusion works     | Per-category speedup      |

**Benchmarks**: 2_chain_fusion (primary)

### Group E: Comparison with LangGraph

**Purpose**: Head-to-head framework comparison

| Experiment             | What it Shows          | Metric            |
|------------------------|------------------------|-------------------|
| E1: Lines of Code      | DSL expressiveness     | LOC ratio         |
| E2: Error Detection    | Compile vs runtime     | Time-to-error ratio|
| E3: End-to-End Latency | Real workflow speed    | ms, speedup       |
| E4: LLM Cost           | API call efficiency    | Calls, $          |

**Benchmarks**: All 10 workloads

---

## Benchmark-to-Claim Mapping

| Benchmark             | Primary Claim          | Secondary Claims | What it Demonstrates                       |
|-----------------------|------------------------|------------------|-------------------------------------------|
| 1_parallel_research   | C4 (auto-parallelism)  | C6 (LOC)         | Dataflow extracts parallelism automatically|
| 2_chain_fusion        | C5 (5x fewer calls)    | C4               | FuseReasoning compiler pass works          |
| 3_type_verification   | C7 (64x faster errors) | C2               | Compile-time type checking saves cost      |
| 4_scalability         | C4 (4x+ speedup)       | -                | Parallelism efficiency at scale            |
| 5_memory_augmented    | C1 (AAM memory)        | C6               | 3-tier memory model is usable              |
| 6_tool_invocation     | C2 (typed ops)         | C6               | Native INV operation                       |
| 7_reflection          | C2 (typed ops)         | C6               | Native REFLECT operation                   |
| 8_planning            | C2 (typed ops)         | C6               | Native PLAN operation                      |
| 9_conditional_routing | C4 (parallelism)       | C6               | Dataflow routing vs conditional edges      |
| 10_multi_agent        | C2 (typed ops)         | C6               | COMMUNICATE operation                      |

---

## Proposed Figures

### Figure 1: Scheduler Overhead Distribution

**Type**: Box plot / Histogram
**X-axis**: Measurement iteration
**Y-axis**: Dispatch latency (μs)
**Shows**: Overhead is ~7.5μs, orders of magnitude below LLM latency (~2000ms)
**Key insight**: "Overhead is 0.0004% of LLM latency"

### Figure 2: Parallelism Efficiency Curve

**Type**: Line chart with error bars
**X-axis**: Parallelism level (N = 2, 4, 8)
**Y-axis**: Efficiency (%) = (Actual Speedup / Theoretical Speedup) × 100
**Lines**: A-PXM (solid), LangGraph (dashed), Ideal (dotted at 100%)
**Shows**: A-PXM maintains higher efficiency as N increases

### Figure 3: FuseReasoning Speedup

**Type**: Grouped bar chart
**X-axis**: Workflow (or chain length)
**Y-axis**: Latency (ms)
**Groups**: O0 (unfused), O1 (fused)
**Annotation**: "5x speedup" arrow between bars

### Figure 4: LLM Calls Comparison

**Type**: Stacked bar chart
**X-axis**: Benchmark workload (1-10)
**Y-axis**: LLM API calls
**Stacks**: A-PXM (blue), LangGraph (orange)
**Shows**: A-PXM uses fewer calls across all workloads

### Figure 5: Lines of Code Comparison

**Type**: Horizontal bar chart
**Y-axis**: Workload name
**X-axis**: Lines of code
**Bars**: A-PXM (blue), LangGraph (orange)
**Annotation**: "12.6x fewer lines on average"

### Figure 6: Error Detection Timeline

**Type**: Timeline/Gantt visualization
**Rows**: A-PXM, LangGraph
**Columns**: Time (ms)
**Shows**: A-PXM catches error at compile time (50ms), LangGraph at runtime (3200ms after 1 LLM call)

---

## Proposed Tables

### Table 1: AAM State Model Validation

| Component        | Operation          | Count | Example                |
|------------------|--------------------|-------|------------------------|
| Beliefs (B)      | qmem stm           | X     | Query recent context   |
| Goals (G)        | plan               | X     | Decompose task         |
| Capabilities (C) | inv                | X     | Tool invocation        |
| STM              | qmem/umem stm      | X     | Fast context           |
| LTM              | qmem/umem ltm      | X     | Persistent knowledge   |
| Episodic         | qmem/umem episodic | X     | Audit trail            |

### Table 2: Type Error Detection Comparison

| Metric                  | A-PXM        | LangGraph   | Improvement    |
|-------------------------|--------------|-------------|----------------|
| Detection time          | ~50ms        | ~3200ms     | 64x faster     |
| LLM calls before error  | 0            | 1           | 1 call saved   |
| Cost of error           | $0.00        | ~$0.01+     | 100% savings   |
| Error location          | Compile-time | Runtime     | Earlier        |

### Table 3: Runtime Overhead Breakdown

| Component          | Latency      | % of LLM Call |
|--------------------|--------------|---------------|
| Token dispatch     | X μs         | X%            |
| Node scheduling    | X μs         | X%            |
| Memory access      | X μs         | X%            |
| **Total overhead** | **~7.5 μs**  | **0.0004%**   |
| LLM call (baseline)| ~2000 ms     | 100%          |

### Table 4: Parallelism Efficiency

| N | Theoretical | A-PXM Actual | Efficiency | LangGraph |
|---|-------------|--------------|------------|-----------|
| 2 | 2.00x       | X.XXx        | XX%        | X.XXx     |
| 4 | 4.00x       | X.XXx        | XX%        | X.XXx     |
| 8 | 8.00x       | X.XXx        | XX%        | X.XXx     |

### Table 5: FuseReasoning Results

| Workflow     | Chains | O0 (unfused) | O1 (fused) | Speedup | Calls Saved |
|--------------|--------|--------------|------------|---------|-------------|
| chain_fusion | 5      | X ms         | X ms       | X.Xx    | 4           |

### Table 6: Fusion Applicability by Task Type

| Task Type             | Fusable  | Quality Impact | Recommendation   |
|-----------------------|----------|----------------|------------------|
| Classification        | Yes      | None           | Recommended      |
| Extraction            | Yes      | Minimal        | Recommended      |
| Multi-step reasoning  | Partial  | Moderate       | Case-by-case     |
| Creative/Open-ended   | No       | Significant    | Not recommended  |

### Table 7: A-PXM vs LangGraph Summary

| Metric              | A-PXM     | LangGraph | Ratio           |
|---------------------|-----------|-----------|-----------------|
| Avg Lines of Code   | X         | X         | 12.6x fewer     |
| Error detection     | Compile   | Runtime   | 64x faster      |
| LLM calls (avg)     | X         | X         | 5x fewer        |
| Parallelism         | Automatic | Manual    | Implicit        |
| Type safety         | Static    | Dynamic   | Compile-time    |

---

## Statistical Requirements

Based on PXM literature patterns:

1. **Iterations**: Minimum 10 iterations per measurement (warmup: 3)
2. **Metrics**: Report mean, std, p50, min, max
3. **Confidence**: 95% confidence intervals for key claims
4. **Effect size**: Report Cohen's d for speedup claims
5. **Variance**: Coefficient of variation (CV) < 10% for stable measurements

### Reporting Format

```text
Mean: X.XX ms (±Y.YY ms, 95% CI)
Speedup: X.Xx (p < 0.001)
```

---

## Interpretation Guidelines

### For Overhead Claims (C3)

- **Success**: Overhead < 100μs (0.005% of LLM latency)
- **Excellent**: Overhead < 10μs (0.0005% of LLM latency)
- **Key message**: "Overhead is negligible; LLM latency dominates"

### For Parallelism Claims (C4)

- **Success**: Efficiency > 50% at N=4
- **Excellent**: Efficiency > 70% at N=4
- **Key message**: "Dataflow enables near-linear scaling"

### For FuseReasoning Claims (C5)

- **Success**: 3x+ speedup with quality preservation
- **Excellent**: 5x+ speedup with <5% quality degradation
- **Key message**: "Compiler optimization reduces LLM costs"

### For Comparison Claims (C6, C7)

- **LOC**: Must show >10x reduction on average
- **Error detection**: Must show >50x faster detection
- **Key message**: "A-PXM improves developer productivity"

---

## Execution Checklist

### Phase 1: Infrastructure

- [x] `apxm_runner.py` - Shared runner module
- [x] CLI `-O` flag for optimization levels
- [ ] Verify all `run.py` scripts use `apxm_runner`

### Phase 2: Data Collection

- [ ] Run all 10 benchmarks with `--json` output
- [ ] Collect 10+ iterations per measurement
- [ ] Record system configuration (CPU, RAM, Ollama version)

### Phase 3: Analysis

- [ ] Compute statistics (mean, std, CI)
- [ ] Generate figures using matplotlib/seaborn
- [ ] Populate table values

### Phase 4: Validation

- [ ] Verify claims match data
- [ ] Cross-check with paper text
- [ ] Fill in \TODO{} and \VERIFY{} placeholders

---

## Files to Update

### Benchmark Scripts (use `apxm_runner`)

All updated to use real CLI execution:

- [x] `2_chain_fusion/run.py`
- [x] `3_type_verification/run.py`
- [x] `4_scalability/run.py`
- [x] `8_planning/run.py`
- [x] `9_conditional_routing/run.py`
- [x] `10_multi_agent/run.py`
- [ ] `1_parallel_research/run.py`
- [ ] `5_memory_augmented/run.py`
- [ ] `6_tool_invocation/run.py`
- [ ] `7_reflection/run.py`

### Paper Files (after data collection)

- `tex/05_evaluation.tex` - Fill in \TODO{} values
- `tab/*.tex` - Populate table numbers
- `fig/*.tex` - Generate figures from data

---

## Measured Results (Dec 2025)

| Metric                 | Target       | Measured           | Status            |
|------------------------|--------------|--------------------|-------------------|
| Substrate overhead     | < 50μs/op    | **7.5μs**          | ✓ Done            |
| Overhead ratio         | < 0.01%      | **0.0004%**        | ✓ Done            |
| STM write/read         | —            | 0.28μs / 0.13μs    | ✓ Done            |
| LTM write/read         | —            | 0.23μs / 0.12μs    | ✓ Done            |
| Episodic write         | —            | 1.36μs             | ✓ Done            |
| Chain fusion (5x)      | Nx reduction | —                  | Pending           |
| Parallelism efficiency | > 70%        | ~85% (claimed)     | Pending           |
| Type verification      | 50+ checks   | **52+ documented** | ✓ Catalog done    |

---

## Commands

```bash
# Substrate overhead benchmark (fast, ~2 seconds)
cargo run --example paper_benchmarks -p apxm-runtime --release

# With JSON output
cargo run --example paper_benchmarks -p apxm-runtime --release -- --json

# Run all workload benchmarks
python papers/cf26/benchmarks/workloads/runner.py --json > results.json

# Run individual benchmark
python papers/cf26/benchmarks/workloads/2_chain_fusion/run.py --json
```
