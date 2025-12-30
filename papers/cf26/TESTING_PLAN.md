# A‑PXM CF'26 Testing Plan (Runbook)

This document is the **minimum** executable checklist that ties **paper claims (C1–C7)** to **commands**, **outputs**, and **success criteria**. The goal is that once the environment is ready, the only remaining work is to **run the commands** and paste the resulting numbers into `tab/*.tex` / `fig/*.tex`.

## Related Documentation

- [Benchmark Architecture](benchmarks/docs/ARCHITECTURE.md) - Execution flow and design decisions
- [JSON Output Schema](benchmarks/docs/JSON_SCHEMA.md) - Complete output format reference
- [Compiler Diagnostics](benchmarks/docs/COMPILER_DIAGNOSTICS.md) - `--emit-diagnostics` format
- [Runtime Metrics](benchmarks/docs/RUNTIME_METRICS.md) - `--emit-metrics` format

## Scope and Success Criteria

1. **Evidence completeness**: Every quantitative number in the paper (including every `\TODO{}` / `\VERIFY{}` value in `tab/*.tex` and `fig/*.tex`) maps to a specific command and raw JSON output under `papers/cf26/benchmarks/results/`.
2. **Implementation alignment**: The plan and paper text match the current prototype (including explicit prototype limitations).
3. **Reproducibility**: A third party can re-run the evaluation from scratch using this runbook and get results within the stated confidence intervals.

## Evaluation Assumptions (MVP)

- **Single model per run**: the current MVP evaluates using **one configured default LLM backend/model** per run (multi-model selection is future work).
- **No “Rust vs Python” performance claim**: cross-framework comparisons are **workflow-level** (LLM calls/tokens, critical-path latency under the *same* LLM backend/model), not language/runtime microbenchmarks.
- **Statistical policy is enforced by the harness**: workloads and suite runners accept `--warmup` and `--iterations` and also honor `APXM_BENCH_WARMUP` / `APXM_BENCH_ITERATIONS`.
- **Warmup purpose**: warmup iterations are **excluded from statistics** to amortize first-run effects (backend queueing/connection setup, model load, caches, JIT/init, OS page cache); only post-warmup iterations are used for mean/p50/p99 and confidence intervals.

## Configuration (One-time)

### LLM model (paper numbers)

- **Primary**: `gpt-oss:120b-cloud`
- **Fallback**: `gpt-oss:20b-cloud`
- Avoid DeepSeek-family models for the paper run.

Where the model is configured:

- A‑PXM runtime/driver: `.apxm/config.toml`
- Bench harness + LangGraph baselines: `papers/cf26/benchmarks/config.json` (`llm.model`) and `APXM_BENCH_OLLAMA_MODEL`.

### Config files to verify before running

- `.apxm/config.toml` points to the desired provider/model (Ollama cloud preferred).
- `papers/cf26/benchmarks/config.json`:
  - `llm.model` is the primary model.
  - `llm.model_priority` lists fallback models.
  - `llm.warmup` is `3` and `llm.iterations` is `10` (or higher for final paper runs).

## Output Convention (Required)

Store raw results as JSON under a timestamped directory:

- `papers/cf26/benchmarks/results/YYYYMMDD_HHMMSS/`

Minimum files per paper refresh:

- `paper_benchmarks.json` (C3 + memory table inputs)
- `workloads.json` (C1, C4–C7 + supporting evidence)
- `loc.json` (C6)

Each JSON should include (or be accompanied by) the exact model string used + `ollama list` digest.

## Commands (Run Last)

### Phase 0: Toolchain readiness (no measurements yet)

Using Python CLI (recommended - handles environment automatically):

```bash
python tools/apxm_cli.py doctor
python tools/apxm_cli.py compiler build
```

Or manually (requires activated conda environment):

```bash
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"
./target/release/apxm doctor
```

If running real LLM mode: `ollama serve` and verify model availability with `ollama list`.

### Phase 0.5: Crate/unit tests (optional but recommended before final runs)

- `cargo test -p apxm-ais -p apxm-driver -p apxm-runtime -p apxm-backends -p apxm-artifact -p apxm-core -p apxm-cli`

### Phase 1: Runtime microbenchmarks (C3 + memory table inputs)

- Scheduler overhead + memory tier latencies (single JSON output contains both):
  - `cargo run --example paper_benchmarks -p apxm-runtime --release -- --json > papers/cf26/benchmarks/results/YYYYMMDD_HHMMSS/paper_benchmarks.json`

### Phase 2: Workload suite (C1, C4–C7)

**Using Python CLI (recommended):**

```bash
# Check all workloads compile
python tools/apxm_cli.py workloads check

# Run a specific workload
python tools/apxm_cli.py workloads run 10_multi_agent

# Run benchmarks with iteration/warmup control
python tools/apxm_cli.py workloads benchmark 2_chain_fusion -n 10 -w 3
python tools/apxm_cli.py workloads benchmark --all --json -o results.json
```

**Full benchmark run** (honors `papers/cf26/benchmarks/config.json` and selects the first available model from `model_priority` via `ollama list`):

```bash
python papers/cf26/benchmarks/run_all.py --workloads --iterations 10 --warmup 3 --json > papers/cf26/benchmarks/results/YYYYMMDD_HHMMSS/workloads.json
```

Optional: run a single workload for debugging (same warmup/iterations policy):

```bash
python papers/cf26/benchmarks/workloads/runner.py --workload 9 --iterations 10 --warmup 3 --json
```

### Phase 3: LOC comparison (C6)

- `python papers/cf26/benchmarks/workloads/loc_comparison/count.py > papers/cf26/benchmarks/results/YYYYMMDD_HHMMSS/loc.json`

## Claims (C1–C7) Checklist

### C1 — AAM state model + 3-tier memory

- **Source**: workload `5_memory_augmented` output inside `workloads.json`
- **Success**:
  - AAM beliefs/goals/capabilities are inspectable (provide one stable dump for `tab/aam-inspection.tex`).
  - Memory operations behave correctly across STM/LTM/Episodic for the benchmark scenario.
- **Paper targets**: `tab/aam-inspection.tex`, `tab/memory-ops.tex`

### C2 — AIS typed operations + compile-time verification

- **Source**:
  - Workload `3_type_verification`
  - `papers/cf26/benchmarks/workloads/3_type_verification/VERIFICATION_CATALOG.md` (52+ checks)
- **Success**:
  - Invalid programs fail before any LLM calls (compile-time).
  - The paper’s “**19 typed operations**” claim matches the implementation (see “AIS Operation Coverage” below).
- **Paper targets**: `tab/type-errors.tex` + AIS discussion

### C3 — Scheduler overhead

- **Source**: `paper_benchmarks.json` (`table_4_overhead` + `table_5_memory`)
- **Success**:
  - `table_4_overhead.per_op_overhead_us < 10` (target) on `--release`
  - Ratio to LLM latency is computed from the **same** model run used in `tab/real-llm.tex`.
- **Paper targets**: `tab/runtime-overhead.tex`, `tab/real-llm.tex` ratio line

### C4 — Automatic parallelism

- **Source**: workloads `1_parallel_research` and `4_scalability`
- **Success**:
  - Report speedup/efficiency under the same model/backend for both A‑PXM and LangGraph baselines.
  - Efficiency target: `> 70%` at `N=4` (paper thresholds may be revised if the data differs).
- **Paper targets**: `fig/speedup-plot.tex`, `fig/dataflow-extraction.tex`

### C5 — FuseReasoning (5× fewer calls; latency reduction; quality)

- **Source**: workload `2_chain_fusion` (`O0` vs `O1`)
- **Success**:
  - Call reduction: `5 -> 1`
  - Measured speedup meets paper claim or the paper updates to the measured value.
  - Quality results are explicitly treated as **empirical** (see paper threats-to-validity); populate `tab/fusion-applicability.tex`.
- **Paper targets**: `fig/fuse-reasoning-demo.tex`, `tab/fusion-applicability.tex`

### C6 — LOC reduction (>10×)

- **Source**: `loc.json`
- **Success**: average semantic LOC ratio `> 10×` across the 10 workloads.
- **Paper targets**: `tab/apxm-vs-langgraph.tex`

### C7 — Faster error detection (>50×)

- **Source**: workload `3_type_verification`
- **Success**: compile-time failure is orders of magnitude faster than runtime failure and wastes `0` LLM calls for A‑PXM.
- **Paper targets**: `tab/apxm-vs-langgraph.tex`

## AIS Operation Coverage (Paper-facing)

### Operation count to keep consistent everywhere

The paper should present a **core 19-op AIS** (typed, verifier-checked) and treat additional syntax/features as **extensions** unless/until the paper text is updated to count them.

- **Core 19 (paper claim)**: `QMEM`, `UMEM`, `RSN`, `PLAN`, `REFLECT`, `VERIFY`, `INV`, `EXC`, `JUMP`, `BRANCH_ON_VALUE`, `LOOP_START`, `LOOP_END`, `RETURN`, `MERGE`, `FENCE`, `WAIT_ALL`, `TRY_CATCH`, `ERR`, `COMMUNICATE`.
- **Implementation extensions**: `SWITCH`, `FLOW_CALL` (now artifact-emittable; see “What changed recently”).
- **Metadata/internal** (not counted in the “19”): `AGENT` (metadata), `CONST_STR` (compiler internal).

### Prototype limitations (must match paper wording)

These ops exist and are exercised in workloads, but their semantics are prototype-level and must be presented as such:

- `COMMUNICATE`: acknowledgment stub (no transport).
- `FLOW_CALL`: records invocation and returns a structured “invoked” object (does not execute the target agent flow yet).
- `EXC`: currently behaves as “raise exception” (not “execute sandboxed code”).
- `TRY_CATCH` / `ERR`: minimal placeholder behavior.

## Paper Consistency Sweep (Do Before Submission)

In the paper repo:

- `rg -n \"\\\\\\\\(TODO|VERIFY)\\\\\\\\{\"` should be empty (or remaining TODOs are explicitly non-numeric and permitted).
- Ensure the evaluation section contains:
  - “Threats to Validity” (backend variability, prompt sensitivity, caching/warmup; workflow-level comparisons).
  - “Prototype scope” table (implemented vs planned).
- Ensure single-model MVP limitation is stated (and multi-model routing is future work).

## Paper Table/Figure Fill Map (What to run to get each number)

This section is intentionally short; it exists to prevent “where did this number come from?” drift.

- `tab/runtime-overhead.tex` + `tab/memory-ops.tex`: `paper_benchmarks.json` (`table_4_overhead`, `table_5_memory`)
- `tab/aam-inspection.tex`: `workloads.json` → workload `5_memory_augmented` (plus one stable AAM dump snippet)
- `tab/type-errors.tex`: `workloads.json` → workload `3_type_verification` + `VERIFICATION_CATALOG.md`
- `tab/apxm-vs-langgraph.tex`: `loc.json` + `workloads.json` (`3_type_verification` time-to-error, plus any LLM-call counts used)
- `tab/fusion-applicability.tex`: workload `2_chain_fusion` **plus a quality-by-task harness** (not fully wired yet; see below)
- `tab/real-llm.tex`: requires a **real-LLM probe script** that records mean/p99 latency + token usage under `gpt-oss:120b-cloud` / `gpt-oss:20b-cloud` (not fully wired yet; see below)
- `tab/compilation-scaling.tex`: requires a **compiler profiling harness** (not fully wired yet; see below)
- `fig/speedup-plot.tex`: `workloads.json` (`1_parallel_research`, `4_scalability`) plus any synthetic scaling run used for the 32-way `\VERIFY{}` placeholders

### Known missing “paper fill” harness pieces (must exist before final run)

- [ ] Real‑LLM probe script (latency p99 + token usage) to populate `tab/real-llm.tex`
- [ ] Compilation scaling/profiling harness to populate `tab/compilation-scaling.tex`
- [ ] Synthetic DAG parallelism experiment (1..32) if the paper keeps the 32-way `\VERIFY{}` claims
- [ ] FuseReasoning quality-by-task harness (classification/extraction vs reasoning) to populate `tab/fusion-applicability.tex`

## What changed recently (keep this plan in sync)

- Workloads `9_conditional_routing` and `10_multi_agent` are **artifact-emittable** (artifact wire format supports `SWITCH` + `FLOW_CALL`).
- Benchmark scripts enforce `warmup=3` / `iterations=10` by default and allow overrides via flags/env.
