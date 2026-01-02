# A-PXM CF'26 Testing Plan

Executable checklist that ties **paper claims (C1–C7)** to **commands** and **success criteria**.

## Related Documentation

- [Benchmark Architecture](benchmarks/docs/ARCHITECTURE.md) - Execution flow
- [JSON Output Schema](benchmarks/docs/JSON_SCHEMA.md) - Output format
- [CODEX_PROMPT.md](CODEX_PROMPT.md) - Evaluation guide

---

## Success Criteria

1. **Evidence completeness**: Every `\TODO{}` / `\VERIFY{}` in paper maps to JSON output
2. **Implementation alignment**: Plan matches current prototype
3. **Reproducibility**: Third party can re-run and get results within CI

---

## Configuration

### LLM Model

- **Model**: `gpt-oss:20b-cloud` (via Ollama)

Config files:

- `.apxm/config.toml` - A-PXM runtime
- `papers/cf26/benchmarks/config.json` - Benchmark harness

### Output Directory

```text
papers/cf26/benchmarks/results/YYYYMMDD_HHMMSS/
├── paper_benchmarks.json  # C3 + memory
├── workloads.json         # C1, C4-C7
└── loc.json               # C6
```

---

## Phase 0: Setup

```bash
# Install toolchain
cargo run -p apxm-cli -- install
pip install typer rich langgraph langchain-ollama

# Add apxm to PATH (add to ~/.zshrc or ~/.bashrc for persistence)
export PATH="$PATH:$(pwd)/bin"

# Build
apxm build

# Check environment
apxm doctor

# Start Ollama
ollama serve
ollama pull gpt-oss:20b-cloud
```

## Phase 0.5: Unit Tests (optional)

```bash
cargo test --workspace
```

## Phase 1: Runtime Microbenchmarks (C3)

```bash
cargo run -p apxm-runtime --example paper_benchmarks --release -- --json \
  > papers/cf26/benchmarks/results/YYYYMMDD_HHMMSS/paper_benchmarks.json
```

## Phase 2: Workload Suite (C1, C4-C7)

```bash
# Check all workloads compile
apxm workloads check

# Run all benchmarks
python papers/cf26/benchmarks/run_all.py --workloads --iterations 10 --warmup 3 \
  > papers/cf26/benchmarks/results/YYYYMMDD_HHMMSS/workloads.json

# Or specific workload
apxm workloads run 2 --json -n 10
```

## Phase 3: LOC Comparison (C6)

```bash
python papers/cf26/benchmarks/workloads/loc_comparison/count.py \
  > papers/cf26/benchmarks/results/YYYYMMDD_HHMMSS/loc.json
```

---

## Claims Checklist

### C1 — AAM State Model + 3-Tier Memory

- **Workload**: `5_memory_augmented`
- **Success**: AAM beliefs/goals/capabilities inspectable, memory tiers work
- **Tables**: `tab/aam-inspection.tex`, `tab/memory-ops.tex`

### C2 — 19 AIS Operations + Compile-Time Verification

- **Workload**: `3_type_verification` (52+ checks)
- **Success**: Invalid programs fail at compile-time, 0 LLM calls wasted
- **Tables**: `tab/type-errors.tex`

### C3 — Scheduler Overhead <10μs

- **Source**: `paper_benchmarks.json`
- **Success**: `per_op_overhead_us < 10` on `--release`
- **Tables**: `tab/runtime-overhead.tex`

### C4 — Automatic Parallelism (4x speedup, >70% efficiency)

- **Workloads**: `1_parallel_research`, `4_scalability`
- **Success**: Efficiency >70% at N=4
- **Figures**: `fig/speedup-plot.tex`, `fig/dataflow-extraction.tex`

### C5 — FuseReasoning (5x fewer calls)

- **Workload**: `2_chain_fusion` (O0 vs O1)
- **Success**: 5 RSN → 1 call, speedup matches claim
- **Tables**: `tab/fusion-applicability.tex`

### C6 — LOC Reduction (>10x)

- **Source**: `loc.json`
- **Success**: Average ratio >10x across workloads
- **Tables**: `tab/apxm-vs-langgraph.tex`

### C7 — Faster Error Detection (>50x)

- **Workload**: `3_type_verification`
- **Success**: Compile-time failure >> runtime failure
- **Tables**: `tab/apxm-vs-langgraph.tex`

---

## Workload Status

| # | Name | Status |
|---|------|--------|
| 1 | `parallel_research` | Active |
| 2 | `chain_fusion` | Active |
| 3 | `type_verification` | Active |
| 4 | `scalability` | Active |
| 5 | `memory_augmented` | Active |
| 6 | `tool_invocation` | **Disabled** |
| 7 | `reflection` | Active |
| 8 | `planning` | Active |
| 9 | `conditional_routing` | Active |
| 10 | `multi_agent` | Active |
| 11 | `compilation_scaling` | Active |
| 12 | `real_llm_probe` | Active |
| 13 | `fusion_quality` | Active |
| 14 | `token_estimation` | Active |

---

## AIS Operation Coverage

### Core 19 Operations (Paper Claim)

`QMEM`, `UMEM`, `RSN`, `PLAN`, `REFLECT`, `VERIFY`, `INV`, `EXC`, `JUMP`, `BRANCH_ON_VALUE`, `LOOP_START`, `LOOP_END`, `RETURN`, `MERGE`, `FENCE`, `WAIT_ALL`, `TRY_CATCH`, `ERR`, `COMMUNICATE`

### Extensions (Not in 19 count)

- `SWITCH`, `FLOW_CALL` - artifact-emittable
- `AGENT` - metadata
- `CONST_STR` - compiler internal

### Prototype Limitations

- `COMMUNICATE`: acknowledgment stub (no transport)
- `FLOW_CALL`: records invocation, doesn't execute target
- `EXC`: raises exception (not sandboxed execution)
- `TRY_CATCH` / `ERR`: minimal placeholder

---

## Paper Table → Command Map

| Table/Figure | Source |
|--------------|--------|
| `tab/runtime-overhead.tex` | `paper_benchmarks.json` |
| `tab/memory-ops.tex` | `paper_benchmarks.json` |
| `tab/aam-inspection.tex` | `workloads.json` → workload 5 |
| `tab/type-errors.tex` | `workloads.json` → workload 3 |
| `tab/apxm-vs-langgraph.tex` | `loc.json` + workload 3 |
| `tab/fusion-applicability.tex` | workload 2 |
| `fig/speedup-plot.tex` | workloads 1, 4 |

---

## Missing Harnesses (TODO)

- [ ] Real-LLM probe script → `tab/real-llm.tex`
- [ ] Compilation profiling → `tab/compilation-scaling.tex`
- [ ] Synthetic DAG (1..32 parallelism) → 32-way `\VERIFY{}`
- [ ] FuseReasoning quality by task → `tab/fusion-applicability.tex`

---

## Pre-Submission Checklist

- [ ] `rg "\\\\(TODO|VERIFY)\\{"` returns empty in paper repo
- [ ] "Threats to Validity" section present
- [ ] "Prototype scope" table present
- [ ] Single-model MVP limitation stated
