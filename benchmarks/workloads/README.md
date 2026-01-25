# A-PXM Benchmark Workloads

DSL comparison benchmarks between A-PXM (AIS DSL) and LangGraph.

## Quick Start

```bash
# Run a single benchmark by folder name
apxm benchmarks run --benchmark 1_parallel_research

# Quick mode (fewer iterations)
apxm benchmarks run --benchmark 1_parallel_research --quick

# JSON output only
apxm benchmarks run --benchmark 1_parallel_research --json

# Run everything (workloads + runtime)
apxm benchmarks run

# Run only DSL comparison workloads
python3 run_all.py --workloads

# Run only Rust runtime benchmarks
python3 run_all.py --runtime
```

## Overview

This benchmark suite measures the developer experience and performance differences between:
- **A-PXM**: Custom agent DSL with MLIR compiler
- **LangGraph**: Python embedded DSL for agent workflows

Each workload is implemented in both DSLs to enable fair comparison.

### Timing Fairness (What we compare)

- **LangGraph**: `mean_ms` is measured as in-process `graph.invoke(...)` wall time (warmups excluded).
- **A-PXM**: `mean_ms` is subprocess wall time for `apxm execute ...` (includes process + compile + runtime).
  For apples-to-apples comparison against LangGraph invoke time, prefer **A-PXM internal runtime**:
  `metrics.runtime_ms.mean_ms` (requires `--emit-metrics`, enabled by default).

The analysis scripts and the `workloads/runner.py` summary prefer **A-PXM runtime_ms** when available.

## Directory Structure

```
workloads/
├── 1_parallel_research/     # Automatic parallelism from dataflow
├── 2_chain_fusion/          # FuseAskOps compiler optimization
├── 3_type_verification/     # Compile-time vs runtime error detection
├── 4_scalability/           # N-way parallelism efficiency
├── 5_memory_augmented/      # 3-tier memory (STM/LTM/Episodic)
├── 6_tool_invocation/       # Native INV operations
├── 7_reflection/            # Built-in reflect(operation)
├── 8_planning/              # Native plan(operation)
├── 9_conditional_routing/   # Dataflow-based routing
├── 10_multi_agent/          # Multi-agent collaboration
├── apxm_runner.py           # Consolidated benchmark runner (WorkloadConfig registry)
├── runner.py                # Master benchmark runner (thin wrapper)
├── dataset_eval.py          # Dataset evaluation utilities (F1, accuracy)
└── README.md                # This file
```

Each workload folder contains:
- `workflow.ais` - A-PXM implementation
- `workflow.py` - LangGraph implementation (where applicable)
- `README.md` - Workload-specific documentation
- `expect_error` - (optional) Marker file for error-testing workloads

### Error-Testing Workloads

Some workloads (e.g., `3_type_verification`) are designed to test compile-time error detection.
These contain intentionally broken code and include an `expect_error` marker file.

When running `apxm workloads check`, these workloads:
- **Pass** if compilation fails (expected behavior)
- **Fail** if compilation succeeds (indicates a bug in the verifier)

## Testing Infrastructure

### Hardware Configuration

| Component | Details |
|-----------|---------|
| **GPU** | 4x NVIDIA A100-SXM4-40GB (40GB VRAM each) |
| **Driver** | NVIDIA 560.35.03 |
| **CPU** | AMD EPYC 75F3 32-Core Processor |
| **RAM** | 755 GB |
| **OS** | Ubuntu 24.04.3 LTS (Kernel 6.1.0-25-amd64) |
| **Ollama Version** | 0.14.2 |
| **Model** | `gpt-oss:120b` (65GB, loaded across GPUs) |
| **Context Window** | 131,072 tokens (`num_ctx` in config) |

### Comparison Methodology

Both frameworks are tested under identical conditions to ensure fair comparison:

1. **Same LLM Backend**: Both A-PXM and LangGraph use the same Ollama endpoint with identical model (`gpt-oss:120b`)
2. **Same Context Window**: 131,072 tokens (`num_ctx`)
3. **Same System Prompts**: Loaded from shared `[instruction]` config section in `~/.apxm/config.toml`
4. **Same Hardware**: All benchmarks run on the same machine

**Measurement Approach**:
- **LangGraph**: In-process `graph.invoke()` wall time (excludes Python interpreter startup)
- **A-PXM**: Internal `runtime_ms` from metrics (excludes subprocess spawn + compilation overhead)
  - For subprocess wall time, use `execution_time_ms` instead
- **LLM Metrics**: Captured via shared instrumentation module (`llm_instrumentation.py`)

### Workload Interpretation Guide

| Workload | Fair Comparison? | What to Report | Why |
|----------|------------------|----------------|-----|
| chain_fusion | Yes | Speedup (1.3x) | Fusion reduces LLM calls |
| conditional_routing | Yes | Speedup (5x) | Dataflow-based parallelism |
| multi_agent | Yes | Speedup (10x) | Native agent operations |
| planning | **No (LLM-bound)** | Code simplicity, native operations | LLM inference > 90% of runtime |
| scalability_n* | **No (LLM-bound)** | Automatic parallelization | Both hit same LLM API limits |

### Assumptions and Limitations

1. **LLM Inference Time Variability**: Inference times vary by 10-50% between runs. Both frameworks are bounded by the same LLM API rate limits.

2. **Fusion Quality Trade-off**: FuseAskOps combines sequential `ask()` calls into batched prompts. This changes prompt structure and may affect response quality. The paper documents this trade-off (Section 5, line 77).

3. **LLM-Bound Workloads**: For planning and scalability workloads, LLM time exceeds 90% of total runtime. Speedup differences in these cases reflect subprocess/compilation overhead, not parallelism inefficiency.

4. **Framework Overhead Metrics**: The `framework_overhead_ms` metric isolates framework-specific costs by subtracting LLM latency from total time. Use this for apples-to-apples comparison.

## Prerequisites

### 1. Configure LLM Backend

LangGraph benchmarks read the same `.apxm/config.toml` as the runtime. Example Ollama config:

```toml
[chat]
providers = ["ollama"]
default_model = "ollama"

[[llm_backends]]
name = "ollama"
provider = "ollama"
model = "gpt-oss:20b-cloud"
endpoint = "http://localhost:11434"

[llm_backends.options]
# Context window size (default is 8192 for most models)
# Increase this if your model supports larger context and you have enough GPU memory
num_ctx = "32768"
```

OpenAI example:

```toml
[chat]
providers = ["openai"]
default_model = "openai"

[[llm_backends]]
name = "openai"
provider = "openai"
model = "gpt-4o-mini"
api_key = "env:OPENAI_API_KEY"
```

vLLM (OpenAI-compatible) example:

```toml
[chat]
providers = ["vllm"]
default_model = "vllm"

[[llm_backends]]
name = "vllm"
provider = "openai"
model = "your-model"
endpoint = "http://localhost:8000/v1"
api_key = "env:VLLM_API_KEY"
```

### 2. Configure Benchmark Settings (Optional)

You can customize benchmark execution settings in `~/.apxm/config.toml`:

```toml
[benchmarks]
timeout_seconds = 300.0   # 5 minutes (default: 120.0)
iterations = 10           # number of benchmark iterations (default: 10)
warmup = 3                # warmup iterations before measurement (default: 3)
```

These can also be overridden via environment variables:
- `APXM_BENCH_TIMEOUT` - timeout in seconds
- `APXM_BENCH_ITERATIONS` - number of iterations
- `APXM_BENCH_WARMUP` - number of warmup iterations

If using Ollama, install and start it:

```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve

# Pull the gpt-oss:20b-cloud model (used by all benchmarks)
ollama pull gpt-oss:20b-cloud
```

### 2. Install Python Dependencies

```bash
cd papers/cf26/benchmarks
pip install -r requirements.txt
```

### 3. Build A-PXM Compiler (REQUIRED for dataset workloads)

The A-PXM CLI must be built with the `driver` feature for dataset workloads (11_hotpotqa, 12_parallelqa) to run:

```bash
cd /path/to/apxm

# Using Python CLI (recommended - handles environment automatically)
apxm compiler build

# Or manually (requires activated conda environment)
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"
cargo build -p apxm-cli --features driver --release
```

**IMPORTANT**: If you see the error:
```
Error: Command requires the `driver` feature. Re-run with: cargo run -p apxm-cli --features driver -- <command>
```
This means the CLI was not built with the driver feature. Run the build command above to fix it.

## Running Benchmarks

### Using Python CLI (Recommended)

The Python CLI provides convenient commands for managing workloads:

```bash
cd /path/to/apxm

# List available workloads
apxm workloads list

# Check all workloads compile correctly
apxm workloads check

# Run a specific workload benchmark
apxm workloads run 10_multi_agent
apxm workloads run 1_parallel_research

# Run benchmarks with iterations/warmup control
apxm workloads benchmark 2_chain_fusion
apxm workloads benchmark --all --json -o results.json
apxm workloads benchmark --all -n 10 -w 3
```

### Run All Workloads

```bash
# JSON output (for analysis)
apxm benchmarks run --workloads --json

# Human-readable output
apxm benchmarks run --workloads

# Specify iterations (DSL workloads only)
apxm workloads benchmark --all -n 20
```

### Suite Runner (Recommended)

```bash
cd papers/cf26/benchmarks

# Run all workloads + runtime benchmarks
apxm benchmarks run

# Save per-workload JSONs to a custom directory
apxm benchmarks run --output-dir results/run_custom
```

### Run Individual Workload

```bash
# By name
apxm workloads run 1_parallel_research

# By number
apxm workloads run 1

# With JSON output
apxm workloads run 1 --json
```

## Workload Details

### 1. Parallel Research
**Measures**: Automatic parallelism efficiency

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Parallelism | Automatic from dataflow | Requires explicit Send API |
| Lines of code | ~12 | ~45 |

### 2. Chain Fusion
**Measures**: FuseAskOps compiler optimization

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| LLM calls | 1 (fused) | 5 (sequential) |
| Latency | ~2s | ~10s |

### 3. Type Verification
**Measures**: Error detection timing

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Error detection | Compile-time | Runtime |
| Cost of error | $0.00 | ~$0.15 |

### 4. Scalability
**Measures**: Parallelism efficiency at N=2,4,8

| N | Theoretical | Expected Efficiency |
|---|-------------|---------------------|
| 2 | 2x | ~85% |
| 4 | 4x | ~75% |
| 8 | 8x | ~60% |

### 5. Memory Augmented
**Measures**: Memory tier access patterns

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Memory ops | Native qmem/umem | Manual state dict |
| Tiers | STM, LTM, Episodic | Checkpoint system |

### 6. Tool Invocation
**Measures**: Tool call overhead

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Tool calls | Native inv | LangChain tools |
| Validation | Compile-time | Runtime |

### 7. Reflection
**Measures**: Self-improvement loop overhead

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Reflection | Native reflect(op | Custom prompting |)| Output format | Structured | Unstructured |

### 8. Planning
**Measures**: Task decomposition

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Planning | Native plan(op | Custom prompting |)| Execution | Auto-parallelism | Manual |

### 9. Conditional Routing
**Measures**: Dynamic control flow efficiency

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Routing | Parallel preparation | Runtime conditional edges |
| Parallelism | Automatic | Sequential |

### 10. Multi-Agent
**Measures**: Agent coordination overhead

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Agents | Native definitions | Function nodes |
| Messaging | communicate op | Shared state |

## AIS Operations Reference

Valid operations in the AIS DSL:

| Operation | Aliases | Purpose |
|-----------|---------|---------|
| `ask` | - | Simple Q&A with LLM |
| `think` | - | Extended thinking with budget |
| `reason` | - | Structured reasoning with belief updates |
| `plan` | - | Task decomposition |
| `reflect` | - | Self-analysis |
| `inv` | invoke, tool | Tool invocation |
| `qmem` | query_memory, mem | Query memory tier |
| `umem` | update_memory | Update memory tier |
| `merge` | - | Merge parallel results |
| `wait` | wait_all | Await async operations |
| `verify` | - | Verification check |
| `execute` | exc, exec | Execute action |
| `communicate` | talk | Inter-agent messaging |

### Memory Tiers

- `stm` - Short-term memory (fast, temporary)
- `ltm` - Long-term memory (persistent)
- `episodic` - Episodic memory (audit trail)

## Output Format

All runners output JSON with this structure:

```json
{
  "meta": {
    "timestamp": "2025-12-26T10:30:00Z",
    "benchmark": "parallel_research"
  },
  "config": {
    "iterations": 10,
    "warmup": 3
  },
  "results": {
    "langgraph": {
      "mean_ms": 3250.2,
      "std_ms": 145.3,
      "p50_ms": 3180.0,
      "has_ollama": true
    },
    "apxm": {
      "note": "Compilation not yet integrated"
    }
  }
}
```

## Step-by-Step Benchmarking Guide

### Step 1: Verify Ollama is Running

```bash
ollama list
# Should show: gpt-oss:20b-cloud
```

### Step 2: Test Single Workload

```bash
cd 1_parallel_research
python workflow.py
# Should see LLM responses
```

### Step 3: Run Individual Benchmark

```bash
apxm workloads run 1_parallel_research
# Shows timing results
```

### Step 4: Run Full Suite

```bash
cd ..
python runner.py --json > results_$(date +%Y%m%d).json
```

### Step 5: Analyze Results

```bash
# View summary
python runner.py

# Parse JSON for detailed analysis
cat results.json | jq '.results[].langgraph.mean_ms'
```

## Interpreting Results

### Key Metrics

1. **Mean latency (ms)**: Average execution time
2. **Std deviation (ms)**: Consistency of timing
3. **P50 (ms)**: Median latency
4. **LLM calls**: Number of API calls made
5. **Efficiency (%)**: Actual speedup / theoretical speedup

### Expected Outcomes

| Workload | A-PXM Advantage |
|----------|-----------------|
| Parallel Research | 3x speedup from auto-parallelism |
| Chain Fusion | 5x fewer LLM calls |
| Type Verification | $0.15 saved per error |
| Scalability | Higher efficiency at all N |
| Memory | Native tier support |

## Troubleshooting

### Ollama Not Available

Benchmarks require real LLM calls. Install Ollama and `langchain-ollama` before running.

### Import Errors

```bash
pip install langgraph langchain-ollama
```

### Timeout Issues

Reduce benchmark load if LLM responses are slow:

- Per-workload: `apxm workloads run 1 -n 3 -w 1`
- Whole suite: `python runner.py --iterations 3 --warmup 1`
- Or via env: `APXM_BENCH_ITERATIONS=3 APXM_BENCH_WARMUP=1 python runner.py`

## Runtime Benchmarks

These benchmarks generate data for the CF'26 paper tables and figures.

### Rust Substrate Overhead (`../runtime/paper_benchmarks.rs`)
Measures runtime scheduler overhead and parallelism scaling.
- **Output**: `tab/runtime-overhead.tex`, `fig/speedup-plot.tex`
- **Metrics**: Per-op overhead (μs), Parallelism efficiency at N=2,4,8,16,32

## Contributing

To add a new workload:

1. Create `N_workload_name/` directory
2. Add `workflow.ais` with valid AIS operations
3. Add `workflow.py` with LangGraph equivalent
4. Add `README.md` with Purpose, What We're Demonstrating, How to Run, Results, Analysis sections
5. Add `WorkloadConfig` entry to `apxm_runner.py` WORKLOADS registry
