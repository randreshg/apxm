# A-PXM Benchmark Workloads

DSL comparison benchmarks between A-PXM (AIS DSL) and LangGraph.

## Overview

This benchmark suite measures the developer experience and performance differences between:
- **A-PXM**: Custom agent DSL with MLIR compiler
- **LangGraph**: Python embedded DSL for agent workflows

Each workload is implemented in both DSLs to enable fair comparison.

## Directory Structure

```
workloads/
├── 1_parallel_research/     # Automatic parallelism from dataflow
├── 2_chain_fusion/          # FuseReasoning compiler optimization
├── 3_type_verification/     # Compile-time vs runtime error detection
├── 4_scalability/           # N-way parallelism efficiency
├── 5_memory_augmented/      # 3-tier memory (STM/LTM/Episodic)
├── 6_tool_invocation/       # Native INV operations
├── 7_reflection/            # Built-in reflect operation
├── 8_planning/              # Native plan operation
├── 9_conditional_routing/   # Dataflow-based routing
├── 10_multi_agent/          # Multi-agent collaboration
├── runner.py                # Master benchmark runner
└── README.md                # This file
```

Each workload folder contains:
- `workflow.ais` - A-PXM implementation
- `workflow.py` - LangGraph implementation
- `run.py` - Individual benchmark runner
- `README.md` - Workload-specific documentation

## Prerequisites

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve

# Pull the phi3:mini model (used by all benchmarks)
ollama pull phi3:mini
```

### 2. Install Python Dependencies

```bash
cd papers/cf26/benchmarks/workloads
pip install langgraph langchain-ollama
```

### 3. Build A-PXM Compiler (optional, for DSL compilation)

```bash
cd /path/to/apxm
cargo build --release -p apxm
```

## Running Benchmarks

### Run All Workloads

```bash
# JSON output (for analysis)
python runner.py --json > results.json

# Human-readable output
python runner.py

# Specify iterations
python runner.py --iterations 20
```

### Run Individual Workload

```bash
cd 1_parallel_research
python run.py --json
```

## Workload Details

### 1. Parallel Research
**Measures**: Automatic parallelism efficiency

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Parallelism | Automatic from dataflow | Requires explicit Send API |
| Lines of code | ~12 | ~45 |

### 2. Chain Fusion
**Measures**: FuseReasoning compiler optimization

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
| Reflection | Native reflect op | Custom prompting |
| Output format | Structured | Unstructured |

### 8. Planning
**Measures**: Task decomposition

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Planning | Native plan op | Custom prompting |
| Execution | Auto-parallelism | Manual |

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
| `rsn` | reason, think, llm | LLM reasoning call |
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
# Should show: phi3:mini
```

### Step 2: Test Single Workload

```bash
cd 1_parallel_research
python workflow.py
# Should see LLM responses
```

### Step 3: Run Individual Benchmark

```bash
python run.py
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

If `HAS_OLLAMA = False`, benchmarks use mock responses. Install Ollama for real LLM measurements.

### Import Errors

```bash
pip install langgraph langchain-ollama
```

### Timeout Issues

Increase timeout in run.py if LLM responses are slow:
```python
BENCHMARK_ITERATIONS = 5  # Reduce iterations
```

## Contributing

To add a new workload:

1. Create `N_workload_name/` directory
2. Add `workflow.ais` with valid AIS operations
3. Add `workflow.py` with LangGraph equivalent
4. Add `run.py` benchmark runner
5. Update `runner.py` to include new workload
