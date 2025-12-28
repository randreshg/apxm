# Chain Fusion Benchmark

Demonstrates FuseReasoning compiler optimization - batching N LLM calls into 1.

## What This Measures

- **A-PXM**: Compiler fuses 5 dependent RSN calls into a single batched prompt
- **LangGraph**: Each node is a separate LLM call (no fusion possible)

## The Optimization

```
WITHOUT FUSION (LangGraph):
┌─────────┐   2s   ┌─────────┐   2s   ┌─────────┐   2s   ┌─────────┐   2s   ┌─────────┐
│ RSN #1  │ ─────► │ RSN #2  │ ─────► │ RSN #3  │ ─────► │ RSN #4  │ ─────► │ RSN #5  │
└─────────┘        └─────────┘        └─────────┘        └─────────┘        └─────────┘

Total: 5 × 2s = 10 seconds
LLM API calls: 5

WITH FUSION (A-PXM FuseReasoning pass):
┌────────────────────────────────────────────────────┐   2s   ┌─────────┐
│ FUSED RSN (single batched prompt)                  │ ─────► │ RESULT  │
│ "Define quantum computing                          │        └─────────┘
│ ---                                                │
│ Using the above, explain qubits                    │
│ ---                                                │
│ Using the above, explain superposition             │
│ ---                                                │
│ Using the above, explain entanglement              │
│ ---                                                │
│ Summarize all concepts above"                      │
└────────────────────────────────────────────────────┘

Total: 1 × 2s = 2 seconds
LLM API calls: 1
SPEEDUP: 5x
```

## Metrics

| Metric | A-PXM | LangGraph |
|--------|-------|-----------|
| LLM calls | 1 | 5 |
| Latency | ~2s | ~10s |
| API cost | ~$0.01 | ~$0.05 |
| Speedup | 5x | 1x |

## Run

```bash
python run.py --json
```
