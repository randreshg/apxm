# Parallel Research Benchmark

Demonstrates automatic parallelism from dataflow dependencies.

## What This Measures

- **A-PXM**: Three RSN operations with no dependencies run in parallel automatically
- **LangGraph**: Requires explicit `Send` API for parallel execution

## Workflow

```
        INPUT: topic
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
 [RSN 1] [RSN 2] [RSN 3]   ← PARALLEL (no dependencies)
    │        │        │
    └────────┼────────┘
             │
         [MERGE]
             │
         [RSN 4]              ← SEQUENTIAL (depends on merge)
             │
        OUTPUT: report
```

## Metrics

| Metric | A-PXM | LangGraph |
|--------|-------|-----------|
| Lines of code | ~12 | ~45 |
| Parallelism | Automatic | Manual Send API |
| Speedup (3 ops) | ~2.5x | ~1.9x |
| Efficiency | ~85% | ~64% |

## Run

```bash
python run.py --json
```
