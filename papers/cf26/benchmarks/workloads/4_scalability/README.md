# Scalability Benchmark

Measures parallelism efficiency at different levels (N = 2, 4, 8).

## What This Measures

- **A-PXM**: Automatic parallelism from dataflow, low overhead scheduler
- **LangGraph**: Manual parallelism with Send API, higher overhead

## Expected Results

```
Speedup
   ^
 8 |                              / Theoretical (y=x)
   |                            /
 6 |                          /   * A-PXM
   |                        /   *
 4 |                      /   *     o LangGraph
   |                    /   *     o
 2 |                  /   *     o
   |                /   *     o
 1 +-------------- *---o--------------> N
   1        2        4        8
```

## Efficiency Table

| N | Theoretical | A-PXM | LangGraph | A-PXM Eff | LG Eff |
|---|-------------|-------|-----------|-----------|--------|
| 1 | 1.00x | 1.00x | 1.00x | 100% | 100% |
| 2 | 2.00x | 1.70x | 1.40x | 85% | 70% |
| 4 | 4.00x | 3.00x | 2.20x | 75% | 55% |
| 8 | 8.00x | 4.80x | 3.20x | 60% | 40% |

## Run

```bash
python run.py --json
```
