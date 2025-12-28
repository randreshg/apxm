# Type Verification Benchmark

Compares compile-time vs runtime error detection.

## What This Measures

- **A-PXM**: Catches undefined variable at compile time (before any LLM calls)
- **LangGraph**: Discovers error at runtime (after LLM call, wasting cost)

## The Error

Both workflows contain the SAME bug: using an undefined variable.

```
A-PXM (Compile-Time):                    LangGraph (Runtime):
──────────────────────                   ─────────────────────

$ apxm compile workflow.ais              $ python workflow.py

error[E0425]: undefined variable         Traceback (most recent call last):
 --> workflow.ais:5:20                     File "workflow.py", line 47
  |                                        in invoke
5 |     rsn undefined_var -> output        KeyError: 'missing_key'
  |         ^^^^^^^^^^^^
  |         not defined                  # After LLM call already made!
                                         # Tokens wasted, cost incurred
COST: $0.00 (caught before LLM)
TIME: 50ms (compile only)                COST: $0.15 (failed after LLM)
                                         TIME: 3.2s (LLM + failure)
```

## Metrics

| Metric | A-PXM | LangGraph |
|--------|-------|-----------|
| Time to error | 50ms | 3.2s |
| Cost of error | $0.00 | $0.15 |
| Error quality | Precise location | Stack trace |
| LLM calls wasted | 0 | 1+ |

## Run

```bash
python run.py --json
```
