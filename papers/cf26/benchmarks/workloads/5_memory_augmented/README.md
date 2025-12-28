# Memory Augmented Benchmark

Demonstrates A-PXM's 3-tier memory system (STM/LTM/Episodic).

## What This Measures

- **A-PXM**: Native 3-tier memory with microsecond access times
- **LangGraph**: Checkpoint-based persistence with millisecond overhead

## Memory Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│                     A-PXM MEMORY SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ STM (fast)   │  │ LTM (persist)│  │ Episodic (audit)     │  │
│  │              │  │              │  │                      │  │
│  │ Working      │  │ Knowledge    │  │ Execution traces     │  │
│  │ memory for   │  │ that persists│  │ for debugging and    │  │
│  │ current task │  │ across runs  │  │ auditability         │  │
│  │              │  │              │  │                      │  │
│  │ Access: ~μs  │  │ Access: ~ms  │  │ Access: ~μs (write)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Workflow

1. Recall cached knowledge from LTM
2. Store working state in STM
3. Reason with context
4. Record execution to Episodic for audit trail
5. Persist new knowledge to LTM

## Metrics

| Metric | A-PXM | LangGraph |
|--------|-------|-----------|
| STM access | ~μs | N/A |
| LTM access | ~ms | ~ms (checkpoint) |
| Episodic write | ~μs | Custom logging |
| Audit trail | Built-in | Manual |

## Run

```bash
python run.py --json
```
