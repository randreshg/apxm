# Reflection Benchmark

Demonstrates A-PXM's built-in REFL operation for self-analysis and improvement.

## What This Measures

- **A-PXM**: Native REFL operation for structured self-critique
- **LangGraph**: Custom prompting for reflection (no native support)

## Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                     REFLECTION FLOW                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  INPUT: task    │                                            │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ RSN: Initial attempt    │  ← First solution attempt          │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ REFL: Self-critique     │  ← Native reflection operation     │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ RSN: Improve answer     │  ← Apply reflection feedback       │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │  OUTPUT: improved       │                                    │
│  └─────────────────────────┘                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Differences

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Reflection support | Native REFL op | Custom prompting |
| Structured output | Built-in format | Manual parsing |
| Iteration control | Compiler-managed | Runtime loops |
| Cost optimization | Fused prompts | Separate calls |

## Run

```bash
python run.py --json
```
