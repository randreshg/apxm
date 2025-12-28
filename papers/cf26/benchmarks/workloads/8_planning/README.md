# Planning Benchmark

Demonstrates A-PXM's native PLAN operation for multi-step task decomposition.

## What This Measures

- **A-PXM**: Native PLAN operation with automatic step execution
- **LangGraph**: Chain-of-thought prompting for planning

## Workflow

```text
┌──────────────────────────────────────────────────────────────────┐
│                      PLANNING FLOW                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  INPUT: goal    │                                            │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ PLAN: Decompose goal    │  ← Native planning operation       │
│  │       into steps        │                                    │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ RSN: Execute step 1     │                                    │
│  │ RSN: Execute step 2     │  ← Parallel where possible         │
│  │ RSN: Execute step 3     │                                    │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ RSN: Synthesize results │  ← Combine step outputs            │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │  OUTPUT: final_result   │                                    │
│  └─────────────────────────┘                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Differences

| Aspect      | A-PXM                    | LangGraph              |
| ----------- | ------------------------ | ---------------------- |
| Planning    | Native PLAN op           | Custom CoT prompting   |
| Step exec   | Automatic parallelism    | Manual orchestration   |
| Step format | Structured output        | Text parsing required  |
| Replanning  | Built-in support         | Custom implementation  |

## Run

```bash
python run.py --json
```
