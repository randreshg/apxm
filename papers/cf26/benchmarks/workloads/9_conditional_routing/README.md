# Conditional Routing Benchmark

Demonstrates A-PXM's dataflow-based routing from LLM output.

## What This Measures

- **A-PXM**: Native switch/case with compile-time route validation
- **LangGraph**: Runtime conditional edges with dynamic routing

## Workflow

```text
┌──────────────────────────────────────────────────────────────────┐
│                   CONDITIONAL ROUTING FLOW                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  INPUT: query   │                                            │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ RSN: Classify input     │  ← LLM determines category         │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │      SWITCH/CASE        │  ← Route based on classification   │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│     ┌─────┼─────┬─────┐                                         │
│     │     │     │     │                                         │
│     ▼     ▼     ▼     ▼                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────────┐                           │
│  │TECH │ │CREA │ │FACT │ │ DEFAULT │                           │
│  └──┬──┘ └──┬──┘ └──┬──┘ └────┬────┘                           │
│     │       │       │         │                                 │
│     └───────┴───────┴─────────┘                                 │
│                 │                                                │
│                 ▼                                                │
│  ┌─────────────────────────┐                                    │
│  │  OUTPUT: response       │                                    │
│  └─────────────────────────┘                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Differences

| Aspect          | A-PXM                  | LangGraph               |
| --------------- | ---------------------- | ----------------------- |
| Route definition | Compile-time switch   | Runtime conditional     |
| Route validation | Static checking       | Runtime errors          |
| Branch handling  | Native control flow   | Conditional edges       |
| Default case     | Compiler-enforced     | Optional                |

## Run

```bash
python run.py --json
```
