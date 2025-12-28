# Tool Invocation Benchmark

Demonstrates A-PXM's native INV operations and capability system.

## What This Measures

- **A-PXM**: Native capability invocation with compile-time validation
- **LangGraph**: Runtime tool binding with LangChain tools

## Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                     TOOL INVOCATION FLOW                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  INPUT: query   │                                            │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ RSN: Decide which tool  │  ← LLM reasons about tool choice   │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ INV: Execute tool       │  ← Invoke registered capability    │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │ RSN: Synthesize answer  │  ← LLM reasons about results       │
│  └────────┬────────────────┘                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────┐                                    │
│  │  OUTPUT: final_answer   │                                    │
│  └─────────────────────────┘                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Differences

| Aspect | A-PXM | LangGraph |
|--------|-------|-----------|
| Tool registration | Compile-time capability | Runtime binding |
| Validation | Static type checking | Runtime errors |
| Invocation overhead | Microseconds | Milliseconds |
| Tool discovery | Capability registry | Dynamic lookup |

## Run

```bash
python run.py --json
```
