# Multi-Agent Benchmark

Demonstrates A-PXM's native multi-agent composition and message passing.

## What This Measures

- **A-PXM**: Native agent spawning with automatic coordination
- **LangGraph**: Manual subgraph composition and state passing

## Workflow

```text
┌──────────────────────────────────────────────────────────────────┐
│                   MULTI-AGENT FLOW                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    COORDINATOR                               ││
│  │  ┌─────────────────┐                                        ││
│  │  │  INPUT: topic   │                                        ││
│  │  └────────┬────────┘                                        ││
│  │           │                                                  ││
│  │           ▼                                                  ││
│  │  ┌─────────────────────────┐                                ││
│  │  │ SPAWN: Researcher       │  ← Spawn child agent           ││
│  │  └────────┬────────────────┘                                ││
│  │           │                                                  ││
│  │           ▼                                                  ││
│  │  ┌─────────────────────────┐                                ││
│  │  │ SPAWN: Critic           │  ← Spawn child agent           ││
│  │  └────────┬────────────────┘                                ││
│  │           │                                                  ││
│  │           ▼                                                  ││
│  │  ┌─────────────────────────┐                                ││
│  │  │ RSN: Synthesize         │  ← Combine agent outputs       ││
│  │  └────────┬────────────────┘                                ││
│  │           │                                                  ││
│  │           ▼                                                  ││
│  │  ┌─────────────────────────┐                                ││
│  │  │  OUTPUT: final_report   │                                ││
│  │  └─────────────────────────┘                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌────────────────────┐    ┌────────────────────┐              │
│  │    RESEARCHER      │    │      CRITIC        │              │
│  │  ┌──────────────┐  │    │  ┌──────────────┐  │              │
│  │  │ RSN: Research│  │    │  │ RSN: Critique│  │              │
│  │  └──────────────┘  │    │  └──────────────┘  │              │
│  └────────────────────┘    └────────────────────┘              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Differences

| Aspect             | A-PXM                   | LangGraph              |
| ------------------ | ----------------------- | ---------------------- |
| Agent spawning     | Native spawn op         | Subgraph composition   |
| Message passing    | Built-in channels       | State dict sharing     |
| Parallel agents    | Automatic detection     | Manual orchestration   |
| Agent lifecycle    | Compiler-managed        | Runtime management     |

## Run

```bash
python run.py --json
```
