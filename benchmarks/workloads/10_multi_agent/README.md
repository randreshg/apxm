# Workload 10: Multi-Agent

## Purpose

Demonstrate A-PXM's native multi-agent composition with automatic coordination. Agents are first-class entities with explicit flows, not nested function graphs.

## What We're Demonstrating

**A-PXM Property**: AAM multi-agent coordination

A-PXM specifies agents as first-class entities with explicit state (Beliefs, Goals, Capabilities). Cross-agent calls use direct flow invocation, and the compiler automatically detects parallelism opportunities between independent agent invocations.

```
                   MULTI-AGENT FLOW
+----------------------------------------------------------------+
|                                                                |
|  +-----------------------------------------------------------+ |
|  |                    COORDINATOR                             | |
|  |  +-----------------+                                       | |
|  |  |  INPUT: topic   |                                       | |
|  |  +--------+--------+                                       | |
|  |           |                                                | |
|  |           v                                                | |
|  |  +-------------------------+  +----------------------+     | |
|  |  | Researcher.research()   |  | Critic.prepare()     |     | |
|  |  | (parallel with Critic)  |  | (parallel)           |     | |
|  |  +--------+----------------+  +----------+-----------+     | |
|  |           |                              |                 | |
|  |           v                              |                 | |
|  |  +-------------------------+             |                 | |
|  |  | Critic.critique()       | <-----------+                 | |
|  |  | (depends on research)   |                               | |
|  |  +--------+----------------+                               | |
|  |           |                                                | |
|  |           v                                                | |
|  |  +-------------------------+                               | |
|  |  | ASK: Synthesize report  |                               | |
|  |  +--------+----------------+                               | |
|  |           |                                                | |
|  |           v                                                | |
|  |  +-------------------------+                               | |
|  |  |  OUTPUT: final_report   |                               | |
|  |  +-------------------------+                               | |
|  +-----------------------------------------------------------+ |
|                                                                |
|  +--------------------+    +--------------------+              |
|  |    RESEARCHER      |    |      CRITIC        |              |
|  |  +-------------+   |    |  +-------------+   |              |
|  |  | flow:       |   |    |  | flow:       |   |              |
|  |  | research()  |   |    |  | critique()  |   |              |
|  |  +-------------+   |    |  | prepare()   |   |              |
|  +--------------------+    +--------------------+              |
|                                                                |
+----------------------------------------------------------------+
```

### A-PXM Code (workflow.ais)

```
agent Researcher {
    flow research(topic: str) -> str {
        ask("Conduct deep research on: " + topic) -> findings
        return findings
    }
}

agent Critic {
    flow critique(content: str) -> str {
        ask("Critically analyze and identify weaknesses: " + content) -> feedback
        return feedback
    }

    flow prepare(topic: str) -> str {
        ask("Prepare initial critique questions for: " + topic) -> questions
        return questions
    }
}

agent Coordinator {
    @entry flow main(topic: str) -> str {
        // 1. These run in parallel (no dependencies between them)
        Researcher.research(topic) -> research_result
        Critic.prepare(topic) -> critique_prep

        // 2. This depends on research_result, runs after
        Critic.critique(research_result) -> critique_result

        // 3. Synthesize final report
        ask("Synthesize into final report. Research: " + research_result +
            " | Prepared questions: " + critique_prep +
            " | Critique: " + critique_result) -> final_report
        return final_report
    }
}
```

### LangGraph Comparison

LangGraph uses subgraph composition:
- Agents are nested StateGraph instances
- State passing through shared dict
- Manual orchestration of agent coordination

---

## How to Run

### Quick Run (Compile + Execute)

```bash
cd papers/cf26/benchmarks/workloads/10_multi_agent

# Compile and execute
apxm execute --emit-metrics metrics.json workflow.ais "Climate-change-impacts"
```

### Compile Only

```bash
# Compile with diagnostics
apxm compile workflow.ais -o workflow.apxmobj --emit-diagnostics diagnostics.json -O1
```

### Run Pre-compiled Artifact

```bash
# Run with metrics export
apxm run --emit-metrics metrics.json workflow.apxmobj "topic"
```

### Run Full Benchmark

```bash
# From repo root
apxm workloads run 10_multi_agent

# With JSON output
apxm workloads run 10_multi_agent --json
```

---

## Results

### Measured Values

| Metric | Value | Notes |
|--------|-------|-------|
| Total Duration | 7,507ms | 4 LLM calls across 3 agents |
| Nodes Executed | 11 | Flow calls + asks + print + return |
| LLM Calls | 4 | research + prepare + critique + synthesize |
| Max Parallelism | 5 | High concurrent execution |
| Avg Parallelism | 3.5 | Sustained parallel execution |
| Total Tokens | 1,368 | 402 input + 966 output |
| Avg LLM Latency | ~2.6s | Per-call average |
| Optimization Level | O1 | Standard optimization |

### A-PXM vs LangGraph Comparison

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Agent spawning | **Native flow calls** | Subgraph composition | Direct invocation |
| Message passing | **Direct parameters** | State dict sharing | Type-safe |
| Parallel agents | **Automatic detection** | Manual orchestration | Compiler-analyzed |
| Agent lifecycle | **Compiler-managed** | Runtime management | Static analysis |
| Max parallelism | **5 concurrent** | Manual async | Dataflow scheduling |

---

## Analysis

### Observations

1. **High parallelism achieved**: Max parallelism of 5 and avg parallelism of 3.5 demonstrates automatic parallelization of independent cross-agent calls.

2. **Cross-agent flow calls work**: `Researcher.research()`, `Critic.prepare()`, and `Critic.critique()` are all invoked correctly as cross-agent flow calls.

3. **Dependency ordering respected**: `Critic.critique(research_result)` correctly waits for `Researcher.research()` to complete before executing.

4. **Multi-agent composition**: Three agents (Coordinator, Researcher, Critic) compose naturally with the coordinator orchestrating the others.

### Key Insight

This workload demonstrates that A-PXM treats agents as first-class entities with explicit state and flows. Multi-agent coordination emerges from dataflow dependencies, not manual orchestration:

- **Automatic parallelism**: Independent agent calls (`research` and `prepare`) run concurrently
- **Type-safe invocation**: Cross-agent calls use typed flow signatures
- **Dependency tracking**: Compiler analyzes dataflow to determine execution order
- **Hierarchical composition**: Coordinator orchestrates sub-agents without manual coordination code
- **AAM integration**: Each agent maintains its own beliefs, goals, and capabilities
