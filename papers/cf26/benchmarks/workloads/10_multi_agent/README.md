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
|  |  | RSN: Synthesize report  |                               | |
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
        rsn("Conduct deep research on: " + topic) -> findings
        return findings
    }
}

agent Critic {
    flow critique(content: str) -> str {
        rsn("Critically analyze and identify weaknesses: " + content) -> feedback
        return feedback
    }

    flow prepare(topic: str) -> str {
        rsn("Prepare initial critique questions for: " + topic) -> questions
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
        rsn("Synthesize into final report. Research: " + research_result +)            " | Prepared questions: " + critique_prep +
            " | Critique: " + critique_result -> final_report
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

### Prerequisites

```bash
# Start Ollama (local LLM backend)
ollama serve
ollama pull gpt-oss:20b-cloud

# Install Python dependencies
pip install langgraph langchain-ollama

# Build A-PXM compiler (from repo root)
apxm compiler build
```

### Run A-PXM Version

```bash
cd papers/CF26/benchmarks/workloads/10_multi_agent

# Compile and run
apxm compiler run workflow.ais -O1
```

### Run LangGraph Comparison

```bash
cd papers/CF26/benchmarks/workloads/10_multi_agent
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 10_multi_agent

# With JSON output
apxm workloads run 10_multi_agent --json
```

---

## Results

*To be filled after benchmark execution*

| Aspect | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| Agent spawning | Native flow calls | Subgraph composition | |
| Message passing | Direct parameters | State dict sharing | |
| Parallel agents | Automatic detection | Manual orchestration | |
| Agent lifecycle | Compiler-managed | Runtime management | |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Automatic parallelism**: `Researcher.research()` and `Critic.prepare()` run concurrently (no dependencies).

2. **Cross-agent dataflow**: Dependencies flow naturally through return values.

3. **Hierarchical composition**: Coordinator orchestrates Researcher and Critic without manual coordination code.

### Key Insight

This workload demonstrates that A-PXM treats agents as first-class entities with explicit state and flows. Multi-agent coordination emerges from dataflow dependencies, not manual orchestration. The compiler analyzes cross-agent calls and parallelizes independent invocations automatically.
