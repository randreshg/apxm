# APXM Examples

Learning progression for APXM agent workflows.

## Quick Start Examples

Follow these examples in order to learn APXM from basics to advanced patterns:

### 1. Basics
- **01_hello.ais** - Minimal agent with single ASK operation
- **01_hello_graph.json** - Same agent in JSON graph format
- **02_tool_use.ais** - Tool/capability invocation patterns
- **02_tool_use_graph.json** - Tool usage in graph format

### 2. Control Flow
- **03_multi_flow.ais** - Cross-agent control flow and coordination
- **03_multi_flow_graph.json** - Multi-flow in graph format

### 3. Multi-Agent Patterns
- **04_multi_agent_communicate.ais** - Agent-to-agent communication
- **05_apxm_council.ais** - Council pattern with multiple agents
- **06_code_review_council.ais** - Code review workflow with councils

### 4. Python Integration
- **python/demo_apxm_agents.py** - Python API usage examples

### 5. Flagship Demo
- **flagship/** - Production-ready 3-agent collaborative research pipeline
  - See `flagship/README.md` for details

## Running Examples

Compile and run an example:

```bash
# Compile AIS source to binary
apxm compile examples/01_hello.ais -o hello.apxm

# Execute the compiled binary
apxm execute hello.apxm

# Or compile from JSON graph
apxm compile examples/01_hello_graph.json -o hello.apxm
```

See `docs/getting-started.md` for detailed tutorials.
