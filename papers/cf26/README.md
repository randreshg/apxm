# A-PXM Evaluation for CF'26

## What is A-PXM?

A-PXM (Agent Program Execution Model) is a **specification** for agent execution, not a competing framework. It defines the formal semantics that any agent system could adopt.

| Component | What It Specifies |
|-----------|-------------------|
| **AAM** | Agent Abstract Machine - explicit state (Beliefs, Goals, Capabilities) + 3-tier memory |
| **AIS** | Agent Instruction Set - 19 typed operations with formal semantics |
| **Dataflow Semantics** | Token-based scheduling - operations fire when input data arrives |

The Rust compiler and runtime in this repository are a **reference implementation** demonstrating the specification works. Any framework (LangGraph, CrewAI, AutoGen) could adopt A-PXM principles.

---

## Unique Contributions (vs Prior Work)

| Contribution | Why It's Novel |
|--------------|----------------|
| **Formal Execution Model** | First PXM for agentic AI (analogous to LLVM IR for compilers) |
| **Typed IR (AIS)** | Enables static verification + compiler optimizations |
| **Explicit 3-Tier Memory** | STM/LTM/Episodic with typed operations (not implicit state) |
| **Dataflow Semantics** | Automatic parallelism from program structure |
| **Full Agent Programs** | Beyond function calling - complete multi-step workflows |

---

## Why Compare to LangGraph?

LangGraph is the most widely-used orchestration framework for agent workflows:
- Production-ready, well-documented
- Represents the "state of practice" for agent development
- Provides a fair baseline for workflow-level comparison

**We are NOT claiming**: "Rust is faster than Python" or "Our runtime beats LangGraph"

**We ARE demonstrating**: What becomes possible when workflows follow A-PXM's formal execution model:
- Typed operations catch errors at compile time
- Explicit state enables inspection and debugging
- Dataflow semantics expose parallelism automatically

---

## How vs LLMCompiler?

LLMCompiler (ICML 2024) introduced DAG-based parallel function calling. A-PXM is complementary:

| Aspect | LLMCompiler | A-PXM |
|--------|-------------|-------|
| **Scope** | Function calls | Full agent programs |
| **State model** | Implicit | Explicit AAM (B,G,C) |
| **IR** | Untyped DAG | Typed AIS (19 operations) |
| **Memory** | None | 3-tier (STM/LTM/Episodic) |
| **Verification** | None | Static type checking |

LLMCompiler could target AIS as an optimization backend - they solve different problems.

---

## Benchmark Workloads

Each workload demonstrates a specific A-PXM property:

| # | Name | A-PXM Property Demonstrated |
|---|------|----------------------------|
| 1 | Parallel Research | Dataflow semantics → automatic parallelism |
| 2 | Chain Fusion | Typed IR → FuseReasoning optimization |
| 3 | Type Verification | Typed operations → compile-time errors |
| 4 | Scalability | Dataflow → Work-Span model validation |
| 5 | Memory Augmented | AAM → 3-tier memory (STM/LTM/Episodic) |
| 6 | Tool Invocation | AIS INV → typed tool calls |
| 7 | Reflection | AIS REFL → native reflection operation |
| 8 | Planning | AIS PLAN → native planning operation |
| 9 | Conditional Routing | Dataflow → parallel preparation |
| 10 | Multi-Agent | AAM → multi-agent coordination |
| 11 | Compilation Scaling | Compiler → linear scaling |
| 12 | Real LLM Probe | Runtime → production feasibility |
| 13 | Fusion Quality | FuseReasoning → task-type analysis |
| 14 | Token Estimation | Typed IR → cost estimation |

See `benchmarks/workloads/` for individual workload documentation.

---

## Running Benchmarks

### Prerequisites

```bash
# Start Ollama (local LLM backend)
ollama serve
ollama pull gpt-oss:20b-cloud

# Install Python dependencies
pip install typer rich langgraph langchain-ollama

# Add apxm to PATH (add to ~/.zshrc or ~/.bashrc for persistence)
export PATH="$PATH:$(pwd)/bin"

# Build A-PXM compiler
apxm compiler build
```

### Run Individual Workload

```bash
# List available workloads
apxm workloads list

# Run a specific workload
apxm workloads run 1_parallel_research

# With JSON output
apxm workloads run 1_parallel_research --json
```

### Run All Workloads

```bash
cd papers/CF26/benchmarks/workloads
python run_all.py --paper
```

---

## The Core Insight

> *"A-PXM is a specification; it defines the execution substrate's semantics, not a particular implementation."*

The evaluation demonstrates that the **specification works** - parallelism emerges from dataflow, types catch errors early, and the overhead is negligible. Any framework adopting these principles would see similar benefits.
