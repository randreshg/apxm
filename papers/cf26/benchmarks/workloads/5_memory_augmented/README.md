# Workload 5: Memory Augmented

## Purpose

Demonstrate A-PXM's 3-tier memory system (STM/LTM/Episodic) with typed operations. Memory is explicit and inspectable, not hidden in closures.

## What We're Demonstrating

**A-PXM Property**: AAM (Agent Abstract Machine) explicit 3-tier memory

A-PXM specifies three memory tiers with distinct semantics:
- **STM (Short-Term Memory)**: Fast, in-memory working state for current task
- **LTM (Long-Term Memory)**: Persistent knowledge that survives across runs
- **Episodic**: Append-only audit trail for debugging and compliance

```
+----------------------------------------------------------------+
|                     A-PXM MEMORY SYSTEM                        |
+----------------------------------------------------------------+
|                                                                |
|  +-------------+  +-------------+  +---------------------+     |
|  | STM (fast)  |  | LTM (persist)|  | Episodic (audit)   |     |
|  |             |  |             |  |                     |     |
|  | Working     |  | Knowledge   |  | Execution traces    |     |
|  | memory for  |  | that persists|  | for debugging and   |     |
|  | current task|  | across runs |  | auditability        |     |
|  |             |  |             |  |                     |     |
|  | Access: ~us |  | Access: ~ms |  | Access: ~us (write) |     |
|  +-------------+  +-------------+  +---------------------+     |
|                                                                |
+----------------------------------------------------------------+
```

### A-PXM Code (workflow.ais)

```
agent MemoryAgent {
    @entry flow main(query: str) -> str {
        // 1. Query long-term memory for cached knowledge
        qmem("domain_knowledge", "ltm") -> cached

        // 2. Update short-term memory with working state
        umem(query, "stm")

        // 3. Reason with context from memory
        rsn("Use the retrieved memory to answer the query.", cached, query) -> answer

        // 4. Update episodic memory for audit trail
        umem(answer, "episodic")

        // 5. Persist new knowledge to long-term memory
        umem(answer, "ltm")

        return answer
    }
}
```

### LangGraph Comparison

LangGraph uses Python dicts and checkpointing:
- State is implicit in closures
- Persistence requires external setup
- No built-in audit trail

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
cd papers/CF26/benchmarks/workloads/5_memory_augmented

# Compile and run
apxm compiler run workflow.ais -O1
```

### Run LangGraph Comparison

```bash
cd papers/CF26/benchmarks/workloads/5_memory_augmented
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 5_memory_augmented

# With JSON output
apxm workloads run 5_memory_augmented --json
```

---

## Results

*To be filled after benchmark execution*

| Metric | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| STM access | ~us | N/A | In-memory hash map |
| LTM access | ~ms | ~ms | Both persist to SQLite |
| Episodic write | ~us | Custom | Built-in vs manual |
| Audit trail | Built-in | Manual | AAM property |
| State inspection | Always | Debugger | Explicit vs implicit |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Memory operations have negligible overhead**: Sub-microsecond for STM, milliseconds for LTM persistence.

2. **Explicit state is inspectable**: AAM state (Beliefs, Goals, Capabilities, Memory) can be dumped at any time.

3. **Audit trail is automatic**: Episodic memory records all state transitions.

### Key Insight

This workload demonstrates that A-PXM's memory model is **explicit by design**. State is not hidden in Python closures - it's part of the formal AAM specification. This enables debugging, auditing, and formal reasoning about agent behavior.
