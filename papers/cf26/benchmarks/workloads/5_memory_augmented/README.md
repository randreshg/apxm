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
        ask("Use the retrieved memory to answer the query.", cached, query) -> answer

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

### Quick Run (Compile + Execute)

```bash
cd papers/cf26/benchmarks/workloads/5_memory_augmented

# Compile and execute
apxm execute workflow.ais "What is quantum computing?"
```

### Compile Only

```bash
# Compile with diagnostics
apxm compile workflow.ais -o workflow.apxmobj --emit-diagnostics diagnostics.json -O1
```

### Run Pre-compiled Artifact

```bash
# Run with metrics export
apxm run --emit-metrics metrics.json workflow.apxmobj "What is quantum computing?"
```

### Run LangGraph Comparison

```bash
python workflow.py
```

### Run Full Benchmark

```bash
# From repo root
apxm workloads run 5_memory_augmented

# With JSON output
apxm workloads run 5_memory_augmented --json
```

---

## Results

### Measured Values

| Metric | Value | Notes |
|--------|-------|-------|
| Total Duration | 1,606ms | Includes memory + LLM |
| Nodes Executed | 8 | qmem, umem×3, ask, print×2 |
| LLM Calls | 1 | Single reasoning operation |
| LLM Latency | 1,592ms | Dominates total time |
| Scheduler Overhead | ~4µs | Memory ops included |

### Memory Operation Performance

| Operation | Space | Latency | Notes |
|-----------|-------|---------|-------|
| qmem (query) | LTM | <1ms | Vector similarity search |
| umem (update) | STM | <1µs | In-memory hash map |
| umem (update) | Episodic | <1ms | Append-only log |
| umem (update) | LTM | <1ms | SQLite write |

### A-PXM vs LangGraph

| Metric | A-PXM | LangGraph | Notes |
|--------|-------|-----------|-------|
| STM access | <1µs | N/A | Built-in vs external |
| LTM access | <1ms | ~ms | Both use persistence |
| Episodic write | <1ms | Custom | Built-in audit trail |
| Audit trail | **Built-in** | Manual | AAM property |
| State inspection | **Always** | Debugger | Explicit vs implicit |

---

## Analysis

### Observations

1. **Memory overhead negligible**: Combined memory operations (qmem + 3× umem) add <5ms to total execution. LLM latency (1,592ms) dominates.

2. **Three-tier memory works end-to-end**:
   - STM: Fast working memory for current query
   - LTM: Persistent knowledge retrieval
   - Episodic: Automatic audit trail for compliance

3. **Explicit state enables inspection**: AAM state (Beliefs, Goals, Capabilities, Memory) can be dumped at any time for debugging.


### Key Insight

This workload demonstrates that A-PXM's memory model is **explicit by design**. State is not hidden in Python closures - it's part of the formal AAM specification. This enables:

- **Debugging**: Inspect memory state at any point
- **Auditing**: Built-in episodic trace for compliance
- **Formal reasoning**: Memory semantics are well-defined
