---
title: "Memory Operations"
description: "AIS instructions for reading, writing, and ordering access to the agent's three-tier memory hierarchy."
---

# Memory Operations

Memory operations read from and write to the AAM's three-tier memory hierarchy (STM, LTM, Episodic). They are the interface between the agent's computational operations and its persistent state.

## QMEM -- QueryMemory

Retrieves a value from the memory hierarchy. The runtime searches tiers in order (STM -> LTM -> Episodic) and returns the first match.

**Signature:**
```
QMEM(q: String, sid: SessionID, k: Int) -> Value
```

| Operand | Type | Description |
|---------|------|-------------|
| `q` | `String` | Query key or search expression |
| `sid` | `SessionID` | Session scope for the lookup |
| `k` | `Int` | Maximum number of results to return |

**Behavior:**

1. Search STM for key `q` within session `sid`. If found, return immediately.
2. If not in STM, search LTM. If found, promote a copy to STM for locality.
3. If not in LTM, search Episodic memory for matching trace entries.
4. Return up to `k` results, ordered by relevance.

**Example:**
```mlir
%user_history = "ais.qmem"(%query, %session, %k) {
  tier_hint = "ltm"
} : (!ais.string, !ais.session_id, i64) -> !ais.value
```

The optional `tier_hint` attribute lets the compiler bypass unnecessary tier searches when the target tier is known statically.

## UMEM -- UpdateMemory

Persists data to memory and updates the AAM's Beliefs accordingly.

**Signature:**
```
UMEM(data: Value, sid: SessionID) -> Void
```

| Operand | Type | Description |
|---------|------|-------------|
| `data` | `Value` | The typed value to store |
| `sid` | `SessionID` | Session scope for the write |

**Behavior:**

1. Write `data` to STM for immediate availability.
2. If the value is marked `durable`, also write to LTM within a transaction.
3. Append a write record to Episodic memory for auditability.
4. Update the AAM Beliefs map: `B' = B[key(data) := data]`.

**Example:**
```mlir
"ais.umem"(%analysis_result, %session) {
  durable = true,
  belief_key = "latest_analysis"
} : (!ais.value, !ais.session_id) -> ()
```

## FENCE -- Memory Barrier

Ensures all prior memory writes are visible before any subsequent memory reads execute. FENCE is a synchronization point in the dataflow graph -- it has no data output, but it creates an ordering edge.

**Signature:**
```
FENCE() -> Void
```

**Behavior:**

1. Block until all preceding UMEM operations in the current subgraph have committed.
2. Flush STM write buffers.
3. Emit a token to signal that subsequent QMEM operations may proceed.

**Example:**
```mlir
// Write results
"ais.umem"(%result_a, %session) : (!ais.value, !ais.session_id) -> ()
"ais.umem"(%result_b, %session) : (!ais.value, !ais.session_id) -> ()

// Ensure writes are visible
"ais.fence"() : () -> ()

// Safe to read updated state
%merged = "ais.qmem"(%merge_query, %session, %k) : (!ais.string, !ais.session_id, i64) -> !ais.value
```

## Ordering Guarantees

| Scenario | Guarantee |
|----------|-----------|
| QMEM after UMEM (same key, same subgraph) | Read sees write (data dependency edge) |
| QMEM after UMEM (different keys) | No guarantee without FENCE |
| UMEM after UMEM (same key) | Last-writer-wins within subgraph ordering |
| Cross-agent memory access | Requires COMM protocol; no shared memory |

## Memory Operation Patterns

### Read-Modify-Write

```mlir
%old = "ais.qmem"(%key, %sid, %one) : (...) -> !ais.value
%new = "ais.ask"(%modify_prompt, %old) : (...) -> !ais.future<!ais.string>
"ais.umem"(%new, %sid) { belief_key = "counter" } : (...) -> ()
```

### Bulk Retrieval

```mlir
%results = "ais.qmem"(%broad_query, %sid, %ten) {
  tier_hint = "ltm",
  similarity_threshold = 0.8
} : (!ais.string, !ais.session_id, i64) -> !ais.value
```

### Write-Through Caching

When `durable = true` is set on UMEM, the write goes to both STM and LTM atomically. This ensures that a subsequent QMEM from any tier will see the updated value, at the cost of higher write latency (~ms for SQLite commit).
