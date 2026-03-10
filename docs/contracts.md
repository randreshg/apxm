# MLIR-to-Runtime and Artifact Wire Format Contracts

This document specifies the synchronization contracts between the APXM compiler
(C++ MLIR backend) and the Rust runtime.  Any change to operation indices,
wire encoding, or shared attribute keys **must** update every sync point listed
below.

---

## 1. Op Kind Contract Chain

Three locations must agree on the integer index assigned to each operation:

| Layer | File | Mechanism |
|-------|------|-----------|
| **C++ Emitter** | `crates/apxm-compiler/mlir/lib/Dialect/AIS/Conversion/Artifact/ArtifactEmitter.cpp` | `enum class OperationKind : uint32_t` |
| **Rust Source of Truth** | `crates/apxm-ais/src/operations/definitions.rs` | `AISOperationType::from_wire_index()` (u32 lookup table) |
| **MLIR TableGen** | `crates/apxm-compiler/mlir/include/ais/Dialect/AIS/IR/AISOps.td` | Op class names (e.g. `AIS_AskOp`) |

The C++ emitter writes the `OperationKind` index as a `u32` into the binary
wire format.  The Rust parser in `crates/apxm-compiler/src/codegen/artifact.rs`
calls `AISOperationType::from_wire_index(op_index)` to recover the enum
variant.  If the two tables disagree, operations are misidentified at runtime.

### 1.1 Index Table (Version 3)

| Index | C++ `OperationKind` | Rust `AISOperationType` | MLIR Mnemonic |
|------:|---------------------|-------------------------|---------------|
| 0 | `Inv` | `Inv` | `ais.inv` |
| 1 | `Ask` | `Ask` | `ais.ask` |
| 2 | `QMem` | `QMem` | `ais.qmem` |
| 3 | `UMem` | `UMem` | `ais.umem` |
| 4 | `Plan` | `Plan` | `ais.plan` |
| 5 | `WaitAll` | `WaitAll` | `ais.wait_all` |
| 6 | `Merge` | `Merge` | `ais.merge` |
| 7 | `Fence` | `Fence` | `ais.fence` |
| 8 | `Exc` | `Exc` | `ais.exc` |
| 9 | `Communicate` | `Communicate` | `ais.communicate` |
| 10 | `Reflect` | `Reflect` | `ais.reflect` |
| 11 | `Verify` | `Verify` | `ais.verify` |
| 12 | `Err` | `Err` | `ais.err` |
| 13 | `ReturnOp` | `Return` | `ais.return` |
| 14 | `Jump` | `Jump` | `ais.jump` |
| 15 | `BranchOnValue` | `BranchOnValue` | `ais.branch_on_value` |
| 16 | `LoopStart` | `LoopStart` | `ais.loop_start` |
| 17 | `LoopEnd` | `LoopEnd` | `ais.loop_end` |
| 18 | `TryCatch` | `TryCatch` | `ais.try_catch` |
| 19 | `ConstStr` | `ConstStr` | `ais.const_str` |
| 20 | `Switch` | `Switch` | `ais.switch` |
| 21 | `FlowCall` | `FlowCall` | `ais.flow_call` |
| 22 | `Print` | `Print` | `ais.print` |
| 23 | `Think` | `Think` | `ais.think` |
| 24 | `Reason` | `Reason` | `ais.reason` |

### 1.2 Phase 1 Extensions (Rust-Only)

The following operations exist in the Rust `AISOperationType` enum, have
runtime handlers in the dispatcher, but are **not yet wired** into the C++
`OperationKind` enum (no wire index assigned).  They can only be created
programmatically in Rust, not compiled from MLIR source:

| Rust `AISOperationType` | Handler module |
|-------------------------|----------------|
| `UpdateGoal` | `handlers/update_goal.rs` |
| `Guard` | `handlers/guard.rs` |
| `Claim` | `handlers/claim.rs` |
| `Pause` | `handlers/pause.rs` |
| `Resume` | `handlers/resume.rs` |

When these operations are added to the compiler, they must be assigned the next
available indices (starting at 25) in both `OperationKind` and
`from_wire_index()`.

---

## 2. Artifact Wire Format

The artifact format has two layers: an outer container and an inner compiler
wire format.

### 2.1 Outer Container (`.apxm` / `.apxmobj`)

Defined in `crates/runtime/apxm-artifact/src/lib.rs`.

The outer container has a **52-byte fixed header** followed by a
bincode-serialized payload:

```
Offset  Size  Field            Description
──────  ────  ─────            ───────────
 0       4    magic            ASCII "APXM" (bytes 0x41 0x50 0x58 0x4D)
 4       4    version          u32 LE, currently 1
 8       8    payload_length   u64 LE, byte length of the payload section
16      32    blake3_hash      BLAKE3 digest of the payload bytes
48       4    flags            u32 LE, reserved (currently 0)
52       *    payload          bincode-serialized ArtifactPayload
```

**Total header size:** 4 + 4 + 8 + 32 + 4 = **52 bytes**.

The payload is a bincode-serialized `ArtifactPayload` struct containing:

- `metadata` -- `ArtifactMetadata` (module name, creation timestamp, compiler
  version)
- `dags` -- `Vec<WireDag>` (one per flow, serde-serialized `ExecutionDag`
  representations)
- `sections` -- `Vec<ArtifactSection>` (extensible data sections)

Integrity is verified by computing BLAKE3 over the raw payload bytes and
comparing against the stored hash.  A mismatch produces
`ArtifactError::HashMismatch`.

### 2.2 Inner Compiler Wire Format (Version 3)

Defined in `ArtifactEmitter.cpp` (C++ writer) and
`crates/apxm-compiler/src/codegen/artifact.rs` (Rust reader via
`parse_wire_dags()`).

This is a hand-rolled little-endian binary format produced by the C++
`ArtifactSerializer` class.  All multi-byte integers are little-endian.

#### Top-Level Structure

```
u32     version         -- must be 3
u64     num_dags        -- number of DAGs in this artifact
DAG[num_dags]           -- DAG records
```

#### DAG Record

```
String  module_name     -- flow/function name
bool    is_entry        -- true if flow has @entry attribute
u64     param_count     -- number of entry flow parameters
  For each parameter:
    String  name        -- parameter name
    String  type_name   -- parameter type (e.g. "any")
u64     node_count      -- number of nodes
  Node[node_count]
u64     edge_count      -- number of edges
  Edge[edge_count]
u64     entry_count     -- number of entry node IDs
  u64[entry_count]      -- entry node IDs
u64     exit_count      -- number of exit node IDs
  u64[exit_count]       -- exit node IDs
```

#### Node Record

```
u64     id              -- unique node identifier
u32     op_type         -- OperationKind index (see section 1.1)
u64     attr_count      -- number of key-value attributes
  For each attribute:
    String  key         -- attribute name
    Value   value       -- typed value (see Value encoding below)
u64     input_count     -- number of input token IDs
  u64[input_count]      -- input token IDs
u64     output_count    -- number of output token IDs
  u64[output_count]     -- output token IDs
u32     priority        -- scheduling priority
bool    has_latency     -- whether estimated_latency is present
u64?    estimated_latency -- only present when has_latency is true
```

#### Edge Record

```
u64     from            -- source node ID
u64     to              -- destination node ID
u64     token           -- token ID carried by this edge
u8      dependency      -- 0=Data, 1=Effect, 2=Control
```

#### Value Encoding

Values are tagged with a `u8` kind discriminant:

| Kind | Tag | Payload |
|------|-----|---------|
| Null | 0 | (none) |
| Bool | 1 | `u8` (0 or 1) |
| Integer | 2 | `i64` LE |
| Float | 3 | `f64` LE (IEEE 754, bit-cast to u64) |
| String | 4 | `u64` length + UTF-8 bytes |
| Array | 5 | `u64` count + Value[count] |
| Object | 6 | `u64` count + (String key + Value)[count] |
| Token | 7 | `u64` token ID |

#### String Encoding

All strings are length-prefixed:

```
u64     length          -- byte length of string (NOT null-terminated)
u8[length]              -- UTF-8 encoded string data
```

#### Bool Encoding

```
u8      value           -- 0 = false, non-zero = true
```

---

## 3. Sync Rules: Adding a New Operation

Follow these steps in order to add a new AIS operation end-to-end:

1. **`crates/apxm-ais/src/operations/definitions.rs`** -- Add variant to the
   `AISOperationType` enum.  Add a corresponding entry in `from_wire_index()`
   with the next available index (currently 25+).  Update `mlir_mnemonic()`,
   `Display`, and all exhaustive match arms.  Add an `OperationSpec` entry to
   the appropriate operations list.

2. **`ArtifactEmitter.cpp`** -- Add a new `OperationKind` enum value with the
   **same integer index** used in step 1.  Add a `.Case<NewOp>()` arm to
   `mapOperation()`.

3. **`AISOps.td`** -- Add the TableGen op definition (`AIS_NewOp`).  This
   drives MLIR C++ codegen for parsing, verification, and printing.

4. **`crates/runtime/apxm-runtime/src/executor/handlers/`** -- Create a new
   handler module (e.g. `new_op.rs`) with `pub async fn execute(...)`.  Register
   the module in `handlers/mod.rs`.

5. **`crates/runtime/apxm-runtime/src/executor/dispatcher.rs`** -- Add a match
   arm routing `AISOperationType::NewOp` to the new handler.

6. **Tests** -- Update `test_operation_counts` and `all_operations_covered`
   assertions to reflect the new totals.  Run the wire round-trip tests.

7. **This document** -- Update the index table in section 1.1.

---

## 4. Shared Constants

`crates/runtime/apxm-core/src/constants.rs` is the canonical source for
string constants shared between the compiler and runtime.  Key namespaces:

- **`constants::graph::attrs`** -- Attribute keys used in node attribute maps
  (e.g. `TEMPLATE_STR`, `QUERY`, `MEMORY_TIER`, `CAPABILITY`, `AGENT_NAME`,
  `FLOW_NAME`, `MODEL`, `TEMPERATURE`, `SYSTEM_PROMPT`, `TOOLS_CONFIG`,
  `TOKEN_BUDGET`, `CASE_LABELS`, `LABEL`, `RECOVERY_TEMPLATE`).

- **`constants::graph::metadata`** -- DAG metadata flags (e.g. `IS_ENTRY`).

- **`constants::inner_plan`** -- Keys for inner-plan structured outputs
  (`GRAPH_PAYLOAD`, `CODELET_DAG`).

- **`constants::diagnostics`** -- Compiler diagnostics modes (`MODE_GRAPH`).

The C++ emitter performs attribute name translation at emission time (e.g.
MLIR `space` becomes runtime `memory_tier`, MLIR `parameters` becomes
`params`).  These translations are hardcoded in `ArtifactEmitter.cpp`'s
`emitNode()` function and must match the keys in `constants::graph::attrs`.
