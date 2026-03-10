# Agent Task: Revise APXM for Rust Best Practices

## Objective

You are to **research** established Rust best practices for medium-to-large codebases, then **audit and revise** the `/home/raherrer/projects/agents/apxm` project so that it consistently applies those practices. Focus on clarity, maintainability, and explicit contracts—without overengineering.

---

## Phase 1: Research (Do First)

1. **Look up current Rust best practices** for:
   - Project layout (workspace, crates, modules)
   - Type design: enums over magic numbers/strings, newtypes where helpful
   - Avoiding hardcoded strings (constants, config, or generated code)
   - Interfaces: traits for behavior contracts, sealed traits where appropriate
   - Documentation: crate/module/public item docs, examples, and contract docs
   - Error handling: typed errors, context, and consistency
   - Testing: unit vs integration, coverage of contracts
   - Avoiding overengineering: YAGNI, single source of truth, minimal indirection

2. **Identify official or widely cited sources** (e.g. Rust API guidelines, rust-lang docs, clippy lints) and note which practices you will apply.

3. **Summarize** in a short list the practices you will enforce in this codebase, with one or two sentence rationale each.

---

## Phase 2: Audit the APXM Codebase

Before changing code, map the following in the repo:

### 2.1 Contract: MLIR Ops ↔ Runtime

The pipeline is:

- **Source of truth (Rust):** `apxm-ais` defines `AISOperationType` and `mlir_mnemonic()` for every AIS op.
- **Compiler (C++):** MLIR TableGen (`AISOps.td`) and `ArtifactEmitter.cpp` define `OperationKind` and emit a **binary artifact** where each node has an **op kind index** (u32).
- **Compiler (Rust):** `apxm-compiler` in `codegen/artifact.rs` uses `OP_KIND_MAP` to map that index → `AISOperationType` when **reading** the artifact.
- **Runtime:** `apxm-runtime` dispatcher matches on `AISOperationType` and calls the right handler; for the **codelet/JSON path**, `executor/handlers/switch.rs` has `map_op_type(u32) -> AISOperationType` which must match the same index convention.

**You must:**

- Trace and document the full contract: **who defines op kinds, in what order, and where that order is duplicated** (C++ `OperationKind`, Rust `OP_KIND_MAP`, Rust `map_op_type`).
- Identify any **stale or redundant** definitions (e.g. `apxm-core` has an `OpType` enum in `types/compiler/binary_format.rs` that is a subset and uses old names like `Rsn`; artifact path uses `OP_KIND_MAP` and wire format uses `AISOperationType`).
- Propose a **single source of truth** and how C++ and Rust should stay in sync (e.g. generated code, or a single documented ordering with tests).

### 2.2 Enums and Constants

- List places that use **magic numbers or string literals** for:
  - Operation kinds, attribute keys, error codes, or protocol fields.
- Check whether **enums** are used consistently for op types, dependency types, and similar concepts.
- Verify that **shared constants** (e.g. `apxm-core/src/constants.rs` for graph/attr keys) are used wherever protocol or graph fields are referenced, and that no new ad-hoc string keys are introduced without going through constants.

### 2.3 Interfaces and Types

- Identify where **traits** would make the contract explicit (e.g. “something that can execute a node” → e.g. `OperationHandler` or similar).
- Check that **public APIs** are expressed in terms of types and traits, not loose functions and strings.
- Note any **overly generic or unnecessary abstractions** that should be simplified (YAGNI).

### 2.4 Documentation

- Check **crate roots** and **public modules** for module-level docs.
- Check that **public types and functions** that are part of a contract (e.g. artifact format, op kinds, handler signature) are clearly documented.
- Identify missing **contract documentation**: e.g. artifact wire format, op kind index table, MLIR op name ↔ runtime behavior.

### 2.5 Overengineering and Duplication

- Flag **duplicated logic** (e.g. op kind mapping in more than one place).
- Flag **unnecessary layers or indirection** that don’t add clarity or reuse.
- Ensure **one canonical place** for op kinds, attribute names, and wire-format rules.

---

## Phase 3: Apply Changes

Apply the practices you listed in Phase 1, guided by the audit. In particular:

1. **Contracts**
   - Document the **MLIR ↔ runtime** and **artifact wire** contract in one place (e.g. `docs/CONTRACTS.md` or a dedicated section in an existing doc). Include:
     - Where op kinds are defined and in what order.
     - Where that order must be kept in sync (C++ `OperationKind`, Rust `OP_KIND_MAP`, `map_op_type`).
   - Add **tests or assertions** that enforce the contract (e.g. that `OP_KIND_MAP` length and order match the C++ side, or that all `AISOperationType` variants are handled in the dispatcher).
   - If possible, **remove or replace** redundant enums (e.g. legacy `OpType` in `binary_format.rs`) with the single source of truth, or clearly mark them as legacy and point to the canonical definition.

2. **No hardcoded strings**
   - Use **constants** (or generated code) for attribute keys, graph metadata keys, and any protocol strings shared between compiler and runtime. Prefer `apxm-core` constants or a shared crate over scattered literals.
   - Replace **string literals** in error messages with constants where they denote protocol fields or fixed identifiers; free-form messages can stay as format strings.

3. **Enums and types**
   - Use **enums** for operation kinds, dependency kinds, and similar closed sets; avoid raw integers or string comparisons where an enum is clearer.
   - Ensure **wire/artifact format** uses the same enum or a documented mapping from a single definition.

4. **Interfaces**
   - Introduce a **trait** for “execute this node” (e.g. `OperationHandler`) if it makes the dispatcher and testing clearer; implement it for each handler module. Keep the trait minimal (e.g. one method). If the current function-based dispatch is already clear and tested, avoid adding heavy machinery.

5. **Documentation**
   - Add or improve **module-level** docs for crates and key modules (especially `apxm-ais`, `apxm-core` types, `apxm-compiler` codegen, `apxm-runtime` executor).
   - Add **doc comments** for all public types and functions that define or depend on the compiler/runtime contract.
   - In docs, **link** to the central contract description (e.g. `CONTRACTS.md`) where relevant.

6. **Avoid overengineering**
   - Do **not** add abstractions “for flexibility” without a concrete use case.
   - Prefer **one clear implementation** and a single source of truth over multiple competing mechanisms.
   - Simplify or remove **redundant** types and mappings once the contract is documented and centralized.

---

## Phase 4: Verification

- Run **tests**: `cargo test` (and any project-specific test commands).
- Run **clippy**: `cargo clippy --all-targets` and address new lints that align with the chosen practices.
- Optionally run **fmt**: `cargo fmt`.
- Confirm that **contract tests** (e.g. op kind order, handler coverage) are in place and pass.

---

## Deliverables

1. **Short research summary** (Phase 1): list of practices and sources.
2. **Audit notes** (Phase 2): contract flow, duplication, and gaps (can be in a comment in a doc or a small `docs/AUDIT.md`).
3. **Code and doc changes** (Phase 3): concrete edits to the repo.
4. **Contract document** (or section): single place that describes MLIR op ↔ runtime and artifact wire format, with op kind ordering and sync points.
5. **Verification**: test and clippy commands run and passing.

---

## Key Paths (for reference)

| Area              | Paths |
|-------------------|--------|
| Op type source of truth | `crates/apxm-ais/src/operations/definitions.rs` (`AISOperationType`, `mlir_mnemonic`) |
| Artifact op kind map (Rust) | `crates/apxm-compiler/src/codegen/artifact.rs` (`OP_KIND_MAP`) |
| Runtime op kind map (codelet path) | `crates/runtime/apxm-runtime/src/executor/handlers/switch.rs` (`map_op_type`) |
| C++ artifact emission | `crates/apxm-compiler/mlir/lib/Dialect/AIS/Conversion/Artifact/ArtifactEmitter.cpp` (`OperationKind`) |
| MLIR op definitions | `crates/apxm-compiler/mlir/include/ais/Dialect/AIS/IR/AISOps.td` |
| Runtime dispatcher | `crates/runtime/apxm-runtime/src/executor/dispatcher.rs` |
| Binary format (legacy?) | `crates/runtime/apxm-core/src/types/compiler/binary_format.rs` (`OpType`) |
| Shared constants | `crates/runtime/apxm-core/src/constants.rs` |
| Wire format (artifact) | `crates/runtime/apxm-artifact/src/wire.rs` |

Use this prompt as the full instruction set for the agent. The agent must complete Phase 1 before Phase 2, and Phase 2 before Phase 3; Phase 4 validates the outcome.
