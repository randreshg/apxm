# A-PXM Implementation Plan (No CLI)

**Focus**: instrumentation, compiler clarity, and documentation for the CFP paper.

---

## Executive Summary

**Goal**: Align implementation with the paper by:
- Adding zero-overhead (compile-time gated) metrics for tokens and latency.
- Making the DSL → MLIR → artifact pipeline explicit and documented.
- Documenting each crate with clear responsibilities and test guidance.

---

## Core Deliverables

1. **Runtime + LLM Metrics**
   - Token usage, latency, retries, and per-op timing recorded when enabled.
   - Compile-time gating ensures zero overhead when disabled.

2. **Compiler Clarity**
   - Clear DSL process documentation (parser → MLIR → passes → codegen).
   - Explicit diagram of TableGen generation from Rust definitions.

3. **Documentation**
   - Per-crate README summaries describing responsibilities and integration.
   - Updated architecture diagrams and flows without CLI references.

4. **Tests**
   - Per-crate testing entry points and targeted checks.

---

## Roadmap

### Phase 1: Metrics + Instrumentation
- Add feature flags for metrics in runtime/backends.
- Record per-LLM-call usage and per-node execution timing.
- Emit episodic events for LLM calls (when enabled).

### Phase 2: Compiler/DSL Clarity
- Add a DSL documentation folder or `DSL.md` under `crates/apxm-compiler/`.
- Document the DSL → MLIR → artifact lowering path.
- Add a pipeline diagram that shows TableGen generation and sharing.

### Phase 3: Docs + Tests
- Per-crate README updates with “how it fits” and “how to test”.
- Add a compilation-process diagram to `docs/diagrams.md`.
- Document test scopes per crate (unit vs integration).

---

## Paper Alignment Notes

- **COMMUNICATE** is currently a stub in the runtime; the paper should reflect this.
- LLM token usage is tracked by backends; runtime aggregation is pending (Phase 1).
