# Crate Documentation

Each crate has its own `README.md` describing its responsibilities, key APIs, and testing notes.

## Tooling

- `crates/apxm-cli/README.md`: CLI usage (doctor/install/compile/run)
- `crates/apxm-driver/README.md`: Driver orchestration and config
- `crates/apxm-compiler/README.md`: MLIR compiler architecture
- `crates/apxm-compiler/dsl/README.md`: DSL front-end notes
- `crates/apxm-ais/README.md`: AIS operation definitions

## Runtime

- `crates/runtime/apxm-runtime/README.md`: Runtime execution model
- `crates/runtime/apxm-backends/README.md`: LLM + storage backends
- `crates/runtime/apxm-core/README.md`: Core types/errors shared across crates
- `crates/runtime/apxm-artifact/README.md`: Artifact format and I/O
