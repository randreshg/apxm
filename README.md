# APXM – Agent Programming eXecution Model

APXM is a full toolchain for building autonomous agents:

- **AIS DSL** for declaring memory, flows, handlers, and tool invocations
- **Compiler** that lowers AIS → MLIR → executable artifacts
- **Runtime** with scheduler, memory system, and LLM registry
- **CLI** for compile/run workflows

---

## Quick Start

```bash
git clone https://github.com/randreshg/apxm
cd apxm

# Install MLIR/LLVM toolchain
cargo run -p apxm-cli -- install
pip install typer rich

# Add apxm to PATH (add to ~/.zshrc or ~/.bashrc for persistence)
export PATH="$PATH:$(pwd)/bin"

# Build the project
apxm build

# Run an example
apxm run examples/hello_world.ais
```

## Prerequisites

- **mamba** or **conda** ([miniforge](https://github.com/conda-forge/miniforge) recommended)
- **Rust nightly** (managed via `rust-toolchain.toml`)
- **Python 3.10+** with `typer` and `rich`

---

## CLI Commands

```bash
apxm doctor           # Check environment
apxm build            # Build full project
apxm build --compiler # Build compiler only
apxm build --runtime  # Build runtime only
apxm build --no-trace # Build without tracing (zero overhead)
apxm run <file>       # Compile and run
apxm run <file> --trace debug  # Run with debug tracing
apxm workloads check  # Validate workloads
```

See [docs/AGENTS.md](docs/AGENTS.md) for complete reference.

---

## Runtime Tracing

APXM includes a two-tier tracing system for debugging and performance analysis:

**Build-time control:**
```bash
apxm build              # Default: tracing compiled in
apxm build --no-trace   # Zero-overhead build (tracing compiled out)
```

**Runtime control** (when built with tracing):
```bash
apxm run workflow.ais                  # Silent execution
apxm run workflow.ais --trace info     # High-level execution flow
apxm run workflow.ais --trace debug    # Detailed worker/operation info
apxm run workflow.ais --trace trace    # Full verbosity (tokens, LLM calls)
```

Trace targets: `apxm::scheduler`, `apxm::ops`, `apxm::llm`, `apxm::tokens`, `apxm::dag`

---

## Reasoning Syntax (Explicit)

`rsn` now requires parentheses, even for single-argument prompts.

```ais
// Single prompt expression (token concatenation).
rsn("Explain the domain background of " + topic) -> background

// Template + context operands (comma separates context).
rsn("Execute step 1: ", steps) -> step1_result
```

The comma does not concatenate. It passes context operands, which are appended at runtime as:

```
<template>

Context 1: <value>
Context 2: <value>
```

`+` merges tokens and cannot be used with goals/handles; use the comma form for those.

---

## IR Debugging

```bash
# Write per-pass MLIR snapshots to the given directory.
APXM_PRINT_IR_DIR=/tmp/apxm-ir apxm compiler compile file.ais -o output.apxmobj

# Optional: print a one-line trace of IR printing config.
APXM_PRINT_IR_TRACE=1 APXM_PRINT_IR_DIR=/tmp/apxm-ir apxm compiler compile file.ais -o output.apxmobj
```

---

## Configuration

Create `.apxm/config.toml`:

```toml
[chat]
providers = ["ollama"]
default_model = "ollama"

[[llm_backends]]
name = "ollama"
provider = "ollama"
model = "gpt-oss:20b-cloud"
endpoint = "http://localhost:11434"
```

---

## Project Layout

```text
crates/
  apxm-ais        # AIS operation definitions
  apxm-cli        # CLI tool
  apxm-compiler   # MLIR compiler
  apxm-driver     # Compiler+runtime orchestration
  runtime/
    apxm-artifact # Artifact format
    apxm-backends # LLM and storage backends
    apxm-core     # Shared types
    apxm-runtime  # Execution engine
examples/         # Sample AIS programs
docs/             # Documentation
```

---

## Documentation

- [AGENTS.md](docs/AGENTS.md) - CLI reference and usage
- [architecture.md](docs/architecture.md) - System architecture
- [contract.md](docs/contract.md) - Compiler-runtime contract
