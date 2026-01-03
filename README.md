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

# Create conda environment (includes Python, MLIR/LLVM, and all dependencies)
mamba env create -f environment.yaml

# Add apxm to PATH (add to ~/.zshrc or ~/.bashrc for persistence)
export PATH="$PATH:$(pwd)/bin"

# Build the project
apxm build

# Verify installation
apxm doctor

# Run an example
apxm execute examples/hello_world.ais
```

## Prerequisites

- **mamba** or **conda** ([miniforge](https://github.com/conda-forge/miniforge) recommended)
- **Rust nightly** (managed via `rust-toolchain.toml`)

---

## CLI Commands

```bash
apxm doctor                     # Check environment
apxm install                    # Install/update conda environment
apxm build                      # Build full project
apxm build --compiler           # Build compiler only
apxm build --runtime            # Build runtime only
apxm build --no-trace           # Build without tracing (zero overhead)
apxm execute <file.ais>         # Compile and run an AIS file
apxm execute <file.ais> --trace debug  # Run with debug tracing
apxm compile <file.ais> -o out.apxmobj # Compile to artifact
apxm run <file.apxmobj>         # Run pre-compiled artifact
apxm workloads list             # List benchmark workloads
apxm workloads check            # Validate workloads compile
apxm workloads run <name>       # Run a specific workload
apxm benchmarks run --workloads # Run all benchmark workloads
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
apxm execute workflow.ais                  # Silent execution
apxm execute workflow.ais --trace info     # High-level execution flow
apxm execute workflow.ais --trace debug    # Detailed worker/operation info
apxm execute workflow.ais --trace trace    # Full verbosity (tokens, LLM calls)
```

Trace targets: `apxm::scheduler`, `apxm::ops`, `apxm::llm`, `apxm::tokens`, `apxm::dag`

---

## LLM Operations

APXM provides three core LLM operations with different reasoning characteristics:

| Operation | Purpose | Example |
|-----------|---------|---------|
| `ask` | Simple Q&A with LLM | `ask("What is 2+2?") -> answer` |
| `think` | Extended thinking with token budget | `think("Analyze this problem", budget: 1000) -> analysis` |
| `reason` | Structured reasoning with belief updates | `reason("Solve step by step", context) -> solution` |

```ais
// Simple ask - direct question and answer
ask("Explain the domain background of " + topic) -> background

// Think - extended reasoning with budget
think("Analyze the implications", data, budget: 2000) -> analysis

// Reason - structured with context operands
reason("Execute step 1: ", steps) -> step1_result
```

Context operands (comma-separated) are appended at runtime as:

```
<template>

Context 1: <value>
Context 2: <value>
```

The `+` operator merges tokens; use the comma form for context operands.

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

- [Getting Started](docs/GETTING_STARTED.md) - Installation, setup, and first steps
- [AGENTS.md](docs/AGENTS.md) - CLI reference and AI agent guide
- [Benchmark Workloads](papers/CF26/benchmarks/workloads/README.md) - DSL comparison benchmarks with examples
