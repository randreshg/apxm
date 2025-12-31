# APXM Agent Guide

Instructions for AI agents (Claude Code, etc.) working with the APXM codebase.

---

## CLI Setup

Add the `apxm` command to your PATH:

```bash
# Add to ~/.zshrc or ~/.bashrc
export PATH="$PATH:$HOME/path/to/apxm/bin"

# Install dependencies
pip install typer rich
```

The CLI automatically handles environment setup, MLIR configuration, and provides convenient commands.

### Quick Start

```bash
# Check environment status
apxm doctor

# Build the project
apxm build

# Run an AIS file
apxm run examples/hello_world.ais

# Check all workloads compile
apxm workloads check
```

### CLI Commands Reference

| Command | Description |
|---------|-------------|
| `apxm doctor` | Check environment status |
| `apxm build` | Build full project (compiler + runtime) |
| `apxm build --compiler` | Build compiler only |
| `apxm build --runtime` | Build runtime only |
| `apxm build --no-trace` | Build with tracing compiled out (zero overhead) |
| `apxm build --trace` | Build with tracing enabled (default) |
| `apxm run <file.apxmobj>` | Execute a pre-compiled artifact |
| `apxm run <file.apxmobj> --trace <level>` | Execute with tracing |
| `apxm compiler compile <file.ais> -o <output>` | Compile to artifact only |
| `apxm compiler run <file.ais> [-O0/-O1/-O2]` | Compile and run an AIS file |
| `apxm compiler run <file.ais> --trace <level>` | Compile and run with tracing |
| `apxm workloads list` | List available benchmark workloads |
| `apxm workloads check [--verbose]` | Verify all workloads compile |
| `apxm workloads run <name>` | Run a specific workload |

### How It Works

The CLI wrapper (`bin/apxm`):
1. Caches the Python path for fast startup
2. Auto-detects the `apxm` conda environment (even if not activated)
3. Sets `MLIR_DIR`, `LLVM_DIR`, `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` automatically
4. Runs cargo/compiler commands with the correct environment

---

## Manual Environment Setup (Alternative)

If you prefer manual setup instead of the Python CLI:

### Prerequisites

- Rust toolchain (nightly, pinned via `rust-toolchain.toml`)
- mamba or conda for MLIR toolchain

### Install Conda Environment

```bash
cargo run -p apxm-cli -- install
```

This creates/updates the `apxm` conda environment from `environment.yaml`.

### Activate Environment

```bash
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"
```

The `activate` command sets required environment variables for MLIR compilation.

### Verify Setup

```bash
cargo run -p apxm-cli -- doctor
```

You should see `mlir-tblgen` and `cmake/mlir` marked as OK.

### Environment Variables (for CI/non-interactive shells)

```bash
export CONDA_PREFIX=/path/to/miniforge3/envs/apxm
export MLIR_DIR=$CONDA_PREFIX/lib/cmake/mlir
export LLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH  # macOS
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH      # Linux
```

---

## Building

```bash
# Build full project
apxm build

# Build with clean
apxm build --clean

# Build specific components
apxm build --compiler
apxm build --runtime

# Build without tracing (zero overhead for benchmarks)
apxm build --no-trace
```

---

## Runtime Tracing

APXM includes a two-tier tracing system for debugging and understanding execution flow.

### Build-Time Control

Control whether tracing code is compiled into the binary:

```bash
apxm build              # Default: tracing compiled in
apxm build --trace      # Explicit: same as default
apxm build --no-trace   # Zero-overhead: all tracing compiled out
```

When built with `--no-trace`, all tracing macros compile to nothing, providing zero runtime overhead for production/benchmark builds.

### Runtime Control

When built with tracing enabled (default), control output level at runtime:

```bash
apxm run workflow.ais                  # Silent execution (no tracing)
apxm run workflow.ais --trace error    # Only errors
apxm run workflow.ais --trace warn     # Warnings and errors
apxm run workflow.ais --trace info     # High-level execution flow
apxm run workflow.ais --trace debug    # Detailed worker/operation info
apxm run workflow.ais --trace trace    # Full verbosity
```

### Trace Levels and Output

| Level | What You See |
|-------|--------------|
| `error` | Operation failures, scheduler errors |
| `warn` | Retries, fallbacks, potential issues |
| `info` | Execution start/stop, LLM calls with token counts |
| `debug` | Worker dispatch, operation completion, timing |
| `trace` | Token flow, raw LLM responses, internal state |

### Trace Targets

Tracing is organized by subsystem:

| Target | Description |
|--------|-------------|
| `apxm::scheduler` | Worker lifecycle, execution start/stop |
| `apxm::ops` | Operation dispatch, completion, retries |
| `apxm::llm` | LLM requests, responses, token usage |
| `apxm::tokens` | Token production and consumption |
| `apxm::dag` | DAG loading and structure |

### Example Output

**With `--trace info`:**
```
INFO apxm::scheduler: Starting DAG execution nodes=6 max_concurrency=4
INFO apxm::llm: LLM response received tokens_in=12 tokens_out=8
INFO apxm::scheduler: DAG execution completed duration_ms=1445 executed=6
```

**With `--trace debug`:**
```
DEBUG apxm::scheduler: Worker 0 started
DEBUG apxm::ops: Dispatching operation node_id=1 op_type=Rsn
DEBUG apxm::ops: Operation completed successfully duration_ms=425
DEBUG apxm::scheduler: Worker 0 stopped
```

### Workflow: Development vs Production

**Development/debugging:**
```bash
apxm build
apxm run workflow.ais --trace debug
```

**Production/benchmarks:**
```bash
apxm build --no-trace
apxm run workflow.ais  # --trace flag has no effect
```

---

## Running Examples

```bash
# Run an AIS file
apxm run examples/hello_world.ais

# Run with optimization level
apxm run examples/hello_world.ais -O2

# Compile only
apxm compiler compile examples/hello_world.ais -o output.apxmobj
```

### Benchmarks

```bash
apxm workloads list              # List workloads
apxm workloads check             # Verify all compile
apxm workloads run 10_multi_agent
apxm workloads run 1 --json      # Run with JSON output
```

**Prerequisites:** Ollama with `gpt-oss:20b-cloud`: `ollama serve && ollama pull gpt-oss:20b-cloud`

---

## Testing

```bash
# Full workspace tests
cargo test --workspace

# Individual crate tests
cargo test -p apxm-core
cargo test -p apxm-runtime
cargo test -p apxm-compiler
cargo test -p apxm-driver
```

---

## LLM Configuration

AIS programs that use `rsn`, `plan`, `reflect`, `verify`, or `talk` require an LLM backend.

### Create `.apxm/config.toml`

```toml
[chat]
providers = ["ollama"]
default_model = "ollama"
planning_model = "ollama"

[[llm_backends]]
name = "ollama"
provider = "ollama"
model = "gpt-oss:20b-cloud"
endpoint = "http://localhost:11434"
```

### Start Ollama

```bash
ollama serve
ollama pull gpt-oss:20b-cloud
```

### Alternative: Cloud APIs

**OpenAI:**
```toml
[chat]
providers = ["openai"]
default_model = "openai"

[[llm_backends]]
name = "openai"
provider = "openai"
model = "gpt-4o-mini"
api_key = "env:OPENAI_API_KEY"
```

**Google Gemini:**
```toml
[chat]
providers = ["gemini"]
default_model = "gemini"

[[llm_backends]]
name = "gemini"
provider = "google"
model = "gemini-2.5-flash"
api_key = "env:GEMINI_API_KEY"
```

---

## AIS DSL Quick Reference

### Single Agent

```ais
agent HelloWorld {
    @entry flow main() -> str {
        rsn("Generate a greeting") -> greeting
        return greeting
    }
}
```

### Multi-Agent

```ais
agent Researcher {
    flow research(topic: str) -> str {
        rsn("Research: " + topic) -> findings
        return findings
    }
}

agent Coordinator {
    @entry flow main(topic: str) -> str {
        Researcher.research(topic) -> result
        return result
    }
}
```

### Key Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Reasoning | `rsn("prompt") -> var` | LLM reasoning |
| Planning | `plan("goal") -> var` | Generate execution plan |
| Reflection | `reflect("trace_id") -> var` | Self-reflection |
| Verification | `verify expr -> var` | Verify condition |
| Memory Query | `qmem(store, query) -> var` | Query memory tier |
| Memory Update | `umem(store, key, value)` | Update memory |
| Flow Call | `Agent.flow(args) -> var` | Cross-agent call |

### Reasoning Syntax (Explicit)

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

### Memory Tiers

- `STM` - Short-term memory (session-scoped)
- `Episodic` - Episode memory (interaction history)
- `LTM` - Long-term memory (persistent)

### Dataflow Execution Semantics

APXM uses **dataflow execution**: operations run when their inputs are ready, not in textual order.

```ais
rsn("analyze", data) -> result
print(result)           // Runs AFTER rsn (depends on result)
print("Done!")          // Runs IMMEDIATELY (no data dependency!)
```

The `print("Done!")` has no inputs, so it can run at any time—even before `rsn` completes.

**To enforce ordering**, pass data to create a dependency:

```ais
rsn("analyze", data) -> result
print(result)                    // Depends on result
print("Done: ", result)          // Also depends on result - runs after rsn
```

This dataflow model enables automatic parallelism: independent operations run concurrently without explicit threading.

---

## Troubleshooting

### "MLIR toolchain not detected"

```bash
conda activate apxm
echo $CONDA_PREFIX  # Should show .../envs/apxm
```

### "Library not loaded: @rpath/libapxm_compiler_c.dylib"

```bash
eval "$(cargo run -p apxm-cli -- activate)"
# Or manually set DYLD_LIBRARY_PATH
```

### Rebuild after C++ changes

```bash
rm -rf target/release/build/apxm-compiler-*
cargo build -p apxm-compiler --release
```

---

## Crate Overview

| Crate | Description | Requires MLIR |
|-------|-------------|---------------|
| `apxm-core` | Core types and errors | No |
| `apxm-ais` | AIS DSL definitions | No |
| `apxm-artifact` | Artifact format | No |
| `apxm-runtime` | DAG executor | No |
| `apxm-backends` | LLM/storage backends | No |
| `apxm-compiler` | MLIR-based compiler | **Yes** |
| `apxm-driver` | Orchestration layer | Yes (via compiler) |
| `apxm-cli` | Command-line interface | Optional (driver feature) |

---

## Common Workflows

### Add a new AIS operation

1. Define in `crates/apxm-ais/src/operations/definitions.rs`
2. Add handler in `crates/runtime/apxm-runtime/src/executor/handlers/`
3. Add lexer token in `crates/apxm-compiler/mlir/lib/Parser/Lexer/Lexer.cpp`
4. Add parser support in `crates/apxm-compiler/mlir/lib/Parser/`
5. Add MLIR codegen in `crates/apxm-compiler/mlir/lib/MLIRGen/`

### Debug compilation errors

```bash
# Enable verbose output (using compiled binary)
RUST_LOG=debug ./target/release/apxm compile file.ais -o /dev/null

# Check MLIR output
./target/release/apxm compile file.ais --emit-mlir -o output.apxmobj
```

**Note:** For debugging, you need to have the environment activated first (see "Manual Environment Setup" above).

### Dump IR after each pass

```bash
# Write per-pass MLIR snapshots to the given directory.
APXM_PRINT_IR_DIR=/tmp/apxm-ir apxm compiler compile file.ais -o output.apxmobj

# Optional: print a one-line trace of IR printing config.
APXM_PRINT_IR_TRACE=1 APXM_PRINT_IR_DIR=/tmp/apxm-ir apxm compiler compile file.ais -o output.apxmobj
```

### Inspect artifact contents

```bash
cargo run -p apxm-artifact --example inspect -- path/to/file.apxmobj
```
