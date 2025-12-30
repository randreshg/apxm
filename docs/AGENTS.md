# APXM Agent Guide

Instructions for AI agents (Claude Code, etc.) working with the APXM codebase.

---

## Python CLI (Recommended)

The Python CLI (`tools/apxm_cli.py`) automatically handles environment setup, MLIR configuration, and provides convenient commands. **Use this instead of manual environment activation.**

### Quick Start

```bash
# Check environment status
python tools/apxm_cli.py doctor

# Build the compiler (auto-sets MLIR environment)
python tools/apxm_cli.py compiler build

# Run an AIS file
python tools/apxm_cli.py compiler run examples/hello_world.ais

# Compile only
python tools/apxm_cli.py compiler compile file.ais -o output.apxmobj

# Check all workloads compile
python tools/apxm_cli.py workloads check

# List available workloads
python tools/apxm_cli.py workloads list

# Run a specific workload benchmark
python tools/apxm_cli.py workloads run 10_multi_agent
```

### CLI Commands Reference

| Command | Description |
|---------|-------------|
| `doctor` | Check environment status (conda, MLIR, compiler binary) |
| `compiler build [--release/--debug] [--clean]` | Build the APXM compiler |
| `compiler run <file.ais> [-O0/-O1/-O2] [--cargo]` | Compile and run an AIS file |
| `compiler compile <file.ais> -o <output> [--cargo]` | Compile to artifact only |
| `workloads list` | List available benchmark workloads |
| `workloads check [--verbose]` | Compile all workloads to verify syntax |
| `workloads run <name>` | Run a specific workload benchmark |
| `workloads benchmark <name> [--all] [-n N] [-w N] [--json]` | Run benchmarks with iterations/warmup |

**Note:** The `--cargo` flag uses `cargo run` instead of the pre-built binary. This is useful for development when iterating on the compiler (auto-rebuilds on each run).

### How It Works

The Python CLI:
1. Auto-detects the `apxm` conda environment (even if not activated)
2. Sets `MLIR_DIR`, `LLVM_DIR`, `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` automatically
3. Runs cargo/compiler commands with the correct environment

### Dependencies

```bash
pip install typer rich
```

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

### Using Python CLI (Recommended)

```bash
# Build compiler with automatic environment setup
python tools/apxm_cli.py compiler build

# Build with clean (removes old build artifacts first)
python tools/apxm_cli.py compiler build --clean
```

### Runtime Only (No MLIR Required)

These crates can be built without the MLIR toolchain:

```bash
cargo build -p apxm-runtime
cargo build -p apxm-core
cargo build -p apxm-artifact
cargo build -p apxm-ais
cargo build -p apxm-backends
```

### Compiler (Requires MLIR - Manual)

If not using the Python CLI, activate the MLIR environment manually:

```bash
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"
cargo build -p apxm-compiler --release
```

### CLI with Driver Feature

The CLI's `driver` feature enables compilation and execution:

```bash
cargo build -p apxm-cli --features driver --release
```

### Full Workspace

```bash
cargo build --workspace
```

**Note:** Full workspace build requires MLIR environment.

---

## Running Examples

### Using Python CLI (Recommended)

```bash
# Run an AIS file
python tools/apxm_cli.py compiler run examples/hello_world.ais

# Run with optimization level
python tools/apxm_cli.py compiler run examples/hello_world.ais -O2

# Compile only
python tools/apxm_cli.py compiler compile examples/hello_world.ais -o output.apxmobj
```

### DSL Examples (Manual - After Building Compiler)

If you've built the compiler manually, you can use the binary directly:

```bash
# Activate environment first
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"

# Use the compiled binary
./target/release/apxm run examples/hello_world.ais
./target/release/apxm compile examples/hello_world.ais -o output.apxmobj
./target/release/apxm run examples/hello_world.ais -O2
```

**Note:** The Python CLI handles environment setup automatically. Use it for a simpler experience.

### Runtime Examples (No Compiler Required)

```bash
cargo run -p apxm-runtime --example basic_runtime
cargo run -p apxm-runtime --example memory_tiers
cargo run -p apxm-runtime --example simple_llm
```

### Benchmarks

```bash
# Using Python CLI
python tools/apxm_cli.py workloads list     # List available workloads
python tools/apxm_cli.py workloads check    # Verify all workloads compile
python tools/apxm_cli.py workloads run 10_multi_agent  # Run specific workload

# Run benchmarks with proper environment and iteration control
python tools/apxm_cli.py workloads benchmark 2_chain_fusion
python tools/apxm_cli.py workloads benchmark --all --json -o results.json
python tools/apxm_cli.py workloads benchmark --all -n 10 -w 3

# Or manually
cd papers/cf26/benchmarks
python workloads/10_multi_agent/run.py      # Run a specific workload
python run_all.py --workloads --quick       # Run all workloads (quick mode)
python run_all.py --paper                   # Run paper benchmarks
```

**Prerequisites for benchmarks:**

- Ollama running with `phi3:mini`: `ollama serve && ollama pull phi3:mini`
- Python deps: `pip install langgraph langchain-ollama typer rich`

---

## Testing

### Core Tests (No MLIR Required)

```bash
cargo test -p apxm-core
cargo test -p apxm-artifact
cargo test -p apxm-ais
cargo test -p apxm-runtime
```

### Compiler Tests (Requires MLIR)

```bash
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"
cargo test -p apxm-compiler
```

### Driver Tests

```bash
cargo test -p apxm-driver
```

### Full Workspace

```bash
cargo test --workspace
```

### Feature-gated Tests

```bash
cargo test -p apxm-runtime --features metrics
cargo test -p apxm-driver --features metrics
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
model = "phi3:mini"
endpoint = "http://localhost:11434"
```

### Start Ollama

```bash
ollama serve
ollama pull phi3:mini
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
        rsn "Generate a greeting" -> greeting
        return greeting
    }
}
```

### Multi-Agent

```ais
agent Researcher {
    flow research(topic: str) -> str {
        rsn "Research: " + topic -> findings
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
| Reasoning | `rsn "prompt" -> var` | LLM reasoning |
| Planning | `plan "goal" -> var` | Generate execution plan |
| Reflection | `reflect "context" -> var` | Self-reflection |
| Verification | `verify expr -> var` | Verify condition |
| Memory Query | `qmem(store, query) -> var` | Query memory tier |
| Memory Update | `umem(store, key, value)` | Update memory |
| Flow Call | `Agent.flow(args) -> var` | Cross-agent call |

### Memory Tiers

- `STM` - Short-term memory (session-scoped)
- `Episodic` - Episode memory (interaction history)
- `LTM` - Long-term memory (persistent)

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

### Inspect artifact contents

```bash
cargo run -p apxm-artifact --example inspect -- path/to/file.apxmobj
```
