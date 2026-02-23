# APXM Development Guide

## Prerequisites

- **conda/mamba** — for managing the build environment
- **Rust nightly** — specified via `rust-toolchain.toml`
- **LLVM/MLIR 21** — only required for the compiler crate (`apxm-compiler`)

## Environment Setup

```bash
mamba env create -f environment.yaml
conda activate apxm
```

This installs LLVM/MLIR, Python (for CLI tooling), cmake, and system libraries.

## Building

```bash
make build          # Runtime only (excludes compiler)
make build-all      # All crates including compiler (requires MLIR)
make build-compiler # Compiler only
make build-runtime  # Runtime only
make build-release  # Release mode
```

## Testing

```bash
make test           # All tests except compiler
make test-all       # All tests including compiler
make test-runtime   # Runtime tests only
cargo test -p <crate>  # Single crate
```

**Note:** The `default_path_respects_home` test in `apxm-cli` can be flaky depending on environment variables.

## Running Examples

```bash
apxm execute examples/hello.ais
apxm compile examples/hello.ais -o hello.apxmobj
apxm run hello.apxmobj
```

Available examples: `hello.ais`, `multi_flow.ais`, `tool_use.ais`.

## Project Structure

| Crate | Description |
|-------|-------------|
| `apxm-ais` | AIS domain-specific language definitions |
| `apxm-cli` | Command-line interface |
| `apxm-compiler` | AIS → MLIR → executable compiler (requires LLVM/MLIR) |
| `apxm-driver` | High-level driver coordinating compile + run |
| `apxm-graph` | Graph IR and JSON serialization |
| `apxm-server` | HTTP server for remote execution |
| `apxm-tools` | Built-in tool definitions (bash, read, write, etc.) |
| `runtime/apxm-artifact` | Compiled artifact format (.apxmobj) |
| `runtime/apxm-backends` | LLM backend adapters (Ollama, OpenAI, Google) |
| `runtime/apxm-core` | Core types, error system, and execution DAG |
| `runtime/apxm-runtime` | Scheduler, memory, and execution engine |

## Code Quality

```bash
make fmt        # Format code
make fmt-check  # Check formatting
make lint       # Run clippy
make check      # Fast compile check
```

## Further Reading

- [Getting Started](docs/GETTING_STARTED.md) — installation, configuration, LLM setup
- [Agents Guide](docs/AGENTS.md) — CLI reference for agents
- [Graph JSON Contract](docs/GRAPH_JSON_CONTRACT.md) — stable JSON format for graph generators
