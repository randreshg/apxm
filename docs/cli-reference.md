# APXM CLI Reference

Complete command reference for the APXM CLI. For installation and first steps, see [Getting Started](getting-started.md).

---

## Commands Overview

| Command | Description |
|---------|-------------|
| `apxm install` | Install/update environment (conda, Rust, build, wrapper) |
| `apxm doctor` | Check environment status and dependencies |
| `apxm build` | Build compiler and runtime |
| `apxm compile <file> -o <out>` | Compile to `.apxmobj` artifact |
| `apxm execute <file> [args]` | Compile and run in one step |
| `apxm run <file.apxmobj> [args]` | Run a pre-compiled artifact |
| `apxm test` | Run test suite |
| `apxm register <subcommand>` | Manage LLM credentials |
| `apxm workloads list\|check\|run` | Manage benchmark workloads |
| `apxm benchmarks run` | Run A-PXM vs LangGraph benchmarks |

---

## Install

```bash
apxm install                # Full install (interactive)
apxm install --check        # Dry-run: report status without changes
apxm install --auto         # Automatic mode (no prompts)
apxm install --skip-deps    # Skip dependency checks
apxm install --skip-build   # Skip build step
```

**Stages:** Platform detection, dependency checks, conda environment, Rust toolchain, build, wrapper generation.

Long-running output (conda, cargo) is captured to `.apxm/install.log` — on failure the last ~30 lines are printed automatically.

---

## Doctor

```bash
apxm doctor
```

Checks: platform, dependencies (Rust, Cargo, CMake, Ninja, Mamba/Conda, Git), conda environment (MLIR/LLVM 21), build status, credentials, CI environment. Exit code `0` = all passed, `1` = failures found.

---

## Build

```bash
apxm build                  # Full project (compiler + runtime)
apxm build --compiler       # Compiler only
apxm build --runtime        # Runtime only
apxm build --debug          # Debug build
apxm build --clean          # Clean before building
apxm build --no-trace       # Zero-overhead (tracing compiled out)
```

---

## Compile

```bash
apxm compile workflow.json -o workflow.apxmobj
apxm compile workflow.json -o workflow.apxmobj -O2
apxm compile workflow.json -o workflow.apxmobj --emit-diagnostics diag.json
apxm compile workflow.json -o workflow.apxmobj --cargo   # Auto-build first
```

**Optimization levels:** `-O0` (no passes), `-O1` (default: normalize, fuse, CSE, DCE), `-O2` (additional MLIR passes), `-O3` (maximum).

---

## Execute & Run

```bash
# Compile + run in one step
apxm execute workflow.json "input text"
apxm execute workflow.json --trace debug
apxm execute workflow.json --emit-metrics metrics.json

# Run pre-compiled artifact
apxm run workflow.apxmobj "input text"
apxm run workflow.apxmobj --trace info
```

---

## Tracing

Build-time control:

```bash
apxm build                  # Tracing compiled in (default)
apxm build --no-trace       # Tracing compiled out (zero overhead)
```

Runtime control (when built with tracing):

| Level | What You See |
|-------|--------------|
| `error` | Operation failures, scheduler errors |
| `warn` | Retries, fallbacks, potential issues |
| `info` | Execution start/stop, LLM calls with token counts |
| `debug` | Worker dispatch, operation completion, timing |
| `trace` | Token flow, raw LLM responses, internal state |

Trace targets: `apxm::scheduler`, `apxm::ops`, `apxm::llm`, `apxm::tokens`, `apxm::dag`.

---

## Credentials

See [LLM Backends](llm-backends.md) for full provider documentation.

```bash
apxm register add <name> --provider <type> --api-key <key>
apxm register add <name> --provider <type>   # Interactive (hidden input)
apxm register list                           # List (keys masked)
apxm register test [name]                    # Test credential(s)
apxm register remove <name>                  # Delete
apxm register generate-config                # Export to config.toml
```

Supported providers: `openai`, `anthropic`, `google`, `ollama`, `openrouter`.

---

## Testing

**No API keys needed.** All 375+ tests use `MockLLMBackend`.

```bash
apxm test                   # All tests except compiler
apxm test --all             # All tests (requires MLIR/LLVM 21)
apxm test --runtime         # Runtime (133 tests)
apxm test --compiler        # Compiler (requires MLIR)
apxm test --credentials     # Credential store
apxm test --backends        # Backend mocks (74 tests)
apxm test --package <name>  # Specific crate
```

---

## Workloads & Benchmarks

```bash
apxm workloads list                           # List workloads
apxm workloads check                          # Verify all compile
apxm workloads run <name>                     # Run specific workload

apxm benchmarks run --workloads --tables      # Full benchmark suite
apxm benchmarks run --workloads --quick       # Quick mode (3 iterations)
apxm benchmarks run --workloads --workload 1  # Specific workload
apxm benchmarks run --list                    # List available
```

---

## Configuration

APXM looks for configuration in this order:
1. `--config` flag (explicit path)
2. `.apxm/config.toml` (project-local, walking up from cwd)
3. `~/.apxm/config.toml` (global)

```toml
[chat]
providers = ["my-openai"]
default_model = "gpt-4"

# Only needed if NOT using `apxm register`:
[[llm_backends]]
name = "my-openai"
provider = "openai"
api_key = "env:OPENAI_API_KEY"
model = "gpt-4"
```

When using `apxm register`, the credential store is the source of truth — `[[llm_backends]]` is unnecessary.

---

## IR Debugging

```bash
# Write per-pass MLIR snapshots
APXM_PRINT_IR_DIR=/tmp/apxm-ir apxm compiler compile file.json -o out.apxmobj

# With trace of IR printing config
APXM_PRINT_IR_TRACE=1 APXM_PRINT_IR_DIR=/tmp/apxm-ir apxm compiler compile file.json -o out.apxmobj

# Inspect artifact contents
cargo run -p apxm-artifact --example inspect -- path/to/file.apxmobj
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

### Rebuild after C++ changes

```bash
rm -rf target/release/build/apxm-compiler-*
apxm build --compiler
```

---

## Troubleshooting

**"apxm: command not found"** — Restart shell (`source ~/.bashrc`) or check `ls ~/.local/bin/apxm`.

**"MLIR toolchain not detected" / "Library not loaded"** — Re-run `apxm install` to regenerate the wrapper.

**Build failures** — Check `.apxm/install.log` for full output.

**Credential issues** — Run `apxm register test` to verify API connectivity.
