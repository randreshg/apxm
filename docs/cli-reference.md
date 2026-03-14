# APXM CLI Reference

Complete command reference for the APXM CLI. For installation and first steps, see [Getting Started](getting-started.md).

---

## Commands Overview

| Command | Description |
|---------|-------------|
| `apxm install` | Install/update environment (conda, Rust, wrapper) |
| `apxm doctor` | Check environment status and dependencies |
| `apxm activate` | Print shell exports for MLIR/LLVM env setup |
| `apxm compile <file> -o <out>` | Compile ApxmGraph JSON/binary to `.apxmobj` artifact |
| `apxm execute <file> [args]` | Compile and run in one step |
| `apxm run <file.apxmobj> [args]` | Run a pre-compiled artifact |
| `apxm validate <file>` | Validate an ApxmGraph JSON file |
| `apxm analyze <file>` | Analyze an ApxmGraph for parallelism |
| `apxm register <subcommand>` | Manage LLM credentials |
| `apxm tools <subcommand>` | Manage external tool/capability registrations |
| `apxm ops <subcommand>` | Browse AIS operations |
| `apxm template <subcommand>` | Browse graph templates |
| `apxm explain <file>` | Explain what a graph does in human terms |
| `apxm codelet <subcommand>` | Compose graph fragments (merge) |

---

## Install

```bash
apxm install                # Full install (interactive)
apxm install --check        # Dry-run: report status without changes
apxm install --auto         # Automatic mode (no prompts)
apxm install --skip-deps    # Skip dependency checks
apxm install --skip-build   # Skip build step
```

**Stages:** Platform detection, dependency checks, conda environment, Rust toolchain, wrapper generation.

Long-running output (conda, cargo) is captured to `.apxm/install.log` — on failure the last ~30 lines are printed automatically.

---

## Doctor

```bash
apxm doctor
```

Checks: platform, dependencies (Rust, Cargo, CMake, Ninja, Mamba/Conda, Git), conda environment (MLIR/LLVM 21), build status, credentials, CI environment. Exit code `0` = all passed, `1` = failures found.

---

## Activate

```bash
eval "$(apxm activate)"     # Set up MLIR/LLVM env vars in current shell
apxm activate               # Print exports (inspect without applying)
```

---

## Compile

```bash
apxm compile workflow.json -o workflow.apxmobj
apxm compile workflow.json -o workflow.apxmobj -O2
apxm compile workflow.json -o workflow.apxmobj --emit-diagnostics diag.json
apxm compile workflow.json -o workflow.apxmobj --dump-ir  # Also dump MLIR IR
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

Runtime control:

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

## Validate

```bash
apxm validate graph.json              # Human-readable validation report
apxm validate graph.json --json       # Machine-readable JSON output
```

---

## Analyze

```bash
apxm analyze graph.json               # Parallelism analysis (phases, speedup estimate)
apxm analyze graph.json --json        # Full analysis as JSON
```

---

## Ops

```bash
apxm ops list                         # All operations, grouped by category
apxm ops list --category reasoning    # Filter by category
apxm ops show ASK                     # Detailed info + example JSON for an op
```

---

## Template

```bash
apxm template list                    # List available starter templates
apxm template show fan-out            # Display template details
apxm template show fan-out --json     # Emit graph JSON ready to use
```

---

## Explain

```bash
apxm explain graph.json               # Human-readable explanation of what a graph does
```

---

## Codelet

```bash
apxm codelet merge a.json b.json -o combined.json   # Merge graph fragments
```

---

## Tools

```bash
apxm tools list                       # List registered external tools
apxm tools add <name> --endpoint <url>  # Register an external tool/capability
apxm tools remove <name>              # Unregister a tool
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
cargo build -p apxm-compiler --release
```

---

## Troubleshooting

**"apxm: command not found"** — Restart shell (`source ~/.bashrc`) or check `ls ~/.local/bin/apxm`.

**"MLIR toolchain not detected" / "Library not loaded"** — Re-run `apxm install` to regenerate the wrapper.

**Install failures** — Check `.apxm/install.log` for full output.

**Credential issues** — Run `apxm register test` to verify API connectivity.
