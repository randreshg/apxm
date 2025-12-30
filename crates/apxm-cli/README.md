# apxm-cli

Minimal CLI wrapper for compile/run workflows.

## Overview

`apxm-cli` is a thin wrapper around `apxm-driver` that exposes:
- `compile` — compile DSL/MLIR to an artifact
- `run` — compile + execute via the runtime
- `doctor` — verify MLIR/compiler dependencies
- `activate` — print shell exports for MLIR/LLVM env setup
- `install` — create/update the conda env from environment.yaml

## Responsibilities

- Provide a minimal, scriptable CLI for compile/run workflows
- Surface MLIR toolchain diagnostics (`doctor`)
- Bootstrap the MLIR toolchain with `install` + `activate`

## How It Fits

`apxm-cli` wraps `apxm-driver` for compile/run (via the `driver` feature) and
uses `apxm-core` diagnostics for environment checks.

## Usage

### Python CLI (Recommended)

The Python CLI (`tools/apxm_cli.py`) handles environment setup automatically:

```bash
python tools/apxm_cli.py doctor
python tools/apxm_cli.py compiler build
python tools/apxm_cli.py compiler run examples/hello_world.ais
```

See `docs/AGENTS.md` for the complete CLI reference.

### Binary Usage (After Building)

```bash
# Build the compiler
cargo build -p apxm-cli --features driver --release

# Use the compiled binary
./target/release/apxm run examples/hello_world.ais
./target/release/apxm compile examples/hello_world.ais -o output.apxmobj
./target/release/apxm doctor
```

### Environment Setup Commands

```bash
# Install/update conda environment
cargo run -p apxm-cli -- install

# Emit exports for your shell
eval "$(cargo run -p apxm-cli -- activate)"
eval "$(cargo run -p apxm-cli -- activate --shell fish)"
```

## Metrics (Optional)

```bash
cargo run -p apxm-cli --features metrics -- run examples/hello_world.ais
```

## Testing

```bash
cargo test -p apxm-cli
```
