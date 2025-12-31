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

### CLI (Recommended)

Add `apxm` to your PATH and use the CLI wrapper:

```bash
# Add to PATH (add to ~/.zshrc or ~/.bashrc for persistence)
export PATH="$PATH:$(pwd)/bin"
pip install typer rich

# Use the CLI
apxm doctor
apxm compiler build
apxm compiler run examples/hello_world.ais
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
