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

```bash
# Compile an AIS file
cargo run -p apxm-cli --features driver -- compile examples/hello_world.ais

# Run an AIS file
cargo run -p apxm-cli --features driver -- run examples/hello_world.ais

# MLIR input
cargo run -p apxm-cli --features driver -- run examples/pipeline.mlir --mlir

# Diagnostics
cargo run -p apxm-cli -- doctor

# Emit exports for your shell
eval "$(cargo run -p apxm-cli -- activate)"
eval "$(cargo run -p apxm-cli -- activate --shell fish)"

# Install/update env (mamba)
cargo run -p apxm-cli -- install
```

## Metrics (Optional)

```bash
cargo run -p apxm-cli --features metrics -- run examples/hello_world.ais
```

## Testing

```bash
cargo test -p apxm-cli
```
