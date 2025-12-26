# Getting Started

This guide walks you through installing dependencies, setting up the MLIR toolchain, and building APXM.

## 1) Install Prerequisites

- Rust toolchain (nightly is pinned via `rust-toolchain.toml`)
- `mamba` (recommended) or `conda` (for the MLIR toolchain)

## 2) Create the Conda Environment

```bash
cd /path/to/apxm
cargo run -p apxm-cli -- install
```

This uses `mamba` to create/update the `apxm` environment from `environment.yaml`.

## 3) Activate the Environment

```bash
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"
```

## 4) Verify the Toolchain

```bash
cargo run -p apxm-cli -- doctor
```

You should see `mlir-tblgen` and `cmake/mlir` marked as OK.

## 5) Build the Workspace

Build everything

```bash
cargo build --workspace
```

## 6) Run Examples

```bash
# Compile/run via the CLI (requires MLIR toolchain)
cargo run -p apxm-cli --features driver -- run examples/hello_world.ais

# Runtime examples (no compiler required)
cargo run -p apxm-runtime --example substrate_demo
cargo run -p apxm-runtime --example ollama_llm_demo
```

## Troubleshooting

- If `doctor` reports missing MLIR tools, re-run:
  - `cargo run -p apxm-cli -- install`
  - `conda activate apxm`
  - `eval "$(cargo run -p apxm-cli -- activate)"`
- If MLIR is installed but the compiler fails to link, check that `MLIR_DIR` and `LLVM_DIR` are set to `$CONDA_PREFIX/lib/cmake/{mlir,llvm}`.
