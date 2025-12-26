# Testing Matrix

## Fast Checks

Run these during development:

```bash
cargo test -p apxm-core
cargo test -p apxm-artifact
cargo test -p apxm-ais
cargo test -p apxm-runtime
cargo test -p apxm-driver
```

## Full Checks

Run these before release or major refactors:

```bash
cargo test --workspace
```

## Compiler Tests (Requires MLIR Toolchain)

```bash
cargo test -p apxm-compiler
```

## Metrics Builds

```bash
cargo test -p apxm-runtime --features metrics
cargo test -p apxm-driver --features metrics
```
