# apxm-ais

Canonical AIS operation definitions shared by compiler and runtime.

## Overview

`apxm-ais` defines all AIS operations and their metadata. It is the single source
of truth used to generate MLIR TableGen for the compiler and Rust metadata for
the runtime dispatcher.

## Responsibilities

- Define all AIS operations and metadata
- Generate TableGen `.td` files for the MLIR compiler
- Provide Rust metadata for validation and dispatch

## How It Fits

The compiler consumes TableGen generated from this crate, while the runtime and
driver consume the Rust metadata. This keeps operation semantics consistent
across the toolchain.

## Operations (21 total)

| Category | Operations |
|----------|------------|
| **Metadata** | Agent |
| **Memory** | QMem, UMem |
| **Reasoning** | Rsn, Plan, Reflect, Verify |
| **Tools** | Inv, Exc |
| **Control Flow** | Jump, BranchOnValue, LoopStart, LoopEnd, Return |
| **Synchronization** | Merge, Fence, WaitAll |
| **Error Handling** | TryCatch, Err |
| **Communication** | Communicate |
| **Internal** | ConstStr |

## Usage

```rust
use apxm_ais::{
    AISOperationType, OperationMetadata, get_operation_metadata, generate_tablegen,
};

let meta: &OperationMetadata = get_operation_metadata(AISOperationType::Rsn);
println!("Operation: {}", meta.name);

let tablegen = generate_tablegen();
println!("TableGen bytes: {}", tablegen.len());
```

## Dependencies

This crate has no internal APXM dependencies.

## Building

```bash
cargo build -p apxm-ais
```

## Testing

```bash
cargo test -p apxm-ais
```
