# apxm-compiler

MLIR-based compiler for the Agent Instruction Set (AIS) dialect.

## Overview

`apxm-compiler` compiles AIS programs through an MLIR-based pipeline. It consists of:
- **Rust FFI layer** (`src/`) - Safe Rust wrappers around C++ MLIR code
- **C++/MLIR dialect** (`mlir/`) - AIS dialect definition, passes, and code generation

The compiler parses AIS DSL, transforms it through optimization passes, and generates executable artifacts.

## Responsibilities

- Parse AIS DSL or MLIR into an MLIR module
- Run the pass pipeline (normalize, schedule, fuse, lower)
- Emit artifacts or Rust source for execution

## How It Fits

The compiler produces artifacts consumed by `apxm-runtime` and is orchestrated
by `apxm-driver`. Operation metadata is shared via `apxm-ais`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      apxm-compiler                          │
├─────────────────────────────────────────────────────────────┤
│  src/                     │  mlir/                          │
│  ├── api/                 │  ├── include/ais/               │
│  │   ├── context.rs       │  │   ├── Dialect/AIS/IR/        │
│  │   ├── module.rs        │  │   ├── CAPI/                  │
│  │   └── pipeline.rs      │  │   └── Parser/                │
│  ├── codegen/             │  └── lib/                       │
│  │   ├── artifact.rs      │      ├── Dialect/AIS/           │
│  │   ├── emitter.rs       │      ├── CAPI/                  │
│  │   └── operations/      │      └── Parser/                │
│  └── ffi/                 │                                 │
│      └── bindings         │                                 │
└─────────────────────────────────────────────────────────────┘
```

## DSL Front End

The AIS DSL front end (lexer, parser, AST, MLIRGen) is documented in
`crates/apxm-compiler/dsl/README.md`.

## TableGen Source

Operation definitions are generated from `apxm-ais` into `.td` files that
back the AIS MLIR dialect. See `crates/apxm-ais/README.md`.

## Key Types

- `Context` - Compiler context managing MLIR state
- `Module` - Compiled module (can parse DSL or MLIR)
- `Pipeline` - Configurable pass pipeline
- `PassManager` - Pass registration and execution

## Usage

```rust
use apxm_compiler::{Context, Module};

// Initialize compiler
let context = Context::new()?;

// Parse DSL source
let source = r#"
    agent TestAgent {
        flow main {
            rsn "Analyze the input" -> result
        }
    }
"#;
let module = Module::parse_dsl(&context, source, "test.ais")?;

// Generate artifact
let artifact_bytes = module.generate_artifact_bytes()?;
```

## Passes

Available optimization passes:
- Scheduling annotation
- Reasoning fusion
- Graph normalization

```rust
use apxm_compiler::{list_passes, find_pass};

// List all available passes
for pass in list_passes() {
    println!("{}: {}", pass.name, pass.description);
}
```

## Build System

The build process (orchestrated by `build.rs`):

1. **Generate TableGen** - From `apxm-ais` Rust definitions
2. **Configure CMake** - With MLIR/LLVM from conda
3. **Build C++ library** - `libapxm_compiler_c.dylib`
4. **Generate bindings** - Using bindgen
5. **Link** - Emit rpath for LLVM libraries

## Dependencies

| Crate | Purpose |
|-------|---------|
| apxm-ais | Operation definitions (source of truth) |
| apxm-core | Error types and utilities |
| apxm-artifact | Artifact serialization |

## Requirements

- LLVM 21 / MLIR 21 (via conda: `environment.yaml`)
- CMake 3.20+
- C++17 compiler

## Building

```bash
# Ensure conda environment is active
cargo build -p apxm-compiler
```

## Testing

```bash
cargo test -p apxm-compiler
```

## Directory Structure

```
crates/apxm-compiler/
├── Cargo.toml
├── build.rs              # Build orchestrator
├── CMakeLists.txt        # Root CMake
├── mlir/                 # C++/MLIR code
│   ├── include/ais/      # Headers
│   └── lib/              # Implementation
└── src/                  # Rust FFI
    ├── api/              # Public API
    ├── codegen/          # Code generation
    ├── ffi/              # FFI bindings
    └── passes/           # Pass management
```
