# apxm-core

Core types and utilities for APXM.

## Overview

`apxm-core` provides foundational types used across all APXM crates:
- **Error types** - Structured error handling
- **Execution types** - Agent, Flow, DAG, Node, Edge, Token
- **Compiler types** - Pass metadata, optimization levels
- **Utilities** - Logging, paths, build helpers

## Responsibilities

- Share core types across compiler, driver, runtime
- Centralize structured error handling
- Provide build helpers for native toolchains

## How It Fits

Every crate depends on `apxm-core` for shared execution types and error
definitions so the compiler, driver, and runtime speak the same language.

## Key Types

### Execution Types

```rust
use apxm_core::types::execution::{Agent, AgentFlow, ExecutionDag};

let mut dag = ExecutionDag::new();
dag.metadata.name = Some("Research.main".to_string());
dag.metadata.is_entry = true;

let main_flow = AgentFlow {
    name: "main".to_string(),
    is_entry: true,
    parameters: vec![],
    codelet_dag: None,
    execution_dag: dag,
};

let agent = Agent::new("Research").add_flow(main_flow);
assert!(agent.entry_flow().is_some());
```

### Error Types

```rust
use apxm_core::error::{CompilerError, RuntimeError};
use apxm_core::error::builder::ErrorBuilder;

// Build structured errors
let error = ErrorBuilder::parse("syntax error")
    .at_line(42)
    .with_suggestion("Check semicolons")
    .build();
```

### Compiler Types

```rust
use apxm_core::types::compiler::{PassInfo, PassCategory, OptimizationLevel};

let pass = PassInfo {
    name: "scheduling".into(),
    category: PassCategory::Analysis,
    description: "Annotate operations for scheduling".into(),
};
```

## Module Structure

```
apxm-core/src/
├── error/
│   ├── cli.rs          # Tooling errors
│   ├── compiler.rs     # Compiler errors
│   ├── runtime.rs      # Runtime errors
│   ├── builder.rs      # Error builder pattern
│   └── codes.rs        # Error codes
├── types/
│   ├── execution/      # Agent, Flow, DAG, Node, Edge
│   │   ├── agent.rs
│   ├── values/         # Token, Value
│   ├── compiler/       # Pass metadata
│   ├── operations/     # AIS operation types
│   └── session/        # Session management
├── logging.rs          # Logging macros
├── paths.rs            # Path utilities
└── utils/
    └── build.rs        # Build script helpers
```

## Build Utilities

For crates with native dependencies:

```rust
use apxm_core::utils::build::{LibraryConfig, locate_library, Platform};

// Locate MLIR installation
let config = LibraryConfig::for_mlir();
let link_spec = locate_library(&config)?;

// Emit cargo link directives
emit_link_directives(&link_spec, install_dir)?;
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| apxm-ais | Re-exports (backward compatibility) |

## Building

```bash
cargo build -p apxm-core
```

## Testing

```bash
cargo test -p apxm-core
```
