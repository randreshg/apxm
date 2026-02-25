# apxm-driver

Orchestration layer bridging compiler and runtime.

## Overview

`apxm-driver` is the integration layer that:
- Manages configuration (TOML-based)
- Links compiler output to runtime
- Configures LLM backends
- Provides the `Linker` API for end-to-end execution

## Responsibilities

- Load config and materialize runtime/backends
- Compile ApxmGraph inputs via `apxm-compiler`
- Execute artifacts via `apxm-runtime`

## How It Fits

`apxm-driver` is the glue between `apxm-compiler` and `apxm-runtime`, and is the
recommended entry point for end-to-end compilation + execution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       apxm-driver                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │    Config    │  │    Linker    │  │  RuntimeExecutor │  │
│  │   (TOML)     │  │ (Compile+Run)│  │   (LLM setup)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐               │
│         ▼                 ▼                 ▼               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  Compiler   │   │  Artifact   │   │   Runtime   │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Key Types

- `ApXmConfig` - Main configuration struct
- `Linker` - Compiles and links programs
- `LinkerConfig` - Linker configuration
- `RuntimeExecutor` - Runtime with LLM backends configured
- `DriverError` - Error type for driver operations

## Usage

```rust
use apxm_driver::{ApXmConfig, Linker, LinkerConfig};
use std::path::Path;

// Load configuration
let config = ApXmConfig::from_file("apxm.toml")?;

// Create linker
let linker_config = LinkerConfig::from_apxm_config(config);
let linker = Linker::new(linker_config).await?;

// Compile and execute
let result = linker.run(Path::new("program.json")).await?;
println!("Result: {:?}", result);
```

## Configuration

Configuration via `apxm.toml`:

```toml
[chat]
providers = ["openai-gpt4"]

[[llm_backends]]
name = "openai-gpt4"
provider = "openai"
model = "gpt-4"
api_key = "env:OPENAI_API_KEY"

[[capabilities]]
name = "web_search"
enabled = true

[execution]
max_parallelism = 4
timeout_seconds = 300
```

## LLM Backend Configuration

```rust
use apxm_driver::{LlmBackendConfig, ApXmConfig};

let config = ApXmConfig {
    llm_backends: vec![
        LlmBackendConfig {
            name: "gpt4".into(),
            provider: Some("openai".into()),
            model: Some("gpt-4".into()),
            api_key: Some("env:OPENAI_API_KEY".into()),
            ..Default::default()
        },
    ],
    ..Default::default()
};
```

## Inner Plan Linking

The driver provides `InnerPlanLinker` for dynamic plan(compilation:)
```rust
use apxm_driver::runtime::CompilerInnerPlanLinker;

// Used by runtime to compile inner plans on-the-fly
let linker = CompilerInnerPlanLinker::new()?;
let dag = linker.link_inner_plan(graph_json, "inner_plan").await?;
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| apxm-compiler | Compilation |
| apxm-runtime | Execution |
| apxm-backends | LLM/storage providers |
| apxm-artifact | Artifact handling |
| apxm-core | Shared types |

## Building

```bash
cargo build -p apxm-driver
```

## Testing

```bash
cargo test -p apxm-driver
```

To enable metrics (compile/runtime timing and LLM usage):

```bash
cargo test -p apxm-driver --features metrics
```

## Module Structure

```
crates/apxm-driver/src/
├── lib.rs
├── error.rs            # DriverError
├── config/             # Configuration types
│   └── mod.rs          # ApXmConfig, LlmBackendConfig
├── linker/             # Linker implementation
│   └── mod.rs          # Linker, LinkerConfig
├── compiler/           # Compiler wrapper
│   └── mod.rs
└── runtime/            # Runtime executor
    ├── mod.rs          # RuntimeExecutor
    ├── llm.rs          # LLM registry setup
    └── inner_plan.rs   # InnerPlanLinker
```
