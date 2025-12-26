//! APxM Driver - Configuration, linker, and orchestration
//!
//! This crate consolidates configuration and orchestration:
//!
//! - **`config`**: TOML-based configuration (`~/.apxm/config.toml`)
//! - **`linker`**: High-level linker orchestrating compiler/runtime
//! - **`compiler`**: Compiler wrapper for DSL/MLIR parsing
//! - **`runtime`**: Runtime executor for DAG execution
//!
//! # Architecture
//!
//! ```text
//!                      ┌─────────────────┐
//!                      │   apxm-driver   │
//!                      └────────┬────────┘
//!                               │
//!        ┌──────────────────────┼──────────────────────┐
//!        ▼                      ▼                      ▼
//!   ┌──────────┐          ┌──────────┐          ┌──────────┐
//!   │  config  │          │  linker  │          │ runtime  │
//!   │  TOML    │          │Orchestr. │          │  Exec    │
//!   │ Parsing  │          │  Logic   │          │  DAGs    │
//!   └──────────┘          └──────────┘          └──────────┘
//! ```

pub mod compiler;
pub mod config;
pub mod error;
pub mod linker;
pub mod runtime;

// ═══════════════════════════════════════════════════════════════════════════
// Config Re-exports
// ═══════════════════════════════════════════════════════════════════════════

pub use config::{
    ApXmConfig, CapabilityConfig, ChatConfig, ConfigError, ExecPolicyConfig, LlmBackendConfig,
    ToolConfig,
};

// ═══════════════════════════════════════════════════════════════════════════
// Linker Re-exports
// ═══════════════════════════════════════════════════════════════════════════

pub use linker::{LinkResult, Linker, LinkerConfig};

// ═══════════════════════════════════════════════════════════════════════════
// Error Re-exports
// ═══════════════════════════════════════════════════════════════════════════

pub use error::DriverError;

// ═══════════════════════════════════════════════════════════════════════════
// Runtime Re-exports
// ═══════════════════════════════════════════════════════════════════════════

pub use runtime::RuntimeExecutor;
