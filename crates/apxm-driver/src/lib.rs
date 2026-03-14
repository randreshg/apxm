//! APxM Driver - Configuration, linker, and orchestration
//!
//! This crate consolidates configuration and orchestration:
//!
//! - **`config`**: TOML-based configuration (`~/.apxm/config.toml`)
//! - **`linker`**: High-level linker orchestrating compiler/runtime
//! - **`compiler`**: Compiler wrapper for ApxmGraph parsing and lowering
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

pub mod cache;
pub mod compiler;
pub mod config;
pub mod error;
pub mod linker;
pub mod runtime;

// --- Config ---
pub use config::{ApXmConfig, ChatConfig, ConfigError, LlmBackendConfig, ToolConfig};

// --- Linker ---
pub use linker::{LinkResult, Linker, LinkerConfig};

// --- Error ---
pub use error::DriverError;

// --- Runtime ---
pub use runtime::RuntimeExecutor;
