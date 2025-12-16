//! APXM Compiler - Compilation system for AI Operations
//!
//! This crate compiles AI operations into executable code. It provides APIs for
//! parsing, transforming, and generating code from AI operation definitions.
//!
//! # Overview
//!
//! The compiler processes AI operations through several stages:
//! - Parses AI operation definitions
//! - Builds intermediate representations
//! - Applies optimization passes
//! - Generates executable code (currently Rust backend)
//!
//! # Components
//!
//! - [`api`]: Interfaces for compiler interaction
//! - [`passes`]: Optimization and transformation passes
//! - [`codegen`]: Code generation backends

pub mod api;
pub mod codegen;
mod ffi;
pub mod passes;

pub use api::{Context, Module, Pipeline};
pub use codegen::{OperationEmitter, RustBackend, RustEmitter};
pub use passes::{PassManager, find_pass, get_pass_count, get_pass_info, list_passes};

pub use apxm_core::error::compiler::{CompilerError, Result};
pub use apxm_core::types::compiler::{PassCategory, PassInfo};
pub use apxm_core::types::{CodegenOptions, OptimizationLevel, PipelineConfig};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
