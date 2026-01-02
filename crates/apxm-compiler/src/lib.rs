//! APXM Compiler - Compilation system for AI Operations
//!
//! This crate compiles AI operations into executable artifacts. It provides APIs for
//! parsing, transforming, and generating binary artifacts from AI operation definitions.
//!
//! # Overview
//!
//! The compiler processes AI operations through several stages:
//! - Parses AI operation definitions (AIS DSL)
//! - Builds intermediate representations (AIS MLIR dialect)
//! - Applies optimization passes
//! - Generates binary artifacts for runtime execution
//!
//! # Components
//!
//! - [`api`]: Interfaces for compiler interaction
//! - [`passes`]: Optimization and transformation passes
//! - [`codegen`]: Artifact generation

pub mod api;
pub mod codegen;
mod ffi;
pub mod passes;

pub use api::{Context, Module, Pipeline};
pub use passes::{PassManager, find_pass, get_pass_count, get_pass_info, list_passes};

pub use apxm_core::error::compiler::{CompilerError, Result};
pub use apxm_core::types::compiler::{PassCategory, PassInfo};
pub use apxm_core::types::{CodegenOptions, OptimizationLevel, PipelineConfig};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
