//! APXM Compiler API
//!
//! This module provides the public API for the APXM compiler.
//! It contains types for managing compilation state, modules, and pipelines.
//!
//! # Components
//!
//! - [`Context`]: Compiler state and configuration
//! - [`Module`]: Compilation units
//! - [`Pipeline`]: Compilation pass sequences

pub mod context;
pub mod module;
pub mod pipeline;

pub use context::Context;
pub use module::Module;
pub use pipeline::Pipeline;
