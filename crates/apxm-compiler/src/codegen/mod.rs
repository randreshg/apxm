//! APXM Code Generation
//!
//! This module translates intermediate representations into executable code.
//! It provides backends for different target languages.
//!
//! # Components
//!
//! - [`RustBackend`]: Generates Rust code
//! - [`RustEmitter`]: Rust code emission
//! - [`OperationEmitter`]: Individual operation emission
//! - [`CodegenOptions`]: Code generation configuration

pub mod artifact;
pub mod backend;
pub mod emitter;
pub mod operations;

pub use apxm_core::types::CodegenOptions;
pub use backend::RustBackend;
pub use emitter::RustEmitter;
pub use operations::OperationEmitter;
