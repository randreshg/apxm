//! APXM Code Generation
//!
//! This module provides artifact generation from AIS MLIR to binary format.
//!
//! # Components
//!
//! - [`artifact`]: Binary artifact serialization/deserialization
//! - [`CodegenOptions`]: Code generation configuration

pub mod artifact;

pub use apxm_core::types::CodegenOptions;
