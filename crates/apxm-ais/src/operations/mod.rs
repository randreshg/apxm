//! AIS Operations - Single Source of Truth
//!
//! This module contains the complete specification for all 31 AIS operations
//! (28 public + 1 metadata + 2 internal). Both the compiler and runtime use
//! these definitions to ensure consistent semantics.
//!
//! The `tablegen` submodule generates MLIR TableGen files from these definitions,
//! enabling Rust to be the single source of truth for operation metadata.

mod category;
mod definitions;
pub mod tablegen;

pub use category::OperationCategory;
pub use definitions::{
    AIS_OPERATIONS, AISOperationType, OperationField, OperationSpec, get_all_operations,
    get_operation_spec,
};
