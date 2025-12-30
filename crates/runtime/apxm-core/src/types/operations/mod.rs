//! AIS operation types module.
//!
//! Re-exports from apxm-ais (single source of truth) and adds runtime-specific definitions.

mod definition;
pub mod metadata;

// Re-export from metadata (which re-exports from apxm-ais)
pub use metadata::{
    AISOperationType, OperationCategory, OperationField, OperationSpec, ValidationError,
    validate_operation,
};

// Runtime-specific types
pub use definition::AISOperation;
