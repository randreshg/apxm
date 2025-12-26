//! AIS operation types module.
//!
//! Re-exports from apxm-ais (single source of truth) and adds runtime-specific definitions.

mod definition;
pub mod metadata;

// Re-export from metadata (which re-exports from apxm-ais)
pub use metadata::{
    validate_operation, AISOperationType, OperationCategory, OperationField, OperationSpec,
    ValidationError,
};

// Runtime-specific types
pub use definition::AISOperation;
