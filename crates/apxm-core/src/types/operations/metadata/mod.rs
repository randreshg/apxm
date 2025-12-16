//! Operation metadata module.
//!
//! Contains metadata definitions for AIS operations used for code generation and analysis.

mod registry;
mod validation;

pub use registry::{
    OPERATION_REGISTRY, OperationField, OperationMetadata, find_operation, operations,
};
pub use validation::validate_operation;
