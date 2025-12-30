//! Operation metadata module.
//!
//! Re-exports from apxm-ais (single source of truth).

mod registry;

// Re-export everything from apxm-ais operations
pub use registry::{
    AIS_OPERATIONS, AISOperationType, INTERNAL_OPERATIONS, OperationCategory, OperationField,
    OperationSpec, find_operation_by_name, get_all_operations, get_operation_spec,
    get_public_operations,
};

// Re-export validation from apxm-ais
pub use apxm_ais::validation::{
    ValidationError, has_required_fields, missing_required_fields, validate_operation,
    validate_operation_strict,
};
