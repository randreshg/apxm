//! Operation metadata registry.
//!
//! Re-exports from apxm-ais (single source of truth).

// Re-export everything from apxm-ais operations
pub use apxm_ais::operations::{
    AIS_OPERATIONS, AISOperationType, INTERNAL_OPERATIONS, OperationCategory, OperationField,
    OperationSpec, find_operation_by_name, get_all_operations, get_operation_spec,
    get_public_operations,
};
