//! AIS Operations - Single Source of Truth
//!
//! This module contains the complete specification for all 21 AIS operations
//! (19 public + 1 metadata + 1 internal). Both the compiler and runtime use
//! these definitions to ensure consistent semantics.
//!
//! The `tablegen` submodule generates MLIR TableGen files from these definitions,
//! enabling Rust to be the single source of truth for operation metadata.

mod category;
mod definitions;
mod metadata;
pub mod tablegen;

pub use category::OperationCategory;
pub use definitions::{
    find_operation_by_mnemonic, find_operation_by_name, get_all_operations, get_operation_spec,
    get_public_operations, AISOperationType, OperationField, OperationSpec, AIS_OPERATIONS,
    INTERNAL_OPERATIONS, METADATA_OPERATIONS,
};
pub use metadata::{
    get_operation_metadata, OperationEmit, OperationMetadata,
    // All static metadata instances
    AGENT, BRANCH_ON_VALUE, COMMUNICATE, CONST_STR, ERR, EXC, FENCE, INV, JUMP, LOOP_END,
    LOOP_START, MERGE, PLAN, QMEM, REFLECT, RETURN, RSN, TRY_CATCH, UMEM, VERIFY, WAIT_ALL,
};
