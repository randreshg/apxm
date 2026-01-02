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
    AIS_OPERATIONS, AISOperationType, INTERNAL_OPERATIONS, METADATA_OPERATIONS, OperationField,
    OperationSpec, find_operation_by_mnemonic, find_operation_by_name, get_all_operations,
    get_operation_spec, get_public_operations,
};
pub use metadata::{
    // All static metadata instances
    AGENT,
    BRANCH_ON_VALUE,
    COMMUNICATE,
    CONST_STR,
    ERR,
    EXC,
    FENCE,
    INV,
    JUMP,
    LOOP_END,
    LOOP_START,
    MERGE,
    OperationEmit,
    OperationMetadata,
    PLAN,
    QMEM,
    REASON,
    REFLECT,
    RETURN,
    ASK,
    THINK,
    TRY_CATCH,
    UMEM,
    VERIFY,
    WAIT_ALL,
    get_operation_metadata,
};
