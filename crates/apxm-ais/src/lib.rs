//! # Agent Instruction Set (AIS)
//!
//! This crate provides the **single source of truth** for all AIS operation definitions.
//! Both the compiler and runtime depend on this crate to ensure consistent operation
//! semantics across the entire system.
//!
//! ## Architecture
//!
//! ```text
//!                     ┌─────────────┐
//!                     │   apxm-ais  │  ← Single Source of Truth
//!                     │  (21 ops)   │
//!                     └──────┬──────┘
//!                            │
//!               ┌────────────┼────────────┐
//!               ▼            │            ▼
//!       ┌───────────────┐    │    ┌───────────────┐
//!       │ apxm-compiler │    │    │ apxm-runtime  │
//!       └───────────────┘    │    └───────────────┘
//! ```
//!
//! ## Operations (21 total)
//!
//! | Category | Operations |
//! |----------|------------|
//! | Metadata | AGENT |
//! | Memory | QMEM, UMEM |
//! | Reasoning | RSN, PLAN, REFLECT, VERIFY |
//! | Tools | INV, EXC |
//! | Control Flow | JUMP, BRANCH_ON_VALUE, LOOP_START, LOOP_END, RETURN |
//! | Synchronization | MERGE, FENCE, WAIT_ALL |
//! | Error Handling | TRY_CATCH, ERR |
//! | Communication | COMMUNICATE |
//! | Internal | CONST_STR |

pub mod aam;
pub mod memory;
pub mod operations;
pub mod types;
pub mod validation;

// Re-export commonly used types
pub use aam::{AAM, Beliefs, Capabilities, Goals};
pub use memory::MemoryTier;
pub use operations::tablegen::generate_tablegen;
pub use operations::{
    AGENT, AIS_OPERATIONS, AISOperationType, BRANCH_ON_VALUE, COMMUNICATE, CONST_STR, ERR, EXC,
    FENCE, INTERNAL_OPERATIONS, INV, JUMP, LOOP_END, LOOP_START, MERGE, METADATA_OPERATIONS,
    OperationCategory, OperationEmit, OperationField, OperationMetadata, OperationSpec, PLAN, QMEM,
    REFLECT, RETURN, RSN, TRY_CATCH, UMEM, VERIFY, WAIT_ALL, find_operation_by_mnemonic,
    find_operation_by_name, get_all_operations, get_operation_metadata, get_operation_spec,
    get_public_operations,
};
pub use types::Value;
pub use validation::{
    ValidationError, has_required_fields, missing_required_fields, validate_operation,
    validate_operation_strict,
};
