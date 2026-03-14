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
//!                     │  (32 ops)   │
//!                     └──────┬──────┘
//!                            │
//!               ┌────────────┼────────────┐
//!               ▼            │            ▼
//!       ┌───────────────┐    │    ┌───────────────┐
//!       │ apxm-compiler │    │    │ apxm-runtime  │
//!       └───────────────┘    │    └───────────────┘
//! ```
//!
//! ## Operations (32 total)
//!
//! | Category | Operations |
//! |----------|------------|
//! | Metadata | AGENT |
//! | Memory | QMEM, UMEM |
//! | LLM/Reasoning | ASK, THINK, REASON, PLAN, REFLECT, VERIFY |
//! | Tools | INV, EXC, PRINT |
//! | Control Flow | JUMP, BRANCH_ON_VALUE, LOOP_START, LOOP_END, RETURN, SWITCH, FLOW_CALL |
//! | Synchronization | MERGE, FENCE, WAIT_ALL |
//! | Error Handling | TRY_CATCH, ERR |
//! | Communication | COMMUNICATE |
//! | Goal/State | UPDATE_GOAL, GUARD, CLAIM, PAUSE, RESUME |
//! | Internal | CONST_STR, YIELD |

pub mod aam;
pub mod memory;
pub mod operations;
pub mod passes;
pub mod types;
pub mod validation;

// Re-export commonly used types
pub use aam::{AAM, Beliefs, Capabilities, Goals};
pub use memory::MemoryTier;
pub use operations::tablegen::generate_tablegen;
pub use operations::{
    AIS_OPERATIONS, AISOperationType, OperationCategory, OperationField, OperationLatency,
    OperationSpec, get_all_operations, get_operation_spec,
};
pub use types::Value;
pub use validation::{
    ValidationError, has_required_fields, missing_required_fields, validate_operation,
    validate_operation_strict,
};

// Re-export pass generation functions (used by build.rs)
pub use passes::{generate_pass_descriptors, generate_pass_dispatch, generate_passes_tablegen};
