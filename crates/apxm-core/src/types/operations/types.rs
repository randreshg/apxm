//! AIS operation type definitions.
//!
//! Contains the AISOperationType enum that represents all possible operation types.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents all possible AIS operation types.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AISOperationType {
    /// Invoke a capability (tool/function call).
    Inv,
    /// Reasoning operation (LLM call for reasoning).
    Rsn,
    /// Query memory (read from memory system).
    QMem,
    /// Update memory (write to memory system).
    UMem,
    /// Planning operation (generate a plan using LLM).
    Plan,
    /// Wait for all input tokens to be ready.
    WaitAll,
    /// Merge multiple tokens into one.
    Merge,
    /// Memory fence (synchronization barrier).
    Fence,
    /// Exception handling.
    Exc,
    /// Communication between agents.
    Communicate,
    /// Reflection operation.
    Reflect,
    /// Verification operation.
    Verify,
    /// Error operation.
    Err,
    /// Return from function.
    Return,
    /// Unconditional jump.
    Jump,
    /// Branch based on value.
    BranchOnValue,
    /// Loop start marker.
    LoopStart,
    /// Loop end marker.
    LoopEnd,
    /// Try-catch exception handling.
    TryCatch,
    /// String constant.
    ConstStr,
}

impl fmt::Display for AISOperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AISOperationType::Inv => write!(f, "INV"),
            AISOperationType::Rsn => write!(f, "RSN"),
            AISOperationType::QMem => write!(f, "QMEM"),
            AISOperationType::UMem => write!(f, "UMEM"),
            AISOperationType::Plan => write!(f, "PLAN"),
            AISOperationType::WaitAll => write!(f, "WAIT_ALL"),
            AISOperationType::Merge => write!(f, "MERGE"),
            AISOperationType::Fence => write!(f, "FENCE"),
            AISOperationType::Exc => write!(f, "EXC"),
            AISOperationType::Communicate => write!(f, "COMMUNICATE"),
            AISOperationType::Reflect => write!(f, "REFLECT"),
            AISOperationType::Verify => write!(f, "VERIFY"),
            AISOperationType::Err => write!(f, "ERR"),
            AISOperationType::Return => write!(f, "RETURN"),
            AISOperationType::Jump => write!(f, "JUMP"),
            AISOperationType::BranchOnValue => write!(f, "BRANCH_ON_VALUE"),
            AISOperationType::LoopStart => write!(f, "LOOP_START"),
            AISOperationType::LoopEnd => write!(f, "LOOP_END"),
            AISOperationType::TryCatch => write!(f, "TRY_CATCH"),
            AISOperationType::ConstStr => write!(f, "CONST_STR"),
        }
    }
}

impl AISOperationType {
    /// Returns the metadata name for this operation type.
    pub fn metadata_name(&self) -> &'static str {
        match self {
            AISOperationType::Inv => "Inv",
            AISOperationType::Rsn => "Rsn",
            AISOperationType::QMem => "QMem",
            AISOperationType::UMem => "UMem",
            AISOperationType::Plan => "Plan",
            AISOperationType::WaitAll => "WaitAll",
            AISOperationType::Merge => "Merge",
            AISOperationType::Fence => "Fence",
            AISOperationType::Exc => "Exc",
            AISOperationType::Communicate => "Communicate",
            AISOperationType::Reflect => "Reflect",
            AISOperationType::Verify => "Verify",
            AISOperationType::Err => "Err",
            AISOperationType::Return => "Return",
            AISOperationType::Jump => "Jump",
            AISOperationType::BranchOnValue => "BranchOnValue",
            AISOperationType::LoopStart => "LoopStart",
            AISOperationType::LoopEnd => "LoopEnd",
            AISOperationType::TryCatch => "TryCatch",
            AISOperationType::ConstStr => "ConstStr",
        }
    }
}
