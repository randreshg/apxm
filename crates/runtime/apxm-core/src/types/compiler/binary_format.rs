//! Binary format for Agent serialization.
//!
//! This module defines the schema for the binary representation of agents,
//! which serves as the contract between the compiler and the runtime.
//!
//! # Design Notes
//!
//! IDs are `u64` instead of `Uuid` for:
//! - Direct compatibility with LLVM/MLIR (no custom UUID generation in C++)
//! - Simpler serialization (bincode native support)
//! - Sufficient uniqueness within a single agent (2^64 possible values)

use serde::{Deserialize, Serialize};

/// Unique identifier for operations and tokens.
///
/// Using `u64` for LLVM/MLIR compatibility. IDs are unique within a single
/// `AgentBinary` but not globally unique across agents.
pub type OpId = u64;

/// Represents a complete Agent binary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentBinary {
    /// Name of the agent.
    pub name: String,
    /// Version of the binary format.
    pub version: u32,
    /// List of operations in the agent.
    pub ops: Vec<OpDef>,
    /// List of entry tokens (initial available tokens).
    pub entry_tokens: Vec<OpId>,
}

impl AgentBinary {
    /// Creates a new AgentBinary.
    pub fn new(name: String, ops: Vec<OpDef>, entry_tokens: Vec<OpId>) -> Self {
        Self {
            name,
            version: 1,
            ops,
            entry_tokens,
        }
    }
}

/// Definition of a single operation in the binary format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpDef {
    /// Unique identifier for the operation.
    pub id: OpId,
    /// Type of the operation.
    pub op_type: OpType,
    /// Input token IDs.
    pub inputs: Vec<OpId>,
    /// Output token IDs.
    pub outputs: Vec<OpId>,
    /// Serialized parameters (JSON).
    #[serde(with = "serde_bytes")]
    pub params: Vec<u8>,
}

/// Operation types supported in the binary format.
///
/// This enum mirrors the AIS dialect operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OpType {
    Inv = 0,
    Rsn = 1,
    QMem = 2,
    UMem = 3,
    Plan = 4,
    WaitAll = 5,
    Merge = 6,
    Fence = 7,
    Exc = 8,
    Communicate = 9,
    Reflect = 10,
    Verify = 11,
    Err = 12,
    Return = 13,
    Jump = 14,
    BranchOnValue = 15,
    LoopStart = 16,
    LoopEnd = 17,
    TryCatch = 18,
}
