//! Internal scheduler state types.
//!
//! These types are used internally by the scheduler for tracking operation
//! and token state. They are not part of the public API.

use std::time::Instant;

use apxm_core::types::{NodeId, OpStatus, Value};

/// Internal operation state tracked during execution.
///
/// This is used by the scheduler to track the status of each operation
/// as it moves through the execution pipeline.
#[derive(Debug)]
pub(crate) struct OpState {
    /// Current execution status.
    pub status: OpStatus,
    /// Number of retry attempts made.
    pub retries: u32,
    /// Last error message (if any).
    pub last_error: Option<String>,
    /// Time when execution started.
    pub started_at: Option<Instant>,
    /// Time when execution finished.
    pub finished_at: Option<Instant>,
}

impl OpState {
    /// Create a new operation state in Pending status.
    pub fn new() -> Self {
        Self {
            status: OpStatus::Pending,
            retries: 0,
            last_error: None,
            started_at: None,
            finished_at: None,
        }
    }
}

impl Default for OpState {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal token state tracked during execution.
///
/// Tokens represent data dependencies in the dataflow graph. This type
/// tracks whether a token is ready and what value it holds.
#[derive(Debug)]
pub(crate) struct TokenState {
    /// Whether the token value is ready.
    pub ready: bool,
    /// The token's value (if ready).
    pub value: Option<Value>,
    /// List of downstream consumers waiting for this token.
    pub consumers: Vec<NodeId>,
}

impl TokenState {
    /// Create a new token state (not ready).
    pub fn new() -> Self {
        Self {
            ready: false,
            value: None,
            consumers: Vec::new(),
        }
    }
}

impl Default for TokenState {
    fn default() -> Self {
        Self::new()
    }
}
