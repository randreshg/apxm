//! Internal scheduler state types.
//!
//! These types are used internally by the scheduler for tracking operation
//! and token state. They are not part of the public API.

use std::time::Instant;

use crate::aam::effects::OperationEffects;
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
    /// Operation effect metadata
    pub effects: OperationEffects,
}

impl OpState {
    /// Create a new operation state in Pending status.
    pub fn new() -> Self {
        Self::new_with_effects(OperationEffects::new())
    }

    pub fn new_with_effects(effects: OperationEffects) -> Self {
        Self {
            status: OpStatus::Pending,
            retries: 0,
            last_error: None,
            started_at: None,
            finished_at: None,
            effects,
        }
    }

    pub fn effects(&self) -> &OperationEffects {
        &self.effects
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
///
/// Note: Token delegation (for switch/case sub-DAG splicing) is tracked
/// separately in `SchedulerState::delegated_tokens` to avoid per-token overhead.
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

/// Promise state for tracking flow call results.
///
/// When a flow call operation executes, it creates a promise token that
/// will be resolved when the sub-flow completes.
#[derive(Debug, Clone)]
pub struct PromiseState {
    /// Target agent name.
    pub target_agent: String,
    /// Target flow name.
    pub target_flow: String,
    /// Time when the promise was created.
    pub created_at: Instant,
    /// Whether the promise has been resolved.
    pub resolved: bool,
    /// The resolved value (if resolved).
    pub value: Option<Value>,
}

impl PromiseState {
    /// Create a new unresolved promise.
    pub fn new(target_agent: String, target_flow: String) -> Self {
        Self {
            target_agent,
            target_flow,
            created_at: Instant::now(),
            resolved: false,
            value: None,
        }
    }
}

/// Execution frame for tracking recursive flow calls.
///
/// Each sub-flow execution pushes a frame onto the stack.
#[derive(Debug, Clone)]
pub struct ExecutionFrame {
    /// Unique execution ID for this frame.
    pub execution_id: String,
    /// The flow being executed.
    pub flow_name: String,
    /// The promise token that will be resolved when this flow completes.
    pub parent_promise: Option<apxm_core::types::TokenId>,
}
