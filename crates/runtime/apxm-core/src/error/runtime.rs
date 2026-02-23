//! Runtime errors.
//!
//! This module defines errors that occur during execution of the program,
//! including scheduler errors, operation executiong failures, and system errors.

use std::time::Duration;

use thiserror::Error;

use crate::error::common::OpId;
use crate::error::security::SecurityError;
use crate::types::AISOperationType;

/// Errors that occur during runtime execution.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Scheduler error: general.
    #[error("Scheduler error: {message}")]
    Scheduler {
        /// Error message describing the scheduler failure.
        message: String,
    },

    /// Scheduler error: missing token.
    #[error("Missing token {token_id} for node {node_id}")]
    SchedulerMissingToken {
        /// Node ID that needs the token.
        node_id: u64,
        /// Token ID that is missing.
        token_id: u64,
    },

    /// Scheduler error: duplicate producer.
    #[error("Duplicate producer for token {token_id}")]
    SchedulerDuplicateProducer {
        /// Token ID with duplicate producer.
        token_id: u64,
    },

    /// Scheduler error: deadlock detected.
    #[error("Deadlock detected after {timeout_ms}ms with {remaining} nodes remaining")]
    SchedulerDeadlock {
        /// Timeout in milliseconds.
        timeout_ms: u64,
        /// Number of remaining nodes.
        remaining: usize,
    },

    /// Scheduler error: execution cancelled.
    #[error("Execution cancelled")]
    SchedulerCancelled,

    /// Scheduler error: retry exhausted.
    #[error("Node {node_id} failed after retries: {reason}")]
    SchedulerRetryExhausted {
        /// Node ID that failed.
        node_id: u64,
        /// Reason for failure.
        reason: String,
    },

    /// Operation execution error.
    #[error("Operation execution failed: {op_type} - {message}")]
    Operation {
        /// Type of operation that failed.
        op_type: AISOperationType,
        /// Error message describing the operation failure.
        message: String,
    },

    /// Capability invocation error.
    #[error("Capability error: {capability} - {message}")]
    Capability {
        /// Name of the capability that failed.
        capability: String,
        /// Error message describing the capability failure.
        message: String,
    },

    /// LLM backend error
    #[error(
        "LLM error{backend}: {message}",
        backend = Self::backend_suffix(.backend)
    )]
    LLM {
        /// Error message describing the LLM failure.
        message: String,
        /// Optional backend identifier.
        backend: Option<String>,
    },

    /// Memory system error.
    #[error(
        "Memory error{space}: {message}",
        space = Self::memory_space_suffix(.space)
    )]
    Memory {
        /// Error message describing the memory failure.
        message: String,
        /// Optional memory space identifier.
        space: Option<String>,
    },

    /// Security error (wraps SecurityError).
    #[error("Security error: {0}")]
    Security(#[from] SecurityError),

    /// Timeout error.
    #[error("Timeout: operation {op_id} exceeded timeout {timeout:?}")]
    Timeout {
        /// Operation ID that timed out.
        op_id: OpId,
        /// Time duration that was exceeded.
        timeout: Duration,
    },

    /// Serialization/Deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Executor error.
    #[error("Executor error: {0}")]
    Executor(String),

    /// State error.
    #[error("State error: {0}")]
    State(String),
}

impl RuntimeError {
    fn backend_suffix(backend: &Option<String>) -> String {
        match backend {
            Some(b) => format!(" (backend: {})", b),
            None => String::new(),
        }
    }

    fn memory_space_suffix(space: &Option<String>) -> String {
        match space {
            Some(s) => format!(" (space: {})", s),
            None => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_error_conversion() {
        let sec_error = SecurityError::PolicyViolation {
            policy: "rate_limit".to_string(),
            reason: "Too many requests".to_string(),
        };

        let runtime_error: RuntimeError = sec_error.into();
        let display = format!("{}", runtime_error);
        assert!(display.contains("Security error"));
        assert!(display.contains("Policy violation"));
    }
}
