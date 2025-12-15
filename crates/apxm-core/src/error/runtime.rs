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
    fn test_scheduler_error() {
        let error = RuntimeError::Scheduler {
            message: "Deadlock detected".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Scheduler error"));
        assert!(display.contains("Deadlock detected"));
    }

    #[test]
    fn test_operation_error() {
        let error = RuntimeError::Operation {
            op_type: AISOperationType::Inv,
            message: "Capability not found".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Operation execution failed"));
        assert!(display.contains("INV"));
        assert!(display.contains("Capability not found"));
    }

    #[test]
    fn test_capability_error() {
        let error = RuntimeError::Capability {
            capability: "http_request".to_string(),
            message: "Connection refused".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Capability error"));
        assert!(display.contains("http_request"));
        assert!(display.contains("Connection refused"));
    }

    #[test]
    fn test_llm_error() {
        let error = RuntimeError::LLM {
            message: "API rate limit exceeded".to_string(),
            backend: None,
        };

        let display = format!("{}", error);
        assert!(display.contains("LLM error"));
        assert!(display.contains("API rate limit exceeded"));
    }

    #[test]
    fn test_llm_error_with_backend() {
        let error = RuntimeError::LLM {
            message: "Invalid API key".to_string(),
            backend: Some("openai".to_string()),
        };

        let display = format!("{}", error);
        assert!(display.contains("LLM error"));
        assert!(display.contains("openai"));
        assert!(display.contains("Invalid API key"));
    }

    #[test]
    fn test_memory_error() {
        let error = RuntimeError::Memory {
            message: "Memory space full".to_string(),
            space: None,
        };

        let display = format!("{}", error);
        assert!(display.contains("Memory error"));
        assert!(display.contains("Memory space full"));
    }

    #[test]
    fn test_memory_error_with_space() {
        let error = RuntimeError::Memory {
            message: "Query failed".to_string(),
            space: Some("LTM".to_string()),
        };

        let display = format!("{}", error);
        assert!(display.contains("Memory error"));
        assert!(display.contains("LTM"));
        assert!(display.contains("Query failed"));
    }

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

    #[test]
    fn test_timeout_error() {
        let error = RuntimeError::Timeout {
            op_id: 42,
            timeout: Duration::from_secs(30),
        };

        let display = format!("{}", error);
        assert!(display.contains("Timeout"));
        assert!(display.contains("42"));
        assert!(display.contains("30s"));
    }

    #[test]
    fn test_error_implements_std_error() {
        let error = RuntimeError::Scheduler {
            message: "Test".to_string(),
        };

        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &error;
    }
}
