//! Security errors.
//!
//! This module defines errors related to security violations, policy enforcement,
//! rate limiting, and sandboxing.

use thiserror::Error;

/// Errors related to security and policy enforcement.
#[derive(Debug, Error)]
pub enum SecurityError {
    /// Policy violation error.
    #[error("Policy violation: policy '{policy}' - {reason}")]
    PolicyViolation {
        /// Name of the policy that was violated.
        policy: String,
        /// Reason for the policy violation.
        reason: String,
    },

    /// Rate limit exceeded error.
    #[error(
        "Rate limit exceeded: resource '{resource}'{}",
        Self::limit_suffix(.limit)
    )]
    RateLimitExceeded {
        /// Resource that exceeded the rate limit.
        resource: String,
        /// Optional limit value that was exceeded.
        limit: Option<u64>,
    },

    /// Sandbox execution error.
    #[error("Sandbox error: {message}")]
    SandboxError {
        /// Error message describing the sandbox failure.
        message: String,
    },

    /// Unauthorized access error.
    #[error(
        "Unauthorized access: resource '{resource}'{}",
        Self::reason_suffix(.reason)
    )]
    Unauthorized {
        /// Resource that was accessed without authorization.
        resource: String,
        /// Optional reason for the unauthorized access.
        reason: Option<String>,
    },
}

impl SecurityError {
    fn limit_suffix(limit: &Option<u64>) -> String {
        match limit {
            Some(value) => format!(" (limit: {})", value),
            None => String::new(),
        }
    }

    fn reason_suffix(reason: &Option<String>) -> String {
        match reason.as_ref() {
            Some(r) => format!(" - {}", r),
            None => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_implements_std_error() {
        let error = SecurityError::PolicyViolation {
            policy: "test".to_string(),
            reason: "test".to_string(),
        };

        // Verify it implements std::error::Error (compile guard)
        let _: &dyn std::error::Error = &error;
    }
}
