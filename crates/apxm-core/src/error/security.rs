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
    fn test_policy_violation_error() {
        let error = SecurityError::PolicyViolation {
            policy: "rate_limit".to_string(),
            reason: "Too many requests per minute".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Policy violation"));
        assert!(display.contains("rate_limit"));
        assert!(display.contains("Too many requests per minute"));
    }

    #[test]
    fn test_rate_limit_exceeded_error() {
        let error = SecurityError::RateLimitExceeded {
            resource: "llm_api".to_string(),
            limit: None,
        };

        let display = format!("{}", error);
        assert!(display.contains("Rate limit exceeded"));
        assert!(display.contains("llm_api"));
    }

    #[test]
    fn test_rate_limit_exceeded_error_with_limit() {
        let error = SecurityError::RateLimitExceeded {
            resource: "llm_api".to_string(),
            limit: Some(100),
        };

        let display = format!("{}", error);
        assert!(display.contains("Rate limit exceeded"));
        assert!(display.contains("llm_api"));
        assert!(display.contains("100"));
    }

    #[test]
    fn test_sandbox_error() {
        let error = SecurityError::SandboxError {
            message: "Execution timeout in sandbox".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Sandbox error"));
        assert!(display.contains("Execution timeout in sandbox"));
    }

    #[test]
    fn test_unauthorized_error() {
        let error = SecurityError::Unauthorized {
            resource: "/api/secret".to_string(),
            reason: None,
        };

        let display = format!("{}", error);
        assert!(display.contains("Unauthorized access"));
        assert!(display.contains("/api/secret"));
    }

    #[test]
    fn test_unauthorized_error_with_reason() {
        let error = SecurityError::Unauthorized {
            resource: "/api/secret".to_string(),
            reason: Some("Missing authentication token".to_string()),
        };

        let display = format!("{}", error);
        assert!(display.contains("Unauthorized access"));
        assert!(display.contains("/api/secret"));
        assert!(display.contains("Missing authentication token"));
    }

    #[test]
    fn test_error_implements_std_error() {
        let error = SecurityError::PolicyViolation {
            policy: "test".to_string(),
            reason: "test".to_string(),
        };

        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &error;
    }
}
