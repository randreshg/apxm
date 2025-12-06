//! APXM Core - Fundamental types and traits for the APXM system.
//!
//! This crate provides the foundational types, data structures, and traits
//! that all other APXM components depend on.

pub mod error;
pub mod types;

pub use error::{
    common::{ErrorContext, ErrorContextExt, OpId, SourceLocation, TraceId},
    compile::CompileError,
    runtime::RuntimeError,
    security::SecurityError,
};
pub use types::{
    AISOperationType, DependencyType, Edge, Node, NodeId, NodeMetadata, Number, Token, TokenId,
    TokenStatus, Value,
};

#[cfg(test)]
mod tests {
    use super::{
        AISOperationType, Number, Token, Value,
        error::{
            common::SourceLocation, compile::CompileError, runtime::RuntimeError,
            security::SecurityError,
        },
    };
    use std::time::Duration;

    #[test]
    fn test_types_export() {
        let num = Number::Integer(42);
        assert!(matches!(num, Number::Integer(42)));

        let val = Value::from(true);
        assert!(matches!(val, Value::Bool(true)));

        let token = Token::new(1);
        assert_eq!(token.id, 1);

        let op = AISOperationType::Inv;
        assert_eq!(op.to_string(), "INV");
    }

    #[test]
    fn test_compile_error_export() {
        let location = SourceLocation::new("test.ais".to_string(), 3, 7);
        let err = CompileError::Parse {
            location: location.clone(),
            message: "Unexpected token".to_string(),
        };

        let display = err.to_string();
        assert!(display.contains("Parse error"));
        assert!(display.contains(&location.to_string()));
        assert!(display.contains("Unexpected token"));
    }

    #[test]
    fn test_runtime_error_from_security_error() {
        let security = SecurityError::Unauthorized {
            resource: "/secret".to_string(),
            reason: Some("Missing token".to_string()),
        };

        let runtime: RuntimeError = security.into();
        let display = runtime.to_string();
        assert!(matches!(runtime, RuntimeError::Security(_)));
        assert!(display.contains("Security error"));
        assert!(display.contains("Missing token"));
    }

    #[test]
    fn test_runtime_timeout_error_display() {
        let err = RuntimeError::Timeout {
            op_id: 7,
            timeout: Duration::from_secs(5),
        };

        let display = err.to_string();
        assert!(display.contains("Timeout"));
        assert!(display.contains("7"));
        assert!(display.contains("5s"));
    }
}
