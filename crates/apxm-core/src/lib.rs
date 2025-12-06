//! APXM Core - Fundamental types and traits for the APXM system.
//!
//! This crate provides the foundational types, data structures, and traits
//! that all other APXM components depend on.

pub mod types;

pub use types::{
    AISOperationType, DependencyType, Edge, Node, NodeId, NodeMetadata, Number, Token, TokenId,
    TokenStatus, Value,
};

#[cfg(test)]
mod tests {
    use super::*;

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
}
