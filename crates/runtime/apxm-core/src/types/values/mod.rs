//! Core value types module.
//!
//! Re-exports fundamental data types from apxm-ais and adds runtime-specific types.

mod token;

// Re-export core types from apxm-ais (single source of truth)
pub use apxm_ais::types::{Number, TokenId, Value};

// Runtime-specific types
pub use token::{Token, TokenStatus};
