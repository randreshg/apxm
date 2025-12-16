//! Core value types module.
//!
//! Contains fundamental data types for value representation.

mod number;
mod token;
mod value;

pub use number::Number;
pub use token::{Token, TokenId, TokenStatus};
pub use value::Value;
