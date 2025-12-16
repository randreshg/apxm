//! AIS operation types module.
//!
//! Contains definitions for all AIS operation types and their metadata.

mod definition;
pub mod metadata;
mod types;

pub use definition::AISOperation;
pub use metadata::validate_operation;
pub use types::AISOperationType;
