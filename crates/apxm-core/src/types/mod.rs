//! Fundamentals types for the APXM system.
//!
//! This module contains the core data structures used throughout APXM:
//! - `Number` : Numeric value representation
//! - `Value` : Unified value representation
//! - `Token` : Token-based dataflow
//! - `AISOperationType`: AIS operation types
//! - `Node`: Execution DAG node
//! - `Edge`: Execution DAG edge
//! - `ExecutionDAG`: Complete execution DAG

pub mod dag;
pub mod edge;
pub mod node;
pub mod number;
pub mod operation;
pub mod token;
pub mod value;

pub use dag::{DagMetadata, ExecutionDag};
pub use edge::{DependencyType, Edge};
pub use node::{Node, NodeId, NodeMetadata};
pub use number::Number;
pub use operation::{AISOperation, AISOperationType};
pub use token::{Token, TokenId, TokenStatus};
pub use value::Value;
