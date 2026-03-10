//! Execution graph types module.
//!
//! Contains types for representing execution DAGs and their components.

mod agent;
mod codelet;
mod dag;
mod edge;
mod node;
mod status;

pub use agent::{
    Agent, AgentFlow, AgentId, AgentMetadata, CapabilityDeclaration, MemoryDeclaration,
};
pub use codelet::{Codelet, CodeletDag, CodeletId, CodeletMetadata};
pub use dag::{DagMetadata, ExecutionDag, FlowParameter};
pub use edge::{DependencyType, Edge};
pub use node::{Node, NodeId, NodeMetadata};
pub use status::{ExecutionStats, NodeStatus, OpStatus};
