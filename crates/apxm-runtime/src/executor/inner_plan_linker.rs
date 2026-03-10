//! Inner plan linker interface for runtime
//!
//! This module provides the interface for linking inner plan graph payloads
//! during runtime execution. The linker acts as a bridge between the
//! runtime and the compiler, delegating parsing/validation to the compiler.

use apxm_core::{
    error::RuntimeError,
    types::execution::{CodeletDag, ExecutionDag},
};
use async_trait::async_trait;

/// Result type for inner plan linking
pub type LinkResult = Result<ExecutionDag, RuntimeError>;

/// Trait for linking inner plan graph payloads into ExecutionDAGs
///
/// The linker bridges the runtime and compiler:
/// - Runtime calls linker with graph JSON payload
/// - Linker delegates to compiler for parsing/validation
/// - Linker returns validated DAG to runtime
#[async_trait]
pub trait InnerPlanLinker: Send + Sync {
    /// Link inner plan graph payload into an ExecutionDAG
    ///
    /// # Arguments
    ///
    /// * `graph_payload` - The ApxmGraph JSON payload from the LLM
    /// * `source_name` - Name for error reporting (e.g., "inner_plan_<execution_id>")
    ///
    /// # Returns
    ///
    /// A validated ExecutionDAG ready for splicing
    ///
    /// # Errors
    ///
    /// Returns RuntimeError::Compiler if parsing/validation fails
    async fn link_inner_plan(&self, graph_payload: &str, source_name: &str) -> LinkResult;

    /// Link a structured inner-plan codelet DAG into an ExecutionDAG.
    async fn link_codelet_dag(&self, dag: CodeletDag) -> LinkResult;
}

/// No-op linker for contexts that don't support inner plan linking
pub struct NoOpLinker;

#[async_trait]
impl InnerPlanLinker for NoOpLinker {
    async fn link_inner_plan(&self, _graph_payload: &str, _source_name: &str) -> LinkResult {
        Err(RuntimeError::State(
            "Inner plan linking not supported in this context".to_string(),
        ))
    }

    async fn link_codelet_dag(&self, _dag: CodeletDag) -> LinkResult {
        Err(RuntimeError::State(
            "Inner plan linking not supported in this context".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_noop_linker() {
        let linker = NoOpLinker;
        let result = linker.link_inner_plan("test", "test.apxm").await;
        assert!(result.is_err());
        let result = linker.link_codelet_dag(CodeletDag::new("test")).await;
        assert!(result.is_err());
    }
}
