//! Inner plan linker interface for runtime
//!
//! This module provides the interface for linking inner plan DSL code
//! during runtime execution. The linker acts as a bridge between the
//! runtime and the compiler, delegating parsing/validation to the compiler.

use apxm_core::{error::RuntimeError, types::execution::ExecutionDag};
use async_trait::async_trait;

/// Result type for inner plan linking
pub type LinkResult = Result<ExecutionDag, RuntimeError>;

/// Trait for linking inner plan DSL code into ExecutionDAGs
///
/// The linker bridges the runtime and compiler:
/// - Runtime calls linker with DSL code
/// - Linker delegates to compiler for parsing/validation
/// - Linker returns validated DAG to runtime
#[async_trait]
pub trait InnerPlanLinker: Send + Sync {
    /// Link inner plan DSL code into an ExecutionDAG
    ///
    /// # Arguments
    ///
    /// * `dsl_code` - The APxM DSL source code from the LLM
    /// * `source_name` - Name for error reporting (e.g., "inner_plan_<execution_id>")
    ///
    /// # Returns
    ///
    /// A validated ExecutionDAG ready for splicing
    ///
    /// # Errors
    ///
    /// Returns RuntimeError::Compiler if parsing/validation fails
    async fn link_inner_plan(&self, dsl_code: &str, source_name: &str) -> LinkResult;
}

/// No-op linker for contexts that don't support inner plan linking
pub struct NoOpLinker;

#[async_trait]
impl InnerPlanLinker for NoOpLinker {
    async fn link_inner_plan(&self, _dsl_code: &str, _source_name: &str) -> LinkResult {
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
    }
}
