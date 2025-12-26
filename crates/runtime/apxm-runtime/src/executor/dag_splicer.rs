//! DAG splicing interface for runtime
//!
//! This module provides the interface for operation handlers to splice
//! inner plan DAGs into the live execution, enabling inner/outer plan unification.

use apxm_core::{error::RuntimeError, types::execution::ExecutionDag};
use async_trait::async_trait;
use std::collections::HashMap;

use apxm_core::types::TokenId;

/// Result type for DAG splicing operations
pub type SpliceResult = Result<HashMap<TokenId, TokenId>, RuntimeError>;

/// Trait for splicing DAGs into live execution
///
/// This enables dynamic inner/outer plan unification where inner plan
/// DAGs generated during execution are merged into the currently executing
/// outer plan DAG.
#[async_trait]
pub trait DagSplicer: Send + Sync {
    /// Splice an inner DAG into the live execution
    ///
    /// # Arguments
    ///
    /// * `inner_dag` - The inner DAG to splice in
    /// * `token_connections` - Mapping from inner DAG input tokens to outer DAG output tokens
    ///
    /// # Returns
    ///
    /// Mapping from original inner DAG token IDs to remapped token IDs
    ///
    /// # Errors
    ///
    /// Returns error if splicing fails (e.g., invalid token connections)
    async fn splice_dag(
        &self,
        inner_dag: ExecutionDag,
        token_connections: HashMap<TokenId, TokenId>,
    ) -> SpliceResult;
}

/// No-op splicer for contexts that don't support dynamic splicing
pub struct NoOpSplicer;

#[async_trait]
impl DagSplicer for NoOpSplicer {
    async fn splice_dag(
        &self,
        _inner_dag: ExecutionDag,
        _token_connections: HashMap<TokenId, TokenId>,
    ) -> SpliceResult {
        Err(RuntimeError::State(
            "Dynamic DAG splicing not supported in this context".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_noop_splicer() {
        let splicer = NoOpSplicer;
        let dag = ExecutionDag::new();
        let connections = HashMap::new();

        let result = splicer.splice_dag(dag, connections).await;
        assert!(result.is_err());
    }
}
