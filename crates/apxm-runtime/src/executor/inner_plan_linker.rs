//! Inner plan linker interface for runtime
//!
//! This module provides the interface for linking inner plan DSL code
//! during runtime execution. The linker acts as a bridge between the
//! runtime and the compiler, delegating parsing/validation to the compiler.

use apxm_core::{error::RuntimeError, types::execution::ExecutionDag};
use async_trait::async_trait;

#[cfg(feature = "compiler")]
use std::sync::Arc;

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

/// Compiler-backed linker implementation
///
/// This implementation requires apxm-compiler to be available.
/// It delegates all parsing and validation to the compiler.
#[cfg(feature = "compiler")]
pub struct CompilerLinker {
    context: Arc<parking_lot::Mutex<apxm_compiler::Context>>,
}

#[cfg(feature = "compiler")]
impl CompilerLinker {
    /// Create a new compiler-backed linker
    pub fn new() -> Result<Self, RuntimeError> {
        let context = apxm_compiler::Context::new().map_err(|e| RuntimeError::Compiler {
            phase: "init_context".to_string(),
            message: format!("Failed to initialize compiler context: {}", e),
        })?;

        Ok(Self {
            context: Arc::new(parking_lot::Mutex::new(context)),
        })
    }

    /// Create a linker with a shared compiler context
    pub fn with_context(context: Arc<parking_lot::Mutex<apxm_compiler::Context>>) -> Self {
        Self { context }
    }
}

#[cfg(feature = "compiler")]
#[async_trait]
impl InnerPlanLinker for CompilerLinker {
    async fn link_inner_plan(&self, dsl_code: &str, source_name: &str) -> LinkResult {
        tracing::debug!(
            source = %source_name,
            dsl_length = dsl_code.len(),
            "Linking inner plan DSL"
        );

        // Delegate to compiler for parsing and validation
        let context = self.context.lock();
        let module =
            apxm_compiler::Module::parse_dsl(&context, dsl_code, source_name).map_err(|e| {
                RuntimeError::Compiler {
                    phase: "parse_dsl".to_string(),
                    message: format!("Inner plan DSL parsing failed: {}", e),
                }
            })?;

        // Generate artifact from module
        let artifact_bytes =
            module
                .generate_artifact_bytes()
                .map_err(|e| RuntimeError::Compiler {
                    phase: "generate_artifact".to_string(),
                    message: format!("Inner plan artifact generation failed: {}", e),
                })?;

        // Parse artifact to get DAG
        let artifact = apxm_artifact::Artifact::from_bytes(&artifact_bytes).map_err(|e| {
            RuntimeError::Compiler {
                phase: "parse_artifact".to_string(),
                message: format!("Inner plan artifact parsing failed: {}", e),
            }
        })?;

        let dag = artifact.into_dag();

        tracing::info!(
            source = %source_name,
            nodes = dag.nodes.len(),
            edges = dag.edges.len(),
            "Inner plan linked successfully"
        );

        Ok(dag)
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

    #[cfg(feature = "compiler")]
    #[tokio::test]
    async fn test_compiler_linker() {
        let linker = CompilerLinker::new().unwrap();

        // Test with simple DSL
        let dsl = r#"CONST_STR(value="hello") -> T1;"#;
        let result = linker.link_inner_plan(dsl, "test_inner.apxm").await;

        match result {
            Ok(dag) => {
                assert!(dag.nodes.len() > 0, "DAG should have nodes");
            }
            Err(e) => {
                panic!("Linking should succeed: {}", e);
            }
        }
    }
}
