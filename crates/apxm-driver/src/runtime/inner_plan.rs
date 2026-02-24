//! Compiler-backed implementation of the runtime linker trait.

use std::sync::Arc;

use apxm_artifact::Artifact;
use apxm_compiler::{self, Module};
use apxm_core::types::execution::{CodeletDag, ExecutionDag};
use apxm_core::{log_debug, log_info};
use apxm_runtime::{InnerPlanLinker, RuntimeError};
use async_trait::async_trait;

/// Compiler-backed implementation of the runtime linker trait.
pub struct CompilerInnerPlanLinker {
    context: Arc<parking_lot::Mutex<apxm_compiler::Context>>,
}

impl CompilerInnerPlanLinker {
    pub fn new() -> Result<Self, RuntimeError> {
        let context = apxm_compiler::Context::new().map_err(|e| {
            RuntimeError::State(format!("Failed to initialize compiler context: {}", e))
        })?;

        Ok(Self {
            context: Arc::new(parking_lot::Mutex::new(context)),
        })
    }
}

#[async_trait]
impl InnerPlanLinker for CompilerInnerPlanLinker {
    async fn link_inner_plan(
        &self,
        dsl_code: &str,
        source_name: &str,
    ) -> Result<ExecutionDag, RuntimeError> {
        log_debug!(
            "driver::inner_plan",
            source = %source_name,
            dsl_length = dsl_code.len(),
            "Linking inner plan DSL"
        );

        let context = self.context.lock();
        let module = Module::parse_dsl(&context, dsl_code, source_name)
            .map_err(|e| RuntimeError::State(format!("Inner plan DSL parsing failed: {}", e)))?;

        let artifact_bytes = module.generate_artifact_bytes().map_err(|e| {
            RuntimeError::State(format!("Inner plan artifact generation failed: {}", e))
        })?;

        let artifact = Artifact::from_bytes(&artifact_bytes).map_err(|e| {
            RuntimeError::State(format!("Inner plan artifact parsing failed: {}", e))
        })?;

        let dag = artifact.into_dag();

        // Validate the inner DAG before returning it to the runtime.
        // This ensures we catch cycles or inconsistent token/node references
        // early and return a clear error instead of allowing the scheduler
        // to detect a deadlock at runtime.
        if let Err(e) = dag.validate() {
            return Err(RuntimeError::State(format!(
                "Inner plan DAG validation failed: {}",
                e
            )));
        }

        log_info!(
            "driver::inner_plan",
            source = %source_name,
            nodes = dag.nodes.len(),
            edges = dag.edges.len(),
            "Inner plan linked successfully"
        );

        Ok(dag)
    }

    async fn link_codelet_dag(&self, dag: CodeletDag) -> Result<ExecutionDag, RuntimeError> {
        log_debug!(
            "driver::inner_plan",
            dag_name = %dag.name,
            codelets = dag.codelets.len(),
            "Linking inner plan codelet DAG"
        );

        let execution_dag = dag.to_execution_dag()?;
        execution_dag.validate()?;

        log_info!(
            "driver::inner_plan",
            nodes = execution_dag.nodes.len(),
            edges = execution_dag.edges.len(),
            "Inner plan codelet DAG linked successfully"
        );

        Ok(execution_dag)
    }
}
