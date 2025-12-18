use std::sync::Arc;

use apxm_artifact::Artifact;
use apxm_compiler::{self, Module};
use apxm_core::types::execution::ExecutionDag;
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
            "linker::inner_plan",
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

        log_info!(
            "linker::inner_plan",
            source = %source_name,
            nodes = dag.nodes.len(),
            edges = dag.edges.len(),
            "Inner plan linked successfully"
        );

        Ok(dag)
    }
}
