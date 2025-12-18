use std::sync::Arc;

use apxm_core::types::execution::ExecutionDag;
use apxm_runtime::{Runtime, RuntimeExecutionResult};

use crate::{config::LinkerConfig, error::LinkerError};

mod llm;
use llm::configure_llm_registry;
mod inner_plan;
use inner_plan::CompilerInnerPlanLinker;

/// Runtime executor used by the linker to run compiled DAGs.
pub struct RuntimeExecutor {
    runtime: Runtime,
}

impl RuntimeExecutor {
    pub async fn new(config: &LinkerConfig) -> Result<Self, LinkerError> {
        let mut runtime = Runtime::new(config.runtime_config.clone())
            .await
            .map_err(LinkerError::Runtime)?;

        configure_llm_registry(runtime.llm_registry(), &config.apxm_config).await?;

        let linker = Arc::new(CompilerInnerPlanLinker::new().map_err(LinkerError::Runtime)?);
        runtime.set_inner_plan_linker(linker);

        Ok(Self { runtime })
    }

    pub async fn execute(&self, dag: ExecutionDag) -> Result<RuntimeExecutionResult, LinkerError> {
        self.runtime
            .execute(dag)
            .await
            .map_err(LinkerError::Runtime)
    }

    /// Get the LLM registry from the runtime
    pub fn llm_registry(&self) -> Arc<apxm_models::registry::LLMRegistry> {
        self.runtime.llm_registry_arc()
    }
}
