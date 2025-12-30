//! Runtime executor used by the driver to run compiled DAGs.

use std::sync::Arc;

use apxm_artifact::Artifact;
use apxm_core::types::execution::ExecutionDag;
use apxm_runtime::{Runtime, RuntimeExecutionResult};

use crate::{error::DriverError, linker::LinkerConfig};

mod llm;
use llm::configure_llm_registry;
mod inner_plan;
use inner_plan::CompilerInnerPlanLinker;

/// Runtime executor used by the driver to run compiled DAGs.
pub struct RuntimeExecutor {
    runtime: Runtime,
}

impl RuntimeExecutor {
    pub async fn new(config: &LinkerConfig) -> Result<Self, DriverError> {
        let mut runtime = Runtime::new(config.runtime_config.clone())
            .await
            .map_err(DriverError::Runtime)?;

        configure_llm_registry(runtime.llm_registry(), &config.apxm_config).await?;

        let linker = Arc::new(CompilerInnerPlanLinker::new().map_err(DriverError::Runtime)?);
        runtime.set_inner_plan_linker(linker);

        Ok(Self { runtime })
    }

    pub async fn execute(&self, dag: ExecutionDag) -> Result<RuntimeExecutionResult, DriverError> {
        self.runtime
            .execute(dag)
            .await
            .map_err(DriverError::Runtime)
    }

    /// Execute an artifact with @entry flow requirement.
    ///
    /// This method will return an error if the artifact doesn't have an @entry flow.
    pub async fn execute_artifact_auto(
        &self,
        artifact: Artifact,
    ) -> Result<RuntimeExecutionResult, DriverError> {
        self.runtime
            .execute_artifact_auto(artifact)
            .await
            .map_err(DriverError::Runtime)
    }

    /// Get the LLM registry from the runtime
    pub fn llm_registry(&self) -> Arc<apxm_backends::LLMRegistry> {
        self.runtime.llm_registry_arc()
    }

    /// Get list of available capability names
    ///
    /// Returns the names of all capabilities registered in the runtime.
    /// Useful for validation, UI display, and constraining LLM generation.
    pub fn capability_names(&self) -> Vec<String> {
        self.runtime.capability_system().list_capability_names()
    }

    /// Get capability system reference (for advanced usage)
    ///
    /// Provides direct access to the capability system for
    /// advanced operations like capability inspection or registration.
    pub fn capability_system(&self) -> &apxm_runtime::capability::CapabilitySystem {
        self.runtime.capability_system()
    }
}
