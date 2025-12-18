use std::path::Path;

use apxm_artifact::Artifact;
use apxm_compiler::Module;
use apxm_core::error::runtime::RuntimeError;
use apxm_core::log_info;
use apxm_runtime::RuntimeExecutionResult;

use crate::{
    compiler::Compiler, config::LinkerConfig, error::LinkerError, runtime::RuntimeExecutor,
};

/// Result returned after linking compilation/runtime steps.
pub struct LinkResult {
    /// Compiler module produced by linking.
    pub module: Module,
    /// Compiled artifact that can be reused.
    pub artifact: Artifact,
    /// Runtime execution report for this artifact.
    pub execution: RuntimeExecutionResult,
}

/// High-level linker that orchestrates compiler and runtime execution.
pub struct Linker {
    compiler: Compiler,
    runtime: RuntimeExecutor,
}

impl Linker {
    /// Create a new linker instance with the provided configuration.
    pub async fn new(config: LinkerConfig) -> Result<Self, LinkerError> {
        let compiler = Compiler::new()?;
        let runtime = RuntimeExecutor::new(&config).await?;

        Ok(Self { compiler, runtime })
    }

    /// Compile the user file and execute the generated artifact through the runtime.
    pub async fn run(&self, input: &Path, mlir: bool) -> Result<LinkResult, LinkerError> {
        log_info!(
            "linker",
            "Compiling {} as {}",
            input.display(),
            if mlir { "MLIR" } else { "DSL" }
        );
        let module = self.compiler.compile(input, mlir)?;
        let artifact_bytes = module.generate_artifact_bytes()?;
        let artifact = Artifact::from_bytes(&artifact_bytes)
            .map_err(|e| LinkerError::Runtime(RuntimeError::State(e.to_string())))?;

        let execution = self.runtime.execute(artifact.dag().clone()).await?;

        Ok(LinkResult {
            module,
            artifact,
            execution,
        })
    }

    /// Get the LLM registry from the runtime
    pub fn runtime_llm_registry(&self) -> std::sync::Arc<apxm_models::registry::LLMRegistry> {
        self.runtime.llm_registry()
    }
}
