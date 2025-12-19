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

    /// Compile DSL/MLIR file without execution (for validation)
    ///
    /// This method compiles the input file to verify syntactic correctness
    /// without executing it. Useful for:
    /// - Pre-validation before execution
    /// - DSL generation feedback loops
    /// - IDE/tooling integration
    ///
    /// Returns the compiled Module on success, or a CompilerError on failure.
    pub fn compile_only(&self, input: &Path, mlir: bool) -> Result<Module, LinkerError> {
        self.compiler.compile(input, mlir)
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

        // Validate the artifact-level DAG before invoking the runtime so we
        // fail fast on obvious cycles or malformed dependency graphs instead
        // of letting the runtime watchdog detect a deadlock later.
        if let Err(e) = artifact.dag().validate() {
            return Err(LinkerError::Runtime(RuntimeError::State(format!(
                "Artifact DAG validation failed: {}",
                e
            ))));
        }

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

    /// Get list of available runtime capabilities
    ///
    /// This is useful for:
    /// - Validating DSL before compilation
    /// - Showing available capabilities to users
    /// - Passing to LLM for constrained generation
    ///
    /// The capability names can be used across the system:
    /// - In `apxm-chat` translator to constrain DSL generation
    /// - In `run` command to validate user-written DSL
    /// - For displaying help/documentation to users
    pub fn runtime_capabilities(&self) -> Vec<String> {
        self.runtime.capability_names()
    }
}
