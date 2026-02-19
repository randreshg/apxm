//! High-level linker that orchestrates compiler and runtime execution.

use std::path::Path;

use apxm_artifact::Artifact;
use apxm_compiler::Module;
use apxm_core::error::runtime::RuntimeError;
use apxm_core::log_info;
use apxm_core::types::OptimizationLevel;
use apxm_runtime::{RuntimeConfig, RuntimeExecutionResult};

use crate::{compiler::Compiler, config::ApXmConfig, error::DriverError, runtime::RuntimeExecutor};

/// Linker configuration that drives compiler and runtime orchestration.
#[derive(Debug, Clone)]
pub struct LinkerConfig {
    /// APxM configuration (providers, tools, policies).
    pub apxm_config: ApXmConfig,

    /// Optional runtime configuration overrides.
    pub runtime_config: RuntimeConfig,

    /// Optimization level for compilation.
    pub opt_level: OptimizationLevel,
}

impl LinkerConfig {
    /// Create a configuration from an `ApXmConfig` instance.
    pub fn from_apxm_config(apxm_config: ApXmConfig) -> Self {
        Self {
            apxm_config,
            runtime_config: RuntimeConfig::default(),
            opt_level: OptimizationLevel::O1,
        }
    }

    /// Set the optimization level.
    pub fn with_opt_level(mut self, opt_level: OptimizationLevel) -> Self {
        self.opt_level = opt_level;
        self
    }
}

/// Result returned after linking compilation/runtime steps.
pub struct LinkResult {
    /// Compiler module produced by linking.
    pub module: Option<Module>,
    /// Compiled artifact that can be reused.
    pub artifact: Artifact,
    /// Runtime execution report for this artifact.
    pub execution: RuntimeExecutionResult,
    /// Metrics about compile/runtime overhead.
    #[cfg(feature = "metrics")]
    pub metrics: LinkMetrics,
}

/// Timing breakdown for a link+execute run.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub struct LinkMetrics {
    pub compile_time: std::time::Duration,
    pub artifact_time: std::time::Duration,
    pub validation_time: std::time::Duration,
    pub runtime_time: std::time::Duration,
}

/// High-level linker that orchestrates compiler and runtime execution.
pub struct Linker {
    compiler: Compiler,
    runtime: RuntimeExecutor,
}

impl Linker {
    /// Create a new linker instance with the provided configuration.
    pub async fn new(config: LinkerConfig) -> Result<Self, DriverError> {
        let compiler = Compiler::with_opt_level(config.opt_level)?;
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
    pub fn compile_only(&self, input: &Path, mlir: bool) -> Result<Module, DriverError> {
        self.compiler.compile(input, mlir)
    }

    /// Compile graph input into an executable artifact.
    pub fn compile_graph(&self, input: &Path) -> Result<Artifact, DriverError> {
        let graph = self.compiler.load_graph(input)?;
        let module = self.compiler.compile_graph(&graph)?;
        let artifact_bytes = module.generate_artifact_bytes()?;
        let artifact = Artifact::from_bytes(&artifact_bytes)
            .map_err(|e| DriverError::Runtime(RuntimeError::State(e.to_string())))?;

        if let Err(err) = artifact.dag().validate() {
            return Err(DriverError::Runtime(RuntimeError::State(format!(
                "Artifact DAG validation failed: {}",
                err
            ))));
        }
        Ok(artifact)
    }

    /// Compile the user file and execute the generated artifact through the runtime.
    pub async fn run(&self, input: &Path, mlir: bool) -> Result<LinkResult, DriverError> {
        log_info!(
            "driver",
            "Compiling {} as {}",
            input.display(),
            if mlir { "MLIR" } else { "DSL" }
        );
        #[cfg(feature = "metrics")]
        let compile_start = std::time::Instant::now();
        let module = self.compiler.compile(input, mlir)?;
        #[cfg(feature = "metrics")]
        let compile_time = compile_start.elapsed();

        #[cfg(feature = "metrics")]
        let artifact_start = std::time::Instant::now();
        let artifact_bytes = module.generate_artifact_bytes()?;
        #[cfg(feature = "metrics")]
        let artifact_time = artifact_start.elapsed();

        let artifact = Artifact::from_bytes(&artifact_bytes)
            .map_err(|e| DriverError::Runtime(RuntimeError::State(e.to_string())))?;

        // Validate the artifact-level DAG before invoking the runtime so we
        // fail fast on obvious cycles or malformed dependency graphs instead
        // of letting the runtime watchdog detect a deadlock later.
        #[cfg(feature = "metrics")]
        let validation_start = std::time::Instant::now();
        if let Err(e) = artifact.dag().validate() {
            return Err(DriverError::Runtime(RuntimeError::State(format!(
                "Artifact DAG validation failed: {}",
                e
            ))));
        }
        #[cfg(feature = "metrics")]
        let validation_time = validation_start.elapsed();

        #[cfg(feature = "metrics")]
        let runtime_start = std::time::Instant::now();
        // Use execute_artifact_auto to enforce @entry flow requirement
        let execution = self.runtime.execute_artifact_auto(artifact.clone()).await?;
        #[cfg(feature = "metrics")]
        let runtime_time = runtime_start.elapsed();

        Ok(LinkResult {
            module: Some(module),
            artifact,
            execution,
            #[cfg(feature = "metrics")]
            metrics: LinkMetrics {
                compile_time,
                artifact_time,
                validation_time,
                runtime_time,
            },
        })
    }

    /// Compile the user file and execute with provided arguments for entry flow parameters.
    pub async fn run_with_args(
        &self,
        input: &Path,
        mlir: bool,
        args: Vec<String>,
    ) -> Result<LinkResult, DriverError> {
        log_info!(
            "driver",
            "Compiling {} as {} with {} args",
            input.display(),
            if mlir { "MLIR" } else { "DSL" },
            args.len()
        );
        #[cfg(feature = "metrics")]
        let compile_start = std::time::Instant::now();
        let module = self.compiler.compile(input, mlir)?;
        #[cfg(feature = "metrics")]
        let compile_time = compile_start.elapsed();

        #[cfg(feature = "metrics")]
        let artifact_start = std::time::Instant::now();
        let artifact_bytes = module.generate_artifact_bytes()?;
        #[cfg(feature = "metrics")]
        let artifact_time = artifact_start.elapsed();

        let artifact = Artifact::from_bytes(&artifact_bytes)
            .map_err(|e| DriverError::Runtime(RuntimeError::State(e.to_string())))?;

        #[cfg(feature = "metrics")]
        let validation_start = std::time::Instant::now();
        if let Err(e) = artifact.dag().validate() {
            return Err(DriverError::Runtime(RuntimeError::State(format!(
                "Artifact DAG validation failed: {}",
                e
            ))));
        }
        #[cfg(feature = "metrics")]
        let validation_time = validation_start.elapsed();

        #[cfg(feature = "metrics")]
        let runtime_start = std::time::Instant::now();
        let execution = self
            .runtime
            .execute_artifact_with_args(artifact.clone(), args)
            .await?;
        #[cfg(feature = "metrics")]
        let runtime_time = runtime_start.elapsed();

        Ok(LinkResult {
            module: Some(module),
            artifact,
            execution,
            #[cfg(feature = "metrics")]
            metrics: LinkMetrics {
                compile_time,
                artifact_time,
                validation_time,
                runtime_time,
            },
        })
    }

    /// Compile graph input and execute with optional entry arguments.
    pub async fn run_graph(
        &self,
        input: &Path,
        args: Vec<String>,
    ) -> Result<LinkResult, DriverError> {
        log_info!("driver", "Compiling graph {}", input.display());
        #[cfg(feature = "metrics")]
        let compile_start = std::time::Instant::now();
        let artifact = self.compile_graph(input)?;
        #[cfg(feature = "metrics")]
        let compile_time = compile_start.elapsed();

        #[cfg(feature = "metrics")]
        let runtime_start = std::time::Instant::now();
        let execution = self
            .runtime
            .execute_artifact_with_args(artifact.clone(), args)
            .await?;
        #[cfg(feature = "metrics")]
        let runtime_time = runtime_start.elapsed();

        Ok(LinkResult {
            module: None,
            artifact,
            execution,
            #[cfg(feature = "metrics")]
            metrics: LinkMetrics {
                compile_time,
                artifact_time: std::time::Duration::ZERO,
                validation_time: std::time::Duration::ZERO,
                runtime_time,
            },
        })
    }

    /// Get the LLM registry from the runtime
    pub fn runtime_llm_registry(&self) -> std::sync::Arc<apxm_backends::LLMRegistry> {
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
    /// - In future tooling to constrain DSL generation
    /// - In `run` command to validate user-written DSL
    /// - For displaying help/documentation to users
    pub fn runtime_capabilities(&self) -> Vec<String> {
        self.runtime.capability_names()
    }
}
