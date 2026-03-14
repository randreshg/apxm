//! High-level linker that orchestrates compiler and runtime execution.

use std::path::Path;

use apxm_artifact::Artifact;
use apxm_core::error::runtime::RuntimeError;
use apxm_core::log_info;
use apxm_core::types::OptimizationLevel;
use apxm_runtime::{RuntimeConfig, RuntimeExecutionResult};

use crate::{
    cache, compiler::Compiler, config::ApXmConfig, error::DriverError, runtime::RuntimeExecutor,
};

/// Linker configuration that drives compiler and runtime orchestration.
#[derive(Debug, Clone)]
pub struct LinkerConfig {
    /// APxM configuration (providers, tools, policies).
    pub apxm_config: ApXmConfig,

    /// Optional runtime configuration overrides.
    pub runtime_config: RuntimeConfig,

    /// Optimization level for compilation.
    pub opt_level: OptimizationLevel,

    /// When `true`, skip the artifact cache entirely.
    pub no_cache: bool,
}

impl LinkerConfig {
    /// Create a configuration from an `ApXmConfig` instance.
    pub fn from_apxm_config(apxm_config: ApXmConfig) -> Self {
        Self {
            apxm_config,
            runtime_config: RuntimeConfig::default(),
            opt_level: OptimizationLevel::O1,
            no_cache: false,
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
    pub runtime_time: std::time::Duration,
}

/// High-level linker that orchestrates compiler and runtime execution.
pub struct Linker {
    compiler: Compiler,
    runtime: RuntimeExecutor,
    no_cache: bool,
}

impl Linker {
    /// Create a new linker instance with the provided configuration.
    pub async fn new(config: LinkerConfig) -> Result<Self, DriverError> {
        let compiler = Compiler::with_opt_level(config.opt_level)?;
        let runtime = RuntimeExecutor::new(&config).await?;
        let no_cache = config.no_cache;

        Ok(Self {
            compiler,
            runtime,
            no_cache,
        })
    }

    /// Compile graph input into an executable artifact.
    ///
    /// When caching is enabled (the default), the graph JSON is hashed and
    /// looked up in `~/.cache/apxm/artifacts/`.  On a cache hit the
    /// compilation step is skipped entirely.
    pub fn compile_graph(&self, input: &Path) -> Result<Artifact, DriverError> {
        let graph = self.compiler.load_graph(input)?;

        // Try the artifact cache first.
        let graph_json = graph.to_json().unwrap_or_default();
        let hash = cache::graph_hash(&graph_json).ok();

        if !self.no_cache {
            if let Some(ref h) = hash {
                if let Some(cached_bytes) = cache::load_cached(h)? {
                    log_info!("driver", "cache hit for graph hash {}", h);
                    let artifact = Artifact::from_bytes(&cached_bytes)
                        .map_err(|e| DriverError::Runtime(RuntimeError::State(e.to_string())))?;
                    return Ok(artifact);
                }
            }
        }

        let module = self.compiler.compile_graph(&graph)?;
        let artifact_bytes = module.generate_artifact_bytes()?;

        // Store in cache for next time.
        if !self.no_cache {
            if let Some(ref h) = hash {
                let _ = cache::store_cached(h, &artifact_bytes);
            }
        }

        let artifact = Artifact::from_bytes(&artifact_bytes)
            .map_err(|e| DriverError::Runtime(RuntimeError::State(e.to_string())))?;

        let dag = artifact.dag().ok_or_else(|| {
            DriverError::Runtime(RuntimeError::State("Artifact contains no DAGs".to_string()))
        })?;
        if let Err(err) = dag.validate() {
            return Err(DriverError::Runtime(RuntimeError::State(format!(
                "Artifact DAG validation failed: {}",
                err
            ))));
        }
        Ok(artifact)
    }

    /// Compile graph input and execute with entry arguments.
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
            artifact,
            execution,
            #[cfg(feature = "metrics")]
            metrics: LinkMetrics {
                compile_time,
                runtime_time,
            },
        })
    }
}
