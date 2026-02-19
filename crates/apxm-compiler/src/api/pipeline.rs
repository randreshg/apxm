//! Pipeline API for compiling and optimizing modules.

use crate::api::{Context, Module};
use crate::passes::PassManager;
use apxm_core::error::compiler::{CompilerError, Result};
use apxm_core::error::{builder::ErrorBuilder, codes::ErrorCode};
use apxm_core::types::{OptimizationLevel, PipelineConfig};
use apxm_graph::ApxmGraph;

/// Pipeline API for compiling and optimizing modules.
pub struct Pipeline<'ctx> {
    context: &'ctx Context,
    config: PipelineConfig,
}

impl<'ctx> Pipeline<'ctx> {
    /// Creates a new pipeline with default configuration.
    pub fn new(context: &'ctx Context) -> Self {
        Self {
            context,
            config: PipelineConfig::default(),
        }
    }

    /// Creates a new pipeline with custom configuration.
    pub fn with_config(context: &'ctx Context, config: PipelineConfig) -> Self {
        Self { context, config }
    }

    pub fn with_opt_level(context: &'ctx Context, level: OptimizationLevel) -> Self {
        let config = PipelineConfig {
            opt_level: level,
            ..Default::default()
        };
        Self { context, config }
    }

    pub fn compile(&self, source: &str) -> Result<Module> {
        let module = Module::parse(self.context, source)?;
        self.process_module(module)
    }

    pub fn compile_dsl(&self, source: &str, filename: &str) -> Result<Module> {
        let module = Module::parse_dsl(self.context, source, filename)?;
        self.process_module(module)
    }

    /// Compile a graph input by lowering to textual MLIR first.
    pub fn compile_graph(&self, graph: &ApxmGraph) -> Result<Module> {
        let mlir_text = graph.to_mlir().map_err(|e| {
            CompilerError::Unsupported(Box::new(ErrorBuilder::generic(
                ErrorCode::InternalError,
                format!("Graph lowering failed: {e}"),
            )))
        })?;
        let module = Module::parse(self.context, &mlir_text)?;
        self.process_module(module)
    }

    fn process_module(&self, module: Module) -> Result<Module> {
        if self.config.verify {
            module.verify()?;
        }

        let pm = PassManager::from_opt_level(self.context, self.config.opt_level)?;
        pm.run(&module)?;

        if self.config.verify {
            module.verify()?;
        }

        Ok(module)
    }

    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}
