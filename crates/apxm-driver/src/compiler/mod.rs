//! Compiler wrapper used by the driver.

use apxm_compiler::{Context, Module, Pipeline};
use apxm_core::types::OptimizationLevel;
use apxm_core::utils::build::MlirEnvReport;
use apxm_graph::ApxmGraph;
use std::fs;
use std::path::Path;

use crate::error::DriverError;

/// Compiler wrapper used by the linker.
pub struct Compiler {
    context: Context,
    opt_level: OptimizationLevel,
}

impl Compiler {
    /// Initialize a new compiler context for linking.
    pub fn new() -> Result<Self, DriverError> {
        Self::with_opt_level(OptimizationLevel::O1)
    }

    /// Initialize a new compiler context with a specific optimization level.
    pub fn with_opt_level(opt_level: OptimizationLevel) -> Result<Self, DriverError> {
        let report = MlirEnvReport::detect();
        report.apply_env();
        if !report.is_ready() {
            return Err(DriverError::Driver(format!(
                "MLIR toolchain not detected.\n{}\nSet MLIR_DIR/MLIR_PREFIX/LLVM_PREFIX/CONDA_PREFIX or ensure mlir-tblgen is on PATH.",
                report.summary()
            )));
        }

        let context = Context::new().map_err(DriverError::Compiler)?;
        Ok(Self { context, opt_level })
    }

    /// Get the current optimization level.
    pub fn opt_level(&self) -> OptimizationLevel {
        self.opt_level
    }

    /// Compile a graph file into a compiler module.
    pub fn compile(&self, path: &Path) -> Result<Module, DriverError> {
        if path.extension().and_then(|ext| ext.to_str()) == Some("mlir")
            || path.extension().and_then(|ext| ext.to_str()) == Some("ais")
        {
            return Err(DriverError::Driver(
                "Graph-only compile path requires ApxmGraph JSON/binary input".to_string(),
            ));
        }

        let graph = self.load_graph(path)?;
        self.compile_graph(&graph)
    }

    /// Compile an in-memory graph by lowering to MLIR and running optimizer passes.
    pub fn compile_graph(&self, graph: &ApxmGraph) -> Result<Module, DriverError> {
        let pipeline = Pipeline::with_opt_level(&self.context, self.opt_level);
        pipeline.compile_graph(graph).map_err(DriverError::Compiler)
    }

    /// Load graph input (JSON or bincode) from disk.
    pub fn load_graph(&self, path: &Path) -> Result<ApxmGraph, DriverError> {
        let bytes = fs::read(path)?;

        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => std::str::from_utf8(&bytes)
                .map_err(|e| DriverError::Driver(format!("Graph file is not UTF-8 JSON: {e}")))
                .and_then(|text| {
                    ApxmGraph::from_json(text)
                        .map_err(|e| DriverError::Driver(format!("Graph parse error: {e}")))
                }),
            _ => ApxmGraph::from_bytes(&bytes)
                .map_err(|e| DriverError::Driver(format!("Graph parse error: {e}"))),
        }
    }
}
