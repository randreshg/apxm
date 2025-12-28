//! Compiler wrapper used by the driver.

use apxm_compiler::{Context, Module, Pipeline};
use apxm_core::types::OptimizationLevel;
use apxm_core::utils::build::MlirEnvReport;
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

    /// Compile the provided file (DSL or MLIR) into a compiler module.
    pub fn compile(&self, path: &Path, is_mlir: bool) -> Result<Module, DriverError> {
        let source = fs::read_to_string(path)?;
        let pipeline = Pipeline::with_opt_level(&self.context, self.opt_level);
        if is_mlir || path.extension().map(|e| e == "mlir").unwrap_or(false) {
            pipeline.compile(&source).map_err(DriverError::Compiler)
        } else {
            let filename = path.to_string_lossy().to_string();
            pipeline
                .compile_dsl(&source, &filename)
                .map_err(DriverError::Compiler)
        }
    }
}
