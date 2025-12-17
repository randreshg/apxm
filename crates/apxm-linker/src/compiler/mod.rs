use apxm_compiler::{Context, Module};
use std::fs;
use std::path::Path;

use crate::error::LinkerError;

/// Compiler wrapper used by the linker.
pub struct Compiler {
    context: Context,
}

impl Compiler {
    /// Initialize a new compiler context for linking.
    pub fn new() -> Result<Self, LinkerError> {
        let context = Context::new().map_err(LinkerError::Compiler)?;
        Ok(Self { context })
    }

    /// Compile the provided file (DSL or MLIR) into a compiler module.
    pub fn compile(&self, path: &Path, is_mlir: bool) -> Result<Module, LinkerError> {
        if is_mlir || path.extension().map(|e| e == "mlir").unwrap_or(false) {
            Module::parse_file(&self.context, path).map_err(LinkerError::Compiler)
        } else {
            let source = fs::read_to_string(path)?;
            let filename = path.to_string_lossy().to_string();
            Module::parse_dsl(&self.context, &source, &filename).map_err(LinkerError::Compiler)
        }
    }
}
