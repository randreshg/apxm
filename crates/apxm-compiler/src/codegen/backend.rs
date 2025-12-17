//! High-level Rust code generation backend.
//!
//! This module provides the main API for generating Rust code from MLIR modules.
//! It orchestrates the emission process using the emitter and operation emitter.

use crate::Module;
use apxm_core::error::compiler::{CompilerError, Result};
use apxm_core::error::{Error, ErrorCode, Span};
use apxm_core::types::CodegenOptions;
use std::path::Path;

/// Rust code generation backend
///
/// Orchestrates the transpilation of MLIR AIS dialect to Rust source code.
///
/// # Example
///
/// ```rust,ignore
/// use apxm_compiler::{Context, Module, codegen::RustBackend};
///
/// let ctx = Context::new()?;
/// let module = Module::parse(&ctx, mlir_source, "agent.mlir")?;
///
/// let backend = RustBackend::new();
/// let rust_code = backend.generate(&module)?;
///
/// std::fs::write("agent.rs", rust_code)?;
/// ```
pub struct RustBackend {
    options: CodegenOptions,
}

impl RustBackend {
    /// Create a new backend with default options
    pub fn new() -> Self {
        Self {
            options: CodegenOptions::default(),
        }
    }

    /// Create a backend with custom options
    pub fn with_options(options: CodegenOptions) -> Self {
        Self { options }
    }

    /// Get a reference to the options
    pub fn options(&self) -> &CodegenOptions {
        &self.options
    }

    /// Get a mutable reference to the options
    pub fn options_mut(&mut self) -> &mut CodegenOptions {
        &mut self.options
    }

    /// Generate Rust source code from an MLIR module
    ///
    /// This is the main entry point for code generation.
    /// It orchestrates the entire emission process.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The module cannot be converted to MLIR text
    /// - The MLIR contains unsupported operations
    /// - Code generation fails for any reason
    pub fn generate(&self, module: &Module) -> Result<String> {
        // Call C++ implementation which emits Rust code directly
        module.generate_rust_code_with_options(&self.options)
    }

    /// Generate and write to file
    ///
    /// Convenience method that generates code and writes it to a file.
    pub fn generate_to_file(&self, module: &Module, path: &Path) -> Result<()> {
        let code = self.generate(module)?;
        std::fs::write(path, code).map_err(|e| {
            CompilerError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to write to {}: {}", path.display(), e),
            ))
        })
    }

    /// Validate that generated code compiles
    ///
    /// This runs `rustc --check` on the generated code to ensure it's valid.
    /// Useful for testing and validation.
    #[allow(dead_code)]
    pub fn validate(&self, source: &str) -> Result<()> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new("rustc")
            .arg("--crate-type")
            .arg("bin")
            .arg("-")
            .arg("--check")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                CompilerError::Internal(Box::new(Error::new(
                    ErrorCode::InternalError,
                    format!("Failed to spawn rustc: {}", e),
                    Span::new("<validation>".to_string(), 0, 0, 0),
                )))
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(source.as_bytes()).map_err(|e| {
                CompilerError::Internal(Box::new(Error::new(
                    ErrorCode::InternalError,
                    format!("Failed to write to rustc stdin: {}", e),
                    Span::new("<validation>".to_string(), 0, 0, 0),
                )))
            })?;
        }

        let output = child.wait_with_output().map_err(|e| {
            CompilerError::Internal(Box::new(Error::new(
                ErrorCode::InternalError,
                format!("Failed to wait for rustc: {}", e),
                Span::new("<validation>".to_string(), 0, 0, 0),
            )))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CompilerError::Internal(Box::new(Error::new(
                ErrorCode::InternalError,
                format!("Generated code does not compile:\n{}", stderr),
                Span::new("<validation>".to_string(), 0, 0, 0),
            ))));
        }

        Ok(())
    }
}

impl Default for RustBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let options = CodegenOptions::default();
        assert!(options.optimize);
        assert!(options.emit_comments);
        assert!(!options.emit_debug_symbols);
        assert!(options.standalone);
    }

    #[test]
    fn test_production_options() {
        let options = CodegenOptions::production();
        assert!(options.optimize);
        assert!(!options.emit_comments);
        assert!(!options.emit_debug_symbols);
    }

    #[test]
    fn test_development_options() {
        let options = CodegenOptions::development();
        assert!(!options.optimize);
        assert!(options.emit_comments);
        assert!(options.emit_debug_symbols);
    }

    #[test]
    fn test_options_builder() {
        let options = CodegenOptions::default()
            .with_module_name("test_agent")
            .with_standalone(false)
            .with_optimize(false)
            .with_comments(true);

        assert_eq!(options.module_name, Some("test_agent".to_string()));
        assert!(!options.standalone);
        assert!(!options.optimize);
        assert!(options.emit_comments);
    }

    #[test]
    fn test_backend_creation() {
        let backend = RustBackend::new();
        assert!(backend.options().standalone);

        let backend = RustBackend::with_options(CodegenOptions::production());
        assert!(!backend.options().emit_comments);
    }

    #[test]
    fn test_backend_options_mutation() {
        let mut backend = RustBackend::new();
        backend.options_mut().optimize = false;
        assert!(!backend.options().optimize);
    }

    #[test]
    fn test_rust_codegen_basic() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::{Context, Module, codegen::RustBackend};

        let ctx = Context::new()?;
        let mlir = "module {}";
        let module = Module::parse(&ctx, mlir)?;

        let backend = RustBackend::new();
        let code = backend.generate(&module)?;

        assert!(code.contains("Generated by APXM Compiler"));
        assert!(code.contains("use apxm_runtime"));

        Ok(())
    }
}
