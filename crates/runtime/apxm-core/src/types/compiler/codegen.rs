//! Code generation configuration types.
//!
//! These types are used across the compiler and related tools for configuring
//! code generation behavior.

/// Options for code generation
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    /// Enable optimization hints in generated code
    pub optimize: bool,
    /// Emit comments in generated code
    pub emit_comments: bool,
    /// Emit debug symbols and annotations
    pub emit_debug_symbols: bool,
    /// Target runtime version (for compatibility)
    pub runtime_version: String,
    /// Generate standalone executable (with main function)
    pub standalone: bool,
    /// Module name for generated code
    pub module_name: Option<String>,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            optimize: true,
            emit_comments: true,
            emit_debug_symbols: false,
            runtime_version: "0.1.0".to_string(),
            standalone: true,
            module_name: None,
        }
    }
}

impl CodegenOptions {
    /// Create options for production builds
    pub fn production() -> Self {
        Self {
            optimize: true,
            emit_comments: false,
            emit_debug_symbols: false,
            runtime_version: "0.1.0".to_string(),
            standalone: true,
            module_name: None,
        }
    }

    /// Create options for development builds
    pub fn development() -> Self {
        Self {
            optimize: false,
            emit_comments: true,
            emit_debug_symbols: true,
            runtime_version: "0.1.0".to_string(),
            standalone: true,
            module_name: None,
        }
    }

    /// Set module name
    pub fn with_module_name(mut self, name: impl Into<String>) -> Self {
        self.module_name = Some(name.into());
        self
    }

    /// Set standalone mode
    pub fn with_standalone(mut self, standalone: bool) -> Self {
        self.standalone = standalone;
        self
    }

    /// Set optimization level
    pub fn with_optimize(mut self, optimize: bool) -> Self {
        self.optimize = optimize;
        self
    }

    /// Set comment emission
    pub fn with_comments(mut self, emit_comments: bool) -> Self {
        self.emit_comments = emit_comments;
        self
    }

    /// Set debug symbol emission
    pub fn with_debug_symbols(mut self, emit_debug_symbols: bool) -> Self {
        self.emit_debug_symbols = emit_debug_symbols;
        self
    }
}
