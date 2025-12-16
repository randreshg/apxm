//! Compilation pipeline metadata shared across APXM components.
//!
//! These enums describe the major compilation stages and emission targets. They
//! live in `apxm-core` so every crate (CLI, compiler, runtime) interprets them
//! consistently.

/// Logical compilation stages exposed to users and automation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompilationStage {
    /// Parse-only pipeline (DSL → MLIR).
    #[default]
    Parse,
    /// Run optimization passes after parsing.
    Optimize,
    /// Lowering stage (AIS → Async).
    Lower,
}

/// Output targets supported by the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitFormat {
    /// Raw MLIR textual output (pre-optimization).
    Mlir,
    /// Optimized MLIR textual output.
    Optimized,
    /// Async-lowered MLIR textual output.
    Async,
    /// JSON wrapper over textual MLIR.
    Json,
    /// Generated Rust source.
    Rust,
}

impl EmitFormat {
    /// Minimum stage required to emit this format.
    pub fn required_stage(self) -> CompilationStage {
        match self {
            EmitFormat::Mlir | EmitFormat::Json => CompilationStage::Parse,
            EmitFormat::Optimized => CompilationStage::Optimize,
            EmitFormat::Async | EmitFormat::Rust => CompilationStage::Lower,
        }
    }
}

/// Deterministic ordering helper for stages (Parse < ... < Binary).
pub fn stage_rank(stage: CompilationStage) -> u8 {
    match stage {
        CompilationStage::Parse => 0,
        CompilationStage::Optimize => 1,
        CompilationStage::Lower => 2,
    }
}
