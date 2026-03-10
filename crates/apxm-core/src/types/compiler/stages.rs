//! Compilation pipeline metadata shared across APXM components.
//!
//! These enums describe the major compilation stages and emission targets. They
//! live in `apxm-core` so every crate (driver, compiler, runtime) interprets them
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
    /// Binary artifact (`.apxmobj`).
    Artifact,
    /// Generated Rust source (kept for debugging).
    Rust,
}

impl EmitFormat {
    /// Minimum stage required to emit this format.
    pub fn required_stage(self) -> CompilationStage {
        match self {
            EmitFormat::Artifact | EmitFormat::Rust => CompilationStage::Lower,
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
