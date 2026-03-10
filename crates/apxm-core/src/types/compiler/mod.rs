//! Compilation pipeline types module.
//!
//! Contains types for compilation stages, optimization levels, and code generation.

mod codegen;
mod optimization;
mod passes;
mod stages;

pub use codegen::CodegenOptions;
pub use optimization::{OptimizationLevel, PipelineConfig};
pub use passes::{PassCategory, PassInfo};
pub use stages::{CompilationStage, EmitFormat, stage_rank};
