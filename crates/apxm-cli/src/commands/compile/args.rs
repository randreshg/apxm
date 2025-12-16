use apxm_core::types::{CompilationStage, EmitFormat, OptimizationLevel};
use clap::{Args, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Args)]
pub struct CompileArgs {
    /// Input file to compile (.apxm DSL or .mlir)
    #[arg(value_name = "INPUT")]
    pub input: Option<PathBuf>,

    /// Output file path
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Treat input as raw MLIR (not DSL)
    #[arg(long)]
    pub mlir: bool,

    /// Optimization level (0=none, 1=basic, 2=standard, 3=aggressive)
    #[arg(
        short = 'O',
        long = "opt-level",
        default_value = "1",
        value_name = "LEVEL"
    )]
    pub opt_level: OptLevel,

    /// Stop compilation at a specific stage
    #[arg(long, value_name = "STAGE")]
    pub stage: Option<StageArg>,

    /// Output format
    #[arg(long, default_value = "mlir", value_name = "FORMAT")]
    pub emit: EmitFormatArg,

    /// Pipeline profile shortcut (mirrors `ais-opt`/`ais-translate` behavior)
    #[arg(long, value_enum, default_value = "default")]
    pub profile: CompileProfileArg,

    /// Run specific passes (can be specified multiple times)
    #[arg(long = "pass", value_name = "PASS")]
    pub passes: Vec<String>,

    /// Skip specific passes (can be specified multiple times)
    #[arg(long = "no-pass", value_name = "PASS")]
    pub skip_passes: Vec<String>,

    /// List all available passes and exit
    #[arg(long)]
    pub list_passes: bool,

    /// Print the pass pipeline that would be run
    #[arg(long)]
    pub print_pipeline: bool,

    /// Print IR after each pass
    #[arg(long)]
    pub dump_ir: bool,

    /// Verify module after each optimization pass
    #[arg(long)]
    pub verify_each: bool,

    /// Print timing information
    #[arg(long)]
    pub timing: bool,

    /// Agent name (defaults to input filename)
    #[arg(long, value_name = "NAME")]
    pub name: Option<String>,
}

/// Optimization level enum for CLI
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OptLevel {
    #[value(name = "0", alias = "none")]
    None,
    #[value(name = "1", alias = "basic")]
    Basic,
    #[value(name = "2", alias = "standard")]
    Standard,
    #[value(name = "3", alias = "aggressive")]
    Aggressive,
}

impl From<OptLevel> for OptimizationLevel {
    fn from(level: OptLevel) -> Self {
        match level {
            OptLevel::None => OptimizationLevel::O0,
            OptLevel::Basic => OptimizationLevel::O1,
            OptLevel::Standard => OptimizationLevel::O2,
            OptLevel::Aggressive => OptimizationLevel::O3,
        }
    }
}

/// CLI-friendly handle for compilation stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum StageArg {
    /// Parse only (DSL → MLIR)
    Parse,
    /// Parse and run optimization passes
    Optimize,
    /// Full lowering (AIS → async)
    Lower,
}

impl From<StageArg> for CompilationStage {
    fn from(arg: StageArg) -> Self {
        match arg {
            StageArg::Parse => CompilationStage::Parse,
            StageArg::Optimize => CompilationStage::Optimize,
            StageArg::Lower => CompilationStage::Lower,
        }
    }
}

/// CLI-friendly handle for output emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EmitFormatArg {
    /// Raw MLIR text
    Mlir,
    /// Optimized MLIR text
    Optimized,
    /// Async-lowered MLIR text
    Async,
    /// JSON representation
    Json,
    /// Rust source
    Rust,
}

impl From<EmitFormatArg> for EmitFormat {
    fn from(arg: EmitFormatArg) -> Self {
        match arg {
            EmitFormatArg::Mlir => EmitFormat::Mlir,
            EmitFormatArg::Optimized => EmitFormat::Optimized,
            EmitFormatArg::Async => EmitFormat::Async,
            EmitFormatArg::Json => EmitFormat::Json,
            EmitFormatArg::Rust => EmitFormat::Rust,
        }
    }
}

/// Predefined compile profiles that mirror historical tooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CompileProfileArg {
    /// Standard pipeline (parse → binary).
    Default,
    /// Optimization-focused run (like `ais-opt`).
    Opt,
    /// Translation-focused run (like `ais-translate`).
    Translate,
}

pub fn selected_stage(stage: Option<StageArg>) -> CompilationStage {
    stage.map(Into::into).unwrap_or_default()
}
