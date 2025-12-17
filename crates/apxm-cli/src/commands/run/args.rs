use clap::Args;
use std::path::PathBuf;

/// Arguments for the `run` command.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// Agent source file (AIS or MLIR).
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Treat input as raw MLIR instead of DSL.
    #[arg(long)]
    pub mlir: bool,

    /// Write the compiled artifact (.apxmobj) to this path.
    #[arg(long, value_name = "FILE")]
    pub emit_artifact: Option<PathBuf>,

    /// Emit the generated Rust code for debugging.
    #[arg(long, value_name = "FILE")]
    pub emit_rust: Option<PathBuf>,
}
