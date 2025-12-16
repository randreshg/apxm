//! APXM CLI - Command-line interface for Agent Programming eXecution Model.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

mod commands;

use apxm_core::error::cli::CliError;

/// APXM: Agent Programming eXecution Model
///
/// A compiler and runtime for agentic AI programs using MLIR.
#[derive(Debug, Parser)]
#[command(name = "apxm")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Enable verbose output (can be repeated: -v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Suppress all output except errors.
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Configuration file path.
    #[arg(short, long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Disable colored output.
    #[arg(long, global = true)]
    pub no_color: bool,
}

/// Available commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Compile MLIR source to AgentBinary.
    ///
    /// Takes MLIR text (in the AIS dialect) and compiles it through
    /// the optimization pipeline to produce an AgentBinary.
    #[command(visible_alias = "c")]
    Compile(commands::CompileArgs),

    /// Run a compiled agent (not yet implemented).
    ///
    /// Executes a compiled AgentBinary using the APXM runtime.
    #[command(visible_alias = "r")]
    // Run {
    //     /// Input AgentBinary file.
    //     input: PathBuf,
    // },

    /// Start interactive chat interface.
    ///
    /// Launches a chat-like interface for interacting with APXM agents,
    /// allowing natural language conversations.
    //Chat(commands::ChatArgs),

    /// Start interactive REPL.
    ///
    /// Launches an interactive Read-Eval-Print Loop for
    /// experimenting with APXM programs.
    //Repl(commands::ReplArgs),

    /// Check installation and dependencies.
    ///
    /// Verifies that all required dependencies (MLIR, LLVM) are
    /// properly installed and configured.
    //Doctor,

    /// Show version information.
    Version,
}

fn print_cli_error(e: &CliError) {
    let error = match e {
        CliError::Compiler {
            source: Some(err), ..
        } => err.as_error(),
        CliError::Compilation {
            source: Some(err), ..
        } => err.as_error(),
        _ => None,
    };

    if let Some(err) = error {
        let src = std::fs::read_to_string(&err.primary_span.file).ok();
        eprintln!("{}", err.pretty_print(src.as_deref()));
    } else {
        eprintln!("{}", e);
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    // Execute command
    let result = match cli.command {
        Command::Compile(args) => commands::compile::execute(args),
        Command::Version => {
            print_version();
            Ok(())
        }
    };

    result.map(|_| ExitCode::SUCCESS).unwrap_or_else(|e| {
        print_cli_error(&e);
        ExitCode::from(e.exit_code() as u8)
    })
}

/// Print version information.
fn print_version() {
    println!("apxm {}", env!("CARGO_PKG_VERSION"));
    println!("apxm-compiler {}", apxm_compiler::VERSION);
    println!();
    println!("Target: {}", std::env::consts::ARCH);
    println!("OS: {}", std::env::consts::OS);
}
