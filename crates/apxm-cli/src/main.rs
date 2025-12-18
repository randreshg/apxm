//! APXM CLI - Command-line interface for Agent Programming eXecution Model.

use clap::{Parser, Subcommand};
use dirs::home_dir;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex, OnceLock};

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

    /// Run a compiled agent.
    ///
    /// Executes a compiled AIS program by invoking the compiler and runtime linker.
    #[command(visible_alias = "r")]
    Run(commands::RunArgs),

    /// Start interactive chat interface.
    ///
    /// Launches a chat-like interface for interacting with APXM agents,
    /// allowing natural language conversations that are translated to DSL and executed.
    Chat(commands::ChatArgs),

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
    init_file_logger();
    let cli = Cli::parse();

    // Execute command
    let result = match cli.command {
        Command::Compile(args) => commands::compile::execute(args),
        Command::Run(args) => commands::run::execute(args, cli.config).await,
        Command::Chat(args) => commands::chat::execute(args, cli.config).await,
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

fn init_file_logger() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        if let Some(writer) = make_log_writer() {
            let _ = tracing_subscriber::fmt()
                .with_ansi(false)
                .with_target(true)
                .with_level(true)
                .with_writer(move || LogWriter {
                    inner: writer.clone(),
                })
                .try_init();
        }
    });
}

fn make_log_writer() -> Option<Arc<Mutex<std::fs::File>>> {
    let home = home_dir()?;
    let logs_dir = home.join(".apxm").join("logs");
    std::fs::create_dir_all(&logs_dir).ok()?;
    let file_path = logs_dir.join("apxm.log");
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)
        .ok()?;
    Some(Arc::new(Mutex::new(file)))
}

struct LogWriter {
    inner: Arc<Mutex<std::fs::File>>,
}

impl Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut guard = self.inner.lock().unwrap();
        guard.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut guard = self.inner.lock().unwrap();
        guard.flush()
    }
}

/// Print version information.
fn print_version() {
    println!("apxm {}", env!("CARGO_PKG_VERSION"));
    println!("apxm-compiler {}", apxm_compiler::VERSION);
    println!();
    println!("Target: {}", std::env::consts::ARCH);
    println!("OS: {}", std::env::consts::OS);
}
