use std::path::PathBuf;

use anyhow::Result;
#[cfg(feature = "driver")]
use anyhow::Context;
use std::env;
use colored::Colorize;
use apxm_core::utils::build::MlirEnvReport;
#[cfg(feature = "driver")]
use apxm_driver::compiler::Compiler;
#[cfg(feature = "driver")]
use apxm_driver::{ApXmConfig, ConfigError, Linker, LinkerConfig};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "apxm")]
#[command(about = "APxM CLI (minimal) - compile and run AIS/MLIR", long_about = None)]
struct Cli {
    /// Optional config path (defaults to .apxm/config.toml or ~/.apxm/config.toml)
    #[arg(long)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile DSL/MLIR to an artifact
    Compile {
        /// Input file (.ais or .mlir)
        input: PathBuf,
        /// Treat input as MLIR
        #[arg(long)]
        mlir: bool,
        /// Output artifact path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Compile + run DSL/MLIR through the runtime
    Run {
        /// Input file (.ais or .mlir)
        input: PathBuf,
        /// Treat input as MLIR
        #[arg(long)]
        mlir: bool,
        /// Optimization level (0 = no optimizations, 1-3 = increasing optimization)
        /// -O0 disables FuseReasoning, -O1+ enables it
        #[arg(short = 'O', long = "opt-level", default_value = "1")]
        opt_level: u8,
    },
    /// Diagnose compiler/runtime dependencies
    Doctor,
    /// Print shell exports for MLIR/LLVM env setup
    Activate {
        /// Shell format (sh, zsh, bash, fish)
        #[arg(long, default_value = "sh")]
        shell: String,
    },
    /// Install or update the conda environment from environment.yaml
    Install,
}

#[cfg(feature = "driver")]
#[tokio::main]
async fn main() -> Result<()> {
    run_cli().await
}

#[cfg(not(feature = "driver"))]
fn main() -> Result<()> {
    run_cli_no_driver()
}

#[cfg(feature = "driver")]
async fn run_cli() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile { input, mlir, output } => compile_command(input, mlir, output),
        Commands::Run { input, mlir, opt_level } => run_command(input, mlir, opt_level, cli.config).await,
        Commands::Doctor => doctor_command(cli.config),
        Commands::Activate { shell } => activate_command(&shell),
        Commands::Install => install_command(),
    }
}

#[cfg(not(feature = "driver"))]
fn run_cli_no_driver() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Doctor => doctor_command(cli.config),
        Commands::Activate { shell } => activate_command(&shell),
        Commands::Install => install_command(),
        _ => Err(anyhow::anyhow!(
            "Command requires the `driver` feature. Re-run with: cargo run -p apxm-cli --features driver -- <command>"
        )),
    }
}

fn activate_command(shell: &str) -> Result<()> {
    let prefix = env::var("CONDA_PREFIX")
        .map(PathBuf::from)
        .map_err(|_| anyhow::anyhow!("CONDA_PREFIX not set. Activate your env first."))?;

    match shell {
        "sh" | "bash" | "zsh" => {
            println!("export MLIR_DIR={}", prefix.join("lib/cmake/mlir").display());
            println!("export LLVM_DIR={}", prefix.join("lib/cmake/llvm").display());
            println!("export MLIR_PREFIX={}", prefix.display());
            println!("export LLVM_PREFIX={}", prefix.display());
            println!("export PATH={}/bin:$PATH", prefix.display());
        }
        "fish" => {
            println!("set -gx MLIR_DIR {}", prefix.join("lib/cmake/mlir").display());
            println!("set -gx LLVM_DIR {}", prefix.join("lib/cmake/llvm").display());
            println!("set -gx MLIR_PREFIX {}", prefix.display());
            println!("set -gx LLVM_PREFIX {}", prefix.display());
            println!("set -gx PATH {}/bin $PATH", prefix.display());
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported shell '{}'. Use sh, bash, zsh, or fish.",
                shell
            ));
        }
    }

    Ok(())
}

fn install_command() -> Result<()> {
    print_section_header("APXM Install");

    if !command_available("mamba") {
        print_status_line("mamba", Status::Error, "not found");
        return Err(anyhow::anyhow!(
            "mamba not found in PATH. Install mamba first."
        ));
    }

    print_status_line("mamba", Status::Ok, "found");
    let installer = "mamba";

    print_status_line("env", Status::Ok, "creating/updating");

    let create_status = std::process::Command::new(installer)
        .args(["env", "create", "-f", "environment.yaml"])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run {installer}: {e}"))?;

    if !create_status.success() {
        let update_status = std::process::Command::new(installer)
            .args(["env", "update", "-f", "environment.yaml", "-n", "apxm"])
            .status()
            .map_err(|e| anyhow::anyhow!("Failed to run {installer}: {e}"))?;

        if !update_status.success() {
            return Err(anyhow::anyhow!(
                "{installer} env create/update failed. Check output."
            ));
        }
    }

    print_status_line("env", Status::Ok, "ready");
    println!();
    print_subsection_header("Next Steps");
    println!("conda activate apxm");
    println!("eval \"$(cargo run -p apxm-cli -- activate)\"");

    Ok(())
}

fn command_available(cmd: &str) -> bool {
    std::process::Command::new(cmd)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(feature = "driver")]
fn compile_command(input: PathBuf, mlir: bool, output: Option<PathBuf>) -> Result<()> {
    let compiler = Compiler::new().context("Failed to initialize compiler")?;
    let module = compiler
        .compile(&input, mlir)
        .with_context(|| format!("Failed to compile {}", input.display()))?;
    let bytes = module
        .generate_artifact_bytes()
        .context("Failed to generate artifact")?;

    let out_path = output.unwrap_or_else(|| input.with_extension("apxmobj"));
    std::fs::write(&out_path, bytes)
        .with_context(|| format!("Failed to write {}", out_path.display()))?;

    println!("Wrote artifact to {}", out_path.display());
    Ok(())
}

#[cfg(feature = "driver")]
async fn run_command(input: PathBuf, mlir: bool, opt_level: u8, config: Option<PathBuf>) -> Result<()> {
    use apxm_core::types::OptimizationLevel;

    let apxm_config = load_config(config).context("Failed to load configuration")?;
    let opt = match opt_level {
        0 => OptimizationLevel::O0,
        1 => OptimizationLevel::O1,
        2 => OptimizationLevel::O2,
        _ => OptimizationLevel::O3,
    };
    let linker_config = LinkerConfig::from_apxm_config(apxm_config).with_opt_level(opt);
    let linker = Linker::new(linker_config)
        .await
        .context("Failed to initialize runtime")?;

    let result = linker
        .run(&input, mlir)
        .await
        .with_context(|| format!("Failed to run {}", input.display()))?;

    println!(
        "Executed {} nodes in {} ms",
        result.execution.stats.total_nodes(),
        result.execution.stats.duration_ms
    );

    for (token_id, value) in result.execution.results.iter() {
        println!("token {} => {}", token_id, value);
    }

    #[cfg(feature = "metrics")]
    {
        let metrics = &result.execution.llm_metrics;
        println!(
            "LLM usage: {} input, {} output ({} total)",
            metrics.total_input_tokens,
            metrics.total_output_tokens,
            metrics.total_input_tokens + metrics.total_output_tokens
        );
        println!("LLM requests: {}", metrics.total_requests);
        println!("LLM avg latency: {:?}", metrics.average_latency);
    }

    Ok(())
}

fn doctor_command(config: Option<PathBuf>) -> Result<()> {
    let report = MlirEnvReport::detect();
    print_section_header("MLIR Doctor");
    print_minimal_mlir_status();

    if report.is_ready() {
        print_status_line("MLIR toolchain", Status::Ok, "ready");
    } else {
        print_status_line("MLIR toolchain", Status::Error, "missing");
        return Err(anyhow::anyhow!("MLIR toolchain not detected"));
    }

    #[cfg(feature = "driver")]
    {
        let _ = load_config(config).context("Failed to load configuration")?;
        print_status_line("Config", Status::Ok, "ok");
    }
    #[cfg(not(feature = "driver"))]
    let _ = config;

    Ok(())
}

fn print_minimal_mlir_status() {
    let conda_prefix = env::var("CONDA_PREFIX").ok().map(PathBuf::from);
    let conda_bin = conda_prefix.as_ref().map(|p| p.join("bin"));
    let mlir_tblgen = conda_bin
        .as_ref()
        .map(|p| p.join("mlir-tblgen"));
    let mlir_cmake = conda_prefix.as_ref().map(|p| p.join("lib/cmake/mlir"));
    let llvm_cmake = conda_prefix.as_ref().map(|p| p.join("lib/cmake/llvm"));

    match conda_prefix.as_ref() {
        Some(prefix) => {
            print_status_line("Conda prefix", Status::Ok, &prefix.display().to_string())
        }
        None => {
            print_status_line("Conda prefix", Status::Error, "<not set>");
            print_hint("Run `cargo run -p apxm-cli -- install`, then `conda activate apxm`.");
            return;
        }
    }

    let has_tblgen = mlir_tblgen.as_ref().map(|p| p.is_file()).unwrap_or(false);
    let has_mlir_cmake = mlir_cmake.as_ref().map(|p| p.is_dir()).unwrap_or(false);
    let has_llvm_cmake = llvm_cmake.as_ref().map(|p| p.is_dir()).unwrap_or(false);

    print_status_line(
        "mlir-tblgen",
        if has_tblgen { Status::Ok } else { Status::Error },
        if has_tblgen { "found" } else { "missing" },
    );
    print_status_line(
        "cmake/mlir",
        if has_mlir_cmake { Status::Ok } else { Status::Error },
        if has_mlir_cmake { "found" } else { "missing" },
    );
    print_status_line(
        "cmake/llvm",
        if has_llvm_cmake { Status::Ok } else { Status::Error },
        if has_llvm_cmake { "found" } else { "missing" },
    );

    if !(has_tblgen && has_mlir_cmake && has_llvm_cmake) {
        print_subsection_header("Suggested Fix");
        println!("cargo run -p apxm-cli -- install");
        println!("conda activate apxm");
        println!("eval \"$(cargo run -p apxm-cli -- activate)\"");
        if let Some(ref prefix) = conda_prefix {
            println!("# Or export directly from: {}", prefix.display());
        }
    }

    let _ = (has_tblgen, has_mlir_cmake, has_llvm_cmake);
}

fn print_section_header(title: &str) {
    let line = "==============================".dimmed();
    println!("{}", line);
    println!("{}", title.bold());
    println!("{}", line);
}

fn print_subsection_header(title: &str) {
    let line = "------------------------------".dimmed();
    println!("{}", line);
    println!("{}", title.bold());
    println!("{}", line);
}

fn print_hint(message: &str) {
    println!("Hint: {}", message);
}

enum Status {
    Ok,
    Error,
}

fn print_status_line(label: &str, status: Status, value: &str) {
    let status_str = match status {
        Status::Ok => "OK".green().bold(),
        Status::Error => "MISSING".red().bold(),
    };
    println!(
        "{:<14} [{}] {}",
        label.bold(),
        status_str,
        value
    );
}

#[cfg(feature = "driver")]
fn load_config(config: Option<PathBuf>) -> Result<ApXmConfig> {
    if let Some(path) = config {
        return ApXmConfig::from_file(&path)
            .map_err(|e| anyhow::anyhow!(e))
            .with_context(|| format!("Failed to load config {}", path.display()));
    }

    match ApXmConfig::load_scoped() {
        Ok(config) => Ok(config),
        Err(ConfigError::Io(err)) if err.kind() == std::io::ErrorKind::NotFound => {
            Ok(ApXmConfig::default())
        }
        Err(ConfigError::HomeDirMissing) => Ok(ApXmConfig::default()),
        Err(err) => Err(anyhow::anyhow!(err)),
    }
}
