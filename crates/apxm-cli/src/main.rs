use std::path::PathBuf;

#[cfg(feature = "driver")]
use anyhow::Context;
use anyhow::Result;
use apxm_core::utils::build::MlirEnvReport;
#[cfg(feature = "driver")]
use apxm_driver::compiler::Compiler;
#[cfg(feature = "driver")]
use apxm_driver::{ApXmConfig, ConfigError, Linker, LinkerConfig};
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::env;

/// Initialize the tracing subscriber based on the --trace flag.
/// If no trace level is provided, no subscriber is registered (zero overhead).
#[cfg(feature = "driver")]
fn initialize_tracing(level: &Option<String>) {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = match level {
        Some(lvl) => {
            // Build filter for apxm crates at specified level
            let filter_str = format!(
                "apxm={lvl},apxm_runtime={lvl},apxm_driver={lvl},apxm_core={lvl}"
            );
            EnvFilter::try_new(&filter_str).unwrap_or_else(|_| EnvFilter::new("apxm=info"))
        }
        None => return, // No subscriber = no overhead
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(
            fmt::layer()
                .with_thread_ids(true)
                .with_target(true)
                .with_ansi(true)
                .with_writer(std::io::stderr),
        )
        .init();
}

#[derive(Parser)]
#[command(name = "apxm")]
#[command(about = "APxM CLI (minimal) - compile and run AIS/MLIR", long_about = None)]
struct Cli {
    /// Optional config path (defaults to .apxm/config.toml or ~/.apxm/config.toml)
    #[arg(long)]
    config: Option<PathBuf>,

    /// Enable runtime tracing (levels: trace, debug, info, warn, error)
    #[arg(long, global = true)]
    trace: Option<String>,

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
        /// Emit diagnostics JSON file with compilation statistics
        #[arg(long)]
        emit_diagnostics: Option<PathBuf>,
        /// Optimization level (0 = no optimizations, 1-3 = increasing optimization)
        #[arg(short = 'O', long = "opt-level", default_value = "1")]
        opt_level: u8,
    },
    /// Compile and execute a DSL/MLIR file through the runtime
    #[command(trailing_var_arg = true)]
    Execute {
        /// Input file (.ais or .mlir)
        input: PathBuf,
        /// Arguments to pass to the entry flow
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
        /// Treat input as MLIR
        #[arg(long)]
        mlir: bool,
        /// Optimization level (0 = no optimizations, 1-3 = increasing optimization)
        /// -O0 disables FuseReasoning, -O1+ enables it
        #[arg(short = 'O', long = "opt-level", default_value = "1")]
        opt_level: u8,
        /// Emit metrics JSON file with runtime execution statistics
        #[arg(long)]
        emit_metrics: Option<PathBuf>,
    },
    /// Run a pre-compiled artifact (.apxmobj)
    Run {
        /// Input artifact file (.apxmobj)
        input: PathBuf,
        /// Arguments to pass to the entry flow
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
        /// Emit metrics JSON file with runtime execution statistics
        #[arg(long)]
        emit_metrics: Option<PathBuf>,
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

    // Initialize tracing if --trace flag is provided
    initialize_tracing(&cli.trace);

    match cli.command {
        Commands::Compile {
            input,
            mlir,
            output,
            emit_diagnostics,
            opt_level,
        } => compile_command(input, mlir, output, emit_diagnostics, opt_level),
        Commands::Execute {
            input,
            args,
            mlir,
            opt_level,
            emit_metrics,
        } => execute_command(input, args, mlir, opt_level, cli.config, emit_metrics).await,
        Commands::Run {
            input,
            args,
            emit_metrics,
        } => run_command(input, args, cli.config, emit_metrics).await,
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
            println!(
                "export MLIR_DIR={}",
                prefix.join("lib/cmake/mlir").display()
            );
            println!(
                "export LLVM_DIR={}",
                prefix.join("lib/cmake/llvm").display()
            );
            println!("export MLIR_PREFIX={}", prefix.display());
            println!("export LLVM_PREFIX={}", prefix.display());
            println!("export PATH={}/bin:$PATH", prefix.display());
        }
        "fish" => {
            println!(
                "set -gx MLIR_DIR {}",
                prefix.join("lib/cmake/mlir").display()
            );
            println!(
                "set -gx LLVM_DIR {}",
                prefix.join("lib/cmake/llvm").display()
            );
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
fn compile_command(
    input: PathBuf,
    mlir: bool,
    output: Option<PathBuf>,
    emit_diagnostics: Option<PathBuf>,
    opt_level: u8,
) -> Result<()> {
    use apxm_core::types::OptimizationLevel;

    let opt = match opt_level {
        0 => OptimizationLevel::O0,
        1 => OptimizationLevel::O1,
        2 => OptimizationLevel::O2,
        _ => OptimizationLevel::O3,
    };

    // Read source for error display
    let source = std::fs::read_to_string(&input)
        .with_context(|| format!("Failed to read {}", input.display()))?;

    let compile_start = std::time::Instant::now();
    let compiler = Compiler::with_opt_level(opt).context("Failed to initialize compiler")?;
    let module = match compiler.compile(&input, mlir) {
        Ok(m) => m,
        Err(err) => {
            // Pretty-print compilation errors with source context
            eprintln!("{}", format_driver_error(&err, &source));
            return Err(anyhow::anyhow!("Compilation failed"));
        }
    };
    let compile_time = compile_start.elapsed();

    let artifact_start = std::time::Instant::now();
    let bytes = module
        .generate_artifact_bytes()
        .context("Failed to generate artifact")?;
    let artifact_time = artifact_start.elapsed();

    let out_path = output.unwrap_or_else(|| input.with_extension("apxmobj"));
    std::fs::write(&out_path, &bytes)
        .with_context(|| format!("Failed to write {}", out_path.display()))?;

    // Emit diagnostics if requested
    if let Some(diag_path) = emit_diagnostics {
        let artifact = apxm_artifact::Artifact::from_bytes(&bytes)
            .map_err(|e| anyhow::anyhow!("Failed to parse artifact: {}", e))?;
        let dag = artifact.dag();

        let diagnostics = serde_json::json!({
            "input": input.display().to_string(),
            "optimization_level": format!("O{}", opt_level),
            "compilation_phases": {
                "total_ms": compile_time.as_secs_f64() * 1000.0,
                "artifact_gen_ms": artifact_time.as_secs_f64() * 1000.0
            },
            "dag_statistics": {
                "total_nodes": dag.nodes.len(),
                "entry_nodes": dag.entry_nodes.len(),
                "exit_nodes": dag.exit_nodes.len(),
                "total_edges": dag.edges.len()
            },
            "passes_applied": match opt_level {
                0 => vec!["lower-to-async"],
                _ => vec!["normalize", "scheduling", "fuse-ask-ops", "canonicalizer", "cse", "symbol-dce", "lower-to-async"]
            }
        });

        std::fs::write(&diag_path, serde_json::to_string_pretty(&diagnostics)?)
            .with_context(|| format!("Failed to write diagnostics to {}", diag_path.display()))?;

        println!("Wrote diagnostics to {}", diag_path.display());
    }

    println!("Wrote artifact to {}", out_path.display());
    println!(
        "Compiled in {:.2}ms, artifact generated in {:.2}ms",
        compile_time.as_secs_f64() * 1000.0,
        artifact_time.as_secs_f64() * 1000.0
    );
    Ok(())
}

#[cfg(feature = "driver")]
async fn execute_command(
    input: PathBuf,
    args: Vec<String>,
    mlir: bool,
    opt_level: u8,
    config: Option<PathBuf>,
    emit_metrics: Option<PathBuf>,
) -> Result<()> {
    use apxm_core::types::OptimizationLevel;

    // Read source for error display
    let source = std::fs::read_to_string(&input)
        .with_context(|| format!("Failed to read {}", input.display()))?;

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

    let result = match linker.run_with_args(&input, mlir, args).await {
        Ok(r) => r,
        Err(err) => {
            // Pretty-print compilation/runtime errors with source context
            eprintln!("{}", format_driver_error(&err, &source));
            return Err(anyhow::anyhow!("Execution failed"));
        }
    };


    // Emit metrics JSON if requested
    if let Some(metrics_path) = emit_metrics {
        let mut metrics_json = serde_json::json!({
            "input": input.display().to_string(),
            "optimization_level": format!("O{}", opt_level),
            "execution": {
                "nodes_executed": result.execution.stats.executed_nodes,
                "nodes_failed": result.execution.stats.failed_nodes,
                "duration_ms": result.execution.stats.duration_ms,
                "status": if result.execution.stats.failed_nodes == 0 { "success" } else { "partial_failure" }
            },
            "scheduler": result.execution.scheduler_metrics.to_json()
        });

        #[cfg(feature = "metrics")]
        {
            let llm_metrics = &result.execution.llm_metrics;
            metrics_json["llm"] = serde_json::json!({
                "total_requests": llm_metrics.total_requests,
                "total_input_tokens": llm_metrics.total_input_tokens,
                "total_output_tokens": llm_metrics.total_output_tokens,
                "avg_latency_ms": llm_metrics.average_latency.as_millis(),
                "p50_latency_ms": llm_metrics.p50_latency.as_millis(),
                "p99_latency_ms": llm_metrics.p99_latency.as_millis()
            });

            let link_metrics = &result.metrics;
            metrics_json["link_phases"] = serde_json::json!({
                "compile_ms": link_metrics.compile_time.as_secs_f64() * 1000.0,
                "artifact_ms": link_metrics.artifact_time.as_secs_f64() * 1000.0,
                "validation_ms": link_metrics.validation_time.as_secs_f64() * 1000.0,
                "runtime_ms": link_metrics.runtime_time.as_secs_f64() * 1000.0
            });
        }

        std::fs::write(&metrics_path, serde_json::to_string_pretty(&metrics_json)?)
            .with_context(|| format!("Failed to write metrics to {}", metrics_path.display()))?;

        println!("Wrote metrics to {}", metrics_path.display());
    }

    // Print workflow outputs
    if result.execution.results.is_empty() {
        eprintln!("Warning: No output values");
    } else {
        for (key, value) in &result.execution.results {
            if result.execution.results.len() == 1 {
                // Single result: print just the value
                println!("{}", value);
            } else {
                // Multiple results: print key=value pairs
                println!("{}={}", key, value);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "driver")]
async fn run_command(
    input: PathBuf,
    args: Vec<String>,
    config: Option<PathBuf>,
    emit_metrics: Option<PathBuf>,
) -> Result<()> {
    use apxm_artifact::Artifact;
    use apxm_driver::runtime::RuntimeExecutor;

    // Validate file extension
    if input.extension().and_then(|e| e.to_str()) != Some("apxmobj") {
        return Err(anyhow::anyhow!(
            "Expected .apxmobj artifact file. Use 'execute' command for .ais source files."
        ));
    }

    // Load artifact
    let artifact_bytes = std::fs::read(&input)
        .with_context(|| format!("Failed to read artifact {}", input.display()))?;
    let artifact = Artifact::from_bytes(&artifact_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse artifact: {}", e))?;

    // Initialize runtime
    let apxm_config = load_config(config).context("Failed to load configuration")?;
    let linker_config = LinkerConfig::from_apxm_config(apxm_config);
    let runtime = RuntimeExecutor::new(&linker_config)
        .await
        .context("Failed to initialize runtime")?;

    // Execute artifact with args
    let result = runtime
        .execute_artifact_with_args(artifact, args)
        .await
        .map_err(|e| anyhow::anyhow!("Execution failed: {}", e))?;

    // Emit metrics JSON if requested
    if let Some(metrics_path) = emit_metrics {
        let metrics_json = serde_json::json!({
            "input": input.display().to_string(),
            "execution": {
                "nodes_executed": result.stats.executed_nodes,
                "nodes_failed": result.stats.failed_nodes,
                "duration_ms": result.stats.duration_ms,
                "status": if result.stats.failed_nodes == 0 { "success" } else { "partial_failure" }
            },
            "scheduler": result.scheduler_metrics.to_json()
        });

        std::fs::write(&metrics_path, serde_json::to_string_pretty(&metrics_json)?)
            .with_context(|| format!("Failed to write metrics to {}", metrics_path.display()))?;

        println!("Wrote metrics to {}", metrics_path.display());
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
    let mlir_tblgen = conda_bin.as_ref().map(|p| p.join("mlir-tblgen"));
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
        if has_tblgen {
            Status::Ok
        } else {
            Status::Error
        },
        if has_tblgen { "found" } else { "missing" },
    );
    print_status_line(
        "cmake/mlir",
        if has_mlir_cmake {
            Status::Ok
        } else {
            Status::Error
        },
        if has_mlir_cmake { "found" } else { "missing" },
    );
    print_status_line(
        "cmake/llvm",
        if has_llvm_cmake {
            Status::Ok
        } else {
            Status::Error
        },
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
    println!("{:<14} [{}] {}", label.bold(), status_str, value);
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

/// Format a driver error with pretty-printed source context.
///
/// This extracts the underlying compiler error and uses `pretty_print()` to
/// display rustc-style error messages with line numbers and source snippets.
#[cfg(feature = "driver")]
fn format_driver_error(err: &apxm_driver::DriverError, source: &str) -> String {
    use apxm_driver::DriverError;

    match err {
        DriverError::Compiler(compiler_err) => compiler_err.pretty_print(Some(source)),
        other => format!("{other}"),
    }
}
