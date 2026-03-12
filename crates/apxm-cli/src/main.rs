//! APXM command-line interface.
//!
//! Provides the `apxm` binary with subcommands for building, compiling,
//! running agent workflows, environment setup (`install`, `doctor`), and
//! LLM credential management (`register`).

use std::path::PathBuf;

#[cfg(feature = "driver")]
use anyhow::Context;
use anyhow::Result;
use apxm_core::utils::build::MlirEnvReport;
use apxm_credentials::CredentialStore;
use apxm_credentials::credential::Credential;
#[cfg(feature = "driver")]
use apxm_driver::compiler::Compiler;
#[cfg(feature = "driver")]
use apxm_driver::{ApXmConfig, ConfigError, Linker, LinkerConfig};
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::collections::BTreeMap;
use std::env;

/// Initialize the tracing subscriber based on the --trace flag.
/// If no trace level is provided, no subscriber is registered (zero overhead).
#[cfg(feature = "driver")]
fn initialize_tracing(level: &Option<String>) {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = match level {
        Some(lvl) => {
            // Build filter for apxm crates at specified level
            let filter_str =
                format!("apxm={lvl},apxm_runtime={lvl},apxm_driver={lvl},apxm_core={lvl}");
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
#[command(about = "APxM CLI (minimal) - compile and run ApxmGraph inputs", long_about = None)]
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
    /// Compile ApxmGraph JSON/binary to an artifact
    Compile {
        /// Input graph file (.json or JSON-encoded binary)
        input: PathBuf,
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
    /// Compile and execute an ApxmGraph file through the runtime
    #[command(trailing_var_arg = true)]
    Execute {
        /// Input graph file (.json or JSON-encoded binary)
        input: PathBuf,
        /// Arguments to pass to the entry flow
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
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
    /// Manage LLM provider credentials
    Register {
        #[command(subcommand)]
        action: RegisterAction,
    },
}

#[derive(Subcommand)]
enum RegisterAction {
    /// Register a new LLM provider credential
    Add {
        /// Credential name (e.g., "my-openai")
        name: String,
        /// Provider type (openai, anthropic, google, openrouter, ollama)
        #[arg(long)]
        provider: String,
        /// API key (omit to enter interactively)
        #[arg(long)]
        api_key: Option<String>,
        /// Base URL override
        #[arg(long)]
        base_url: Option<String>,
        /// Default model
        #[arg(long)]
        model: Option<String>,
        /// Extra headers as key=value pairs
        #[arg(long, value_parser = parse_header)]
        header: Vec<(String, String)>,
    },
    /// List registered credentials
    List,
    /// Remove a credential
    Remove {
        /// Name of the credential to remove
        name: String,
    },
    /// Validate a credential by making a test API call
    Test {
        /// Name of the credential to test (omit to test all)
        name: Option<String>,
    },
}

fn parse_header(s: &str) -> Result<(String, String), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid header: no '=' found in '{s}'"))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}

#[cfg(feature = "driver")]
#[tokio::main]
async fn main() -> Result<()> {
    run_cli().await
}

#[cfg(not(feature = "driver"))]
fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_cli_no_driver())
}

#[cfg(feature = "driver")]
async fn run_cli() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing if --trace flag is provided
    initialize_tracing(&cli.trace);

    match cli.command {
        Commands::Compile {
            input,
            output,
            emit_diagnostics,
            opt_level,
        } => compile_command(input, output, emit_diagnostics, opt_level),
        Commands::Execute {
            input,
            args,
            opt_level,
            emit_metrics,
        } => execute_command(input, args, opt_level, cli.config, emit_metrics).await,
        Commands::Run {
            input,
            args,
            emit_metrics,
        } => run_command(input, args, cli.config, emit_metrics).await,
        Commands::Doctor => doctor_command(cli.config),
        Commands::Activate { shell } => activate_command(&shell),
        Commands::Install => install_command(),
        Commands::Register { action } => register_command(action).await,
    }
}

#[cfg(not(feature = "driver"))]
async fn run_cli_no_driver() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Doctor => doctor_command(cli.config),
        Commands::Activate { shell } => activate_command(&shell),
        Commands::Install => install_command(),
        Commands::Register { action } => register_command(action).await,
        _ => Err(anyhow::anyhow!(
            "Command requires the `driver` feature. Re-run with: cargo run -p apxm-cli --features driver -- <command>"
        )),
    }
}

fn activate_command(shell: &str) -> Result<()> {
    let prefix = detect_conda_prefix().ok_or_else(|| {
        anyhow::anyhow!("Could not detect conda prefix. Activate your env or install it first.")
    })?;

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
    output: Option<PathBuf>,
    emit_diagnostics: Option<PathBuf>,
    opt_level: u8,
) -> Result<()> {
    use apxm_core::constants::diagnostics;
    use apxm_core::types::OptimizationLevel;

    let opt = match opt_level {
        0 => OptimizationLevel::O0,
        1 => OptimizationLevel::O1,
        2 => OptimizationLevel::O2,
        _ => OptimizationLevel::O3,
    };

    let compile_start = std::time::Instant::now();
    let compiler = Compiler::with_opt_level(opt).context("Failed to initialize compiler")?;
    let graph = compiler
        .load_graph(&input)
        .map_err(|e| anyhow::anyhow!("Failed to parse graph: {e}"))?;
    let module = compiler
        .compile_graph(&graph)
        .map_err(|e| anyhow::anyhow!("Failed to compile graph: {e}"))?;
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
        let dag = artifact
            .dag()
            .ok_or_else(|| anyhow::anyhow!("Artifact contains no DAGs"))?;

        let diagnostics = serde_json::json!({
            "input": input.display().to_string(),
            "mode": diagnostics::MODE_GRAPH,
            "graph_name": graph.name,
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
                0 => Vec::<&str>::new(),
                _ => vec!["normalize", "build-prompt", "unconsumed-value-warning", "scheduling", "fuse-ask-ops", "canonicalizer", "cse", "symbol-dce"]
            }
        });

        std::fs::write(&diag_path, serde_json::to_string_pretty(&diagnostics)?)
            .with_context(|| format!("Failed to write diagnostics to {}", diag_path.display()))?;

        println!("Wrote diagnostics to {}", diag_path.display());
    }

    println!("Wrote graph artifact to {}", out_path.display());
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
    opt_level: u8,
    config: Option<PathBuf>,
    emit_metrics: Option<PathBuf>,
) -> Result<()> {
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

    let result = match linker.run_graph(&input, args).await {
        Ok(r) => r,
        Err(err) => {
            eprintln!("{}", err);
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
            "Expected .apxmobj artifact file. Use 'execute' command for graph source files."
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

async fn register_command(action: RegisterAction) -> Result<()> {
    let store = CredentialStore::open().map_err(|e| anyhow::anyhow!("{e}"))?;

    match action {
        RegisterAction::Add {
            name,
            provider,
            api_key,
            base_url,
            model,
            header,
        } => {
            let api_key = match api_key {
                Some(key) => Some(key),
                None if provider != "ollama" => {
                    eprint!("Enter API key for {name}: ");
                    let key = rpassword::read_password()
                        .map_err(|e| anyhow::anyhow!("Failed to read API key: {e}"))?;
                    if key.is_empty() { None } else { Some(key) }
                }
                None => None,
            };

            let headers: BTreeMap<String, String> = header.into_iter().collect();

            let cred = Credential {
                provider: provider.clone(),
                api_key,
                base_url,
                model,
                headers,
            };

            store.add(&name, cred).map_err(|e| anyhow::anyhow!("{e}"))?;

            print_section_header("Credential Registered");
            print_status_line("Name", Status::Ok, &name);
            print_status_line("Provider", Status::Ok, &provider);
            print_status_line("Store", Status::Ok, &store.path().display().to_string());
        }
        RegisterAction::List => {
            let creds = store.list().map_err(|e| anyhow::anyhow!("{e}"))?;
            if creds.is_empty() {
                println!("No credentials registered.");
                println!("Add one with: apxm register add <name> --provider <provider>");
                return Ok(());
            }

            print_section_header("Registered Credentials");
            for (name, summary) in &creds {
                let key_display = summary.masked_key.as_deref().unwrap_or("<none>");
                println!(
                    "  {:<16} {:<12} key={}{}{}",
                    name.bold(),
                    summary.provider,
                    key_display,
                    summary
                        .model
                        .as_ref()
                        .map(|m| format!("  model={m}"))
                        .unwrap_or_default(),
                    if summary.header_count > 0 {
                        format!("  +{} headers", summary.header_count)
                    } else {
                        String::new()
                    }
                );
            }
            println!();
            println!("Store: {}", store.path().display());
        }
        RegisterAction::Remove { name } => {
            store.remove(&name).map_err(|e| anyhow::anyhow!("{e}"))?;
            print_section_header("Credential Removed");
            print_status_line(&name, Status::Ok, "removed");
        }
        RegisterAction::Test { name } => {
            let creds_to_test: Vec<(String, Credential)> = match name {
                Some(ref n) => {
                    let cred = store
                        .get(n)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                        .ok_or_else(|| anyhow::anyhow!("Credential '{n}' not found"))?;
                    vec![(n.clone(), cred)]
                }
                None => store.list_all().map_err(|e| anyhow::anyhow!("{e}"))?,
            };

            if creds_to_test.is_empty() {
                println!("No credentials to test.");
                return Ok(());
            }

            print_section_header("Testing Credentials");
            let mut all_ok = true;
            for (cname, cred) in &creds_to_test {
                match apxm_credentials::validate::validate_credential(cname, cred).await {
                    Ok(msg) => print_status_line(cname, Status::Ok, &msg),
                    Err(e) => {
                        print_status_line(cname, Status::Error, &e.to_string());
                        all_ok = false;
                    }
                }
            }
            if !all_ok {
                return Err(anyhow::anyhow!("Some credentials failed validation"));
            }
        }
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

/// Auto-detect the conda prefix for the `apxm` environment.
///
/// Resolution order:
/// 1. `CONDA_PREFIX` env var (if it points to an apxm env or contains lib/cmake/mlir)
/// 2. `conda info --envs --json` output (looks for an env named "apxm")
/// 3. Common paths: ~/miniforge3/envs/apxm, ~/mambaforge/envs/apxm, ~/miniconda3/envs/apxm
fn detect_conda_prefix() -> Option<PathBuf> {
    // 1. Check CONDA_PREFIX env var
    if let Ok(prefix) = env::var("CONDA_PREFIX") {
        let p = PathBuf::from(&prefix);
        if p.is_dir() {
            // Accept if it looks like an apxm env or has MLIR cmake files
            let name_ok = p.file_name().map(|n| n == "apxm").unwrap_or(false);
            let mlir_ok = p.join("lib/cmake/mlir").is_dir();
            if name_ok || mlir_ok {
                return Some(p);
            }
            // Even if it doesn't match our heuristics, honour the explicit env var
            return Some(p);
        }
    }

    // 2. Try `conda info --envs --json`
    if let Ok(output) = std::process::Command::new("conda")
        .args(["info", "--envs", "--json"])
        .output()
    {
        if output.status.success() {
            if let Ok(text) = String::from_utf8(output.stdout) {
                // Minimal JSON parsing: look for paths ending in /apxm
                for line in text.lines() {
                    let trimmed = line.trim().trim_matches('"').trim_end_matches(',');
                    let candidate = PathBuf::from(trimmed);
                    if candidate.file_name().map(|n| n == "apxm").unwrap_or(false)
                        && candidate.is_dir()
                    {
                        return Some(candidate);
                    }
                }
            }
        }
    }

    // 3. Check common paths
    if let Some(home) = dirs::home_dir() {
        let candidates = [
            home.join("miniforge3/envs/apxm"),
            home.join("mambaforge/envs/apxm"),
            home.join("miniconda3/envs/apxm"),
        ];
        for candidate in &candidates {
            if candidate.is_dir() {
                return Some(candidate.clone());
            }
        }
    }

    None
}

fn print_minimal_mlir_status() {
    let conda_prefix = detect_conda_prefix();
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
    println!();
    println!("  {}", title.bold().cyan());
    println!("  {}", "\u{2500}".repeat(title.len()).dimmed());
}

fn print_subsection_header(title: &str) {
    println!();
    println!("  {}", title.bold());
}

fn print_hint(message: &str) {
    println!("  {} {}", "\u{2139}".cyan(), message);
}

enum Status {
    Ok,
    Error,
}

fn print_status_line(label: &str, status: Status, value: &str) {
    let (icon, status_str) = match status {
        Status::Ok => ("\u{2713}".green(), "OK".green().bold()),
        Status::Error => ("\u{2717}".red(), "MISSING".red().bold()),
    };
    println!("  {} {:<14} [{}] {}", icon, label.bold(), status_str, value);
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
