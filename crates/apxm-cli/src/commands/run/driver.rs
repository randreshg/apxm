use std::fs;
use std::path::PathBuf;

use apxm_config::ApXmConfig;
use apxm_core::error::cli::{CliError, CliResult};
use apxm_core::log_info;
use apxm_linker::error::LinkerError;
use apxm_linker::{LinkResult, Linker, LinkerConfig};

use super::args::RunArgs;

async fn load_config(path: Option<PathBuf>) -> Result<ApXmConfig, CliError> {
    match path {
        Some(path) => ApXmConfig::from_file(&path).map_err(|err| CliError::Config {
            message: err.to_string(),
        }),
        None => Ok(ApXmConfig::load_scoped().unwrap_or_default()),
    }
}

pub async fn execute(args: RunArgs, config: Option<PathBuf>) -> CliResult<()> {
    let apxm_config = load_config(config).await?;
    let linker_config = LinkerConfig::from_apxm_config(apxm_config);
    let linker = Linker::new(linker_config)
        .await
        .map_err(linker_error_to_cli)?;

    let LinkResult {
        module,
        artifact,
        execution,
    } = linker
        .run(&args.input, args.mlir)
        .await
        .map_err(linker_error_to_cli)?;

    log_info!(
        "run",
        "Linker compiled module from {}",
        args.input.display()
    );
    if let Ok(text) = module.to_string() {
        log_info!("run", "Module IR size: {} bytes", text.len());
    }

    if let Ok(hash) = artifact.payload_hash() {
        let hash_str: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        log_info!("run", "Artifact payload hash: {}", hash_str);
    }
    if let Some(name) = &artifact.metadata().module_name {
        log_info!("run", "Artifact module: {}", name);
    }
    log_info!(
        "run",
        "Executed {} nodes in {} ms",
        execution.stats.executed_nodes,
        execution.stats.duration_ms
    );

    if let Some(path) = args.emit_artifact.as_ref() {
        let bytes = artifact.to_bytes().map_err(|e| CliError::Runtime {
            message: format!("Failed to encode artifact: {e}"),
        })?;
        fs::write(path, &bytes).map_err(CliError::Io)?;
        log_info!("run", "Wrote artifact to {}", path.display());
    }

    if let Some(path) = args.emit_rust.as_ref() {
        let code = module
            .generate_rust_code()
            .map_err(CliError::from_compiler_error)?;
        fs::write(path, code).map_err(CliError::Io)?;
        log_info!("run", "Wrote Rust source to {}", path.display());
    }

    Ok(())
}

fn linker_error_to_cli(err: LinkerError) -> CliError {
    match err {
        LinkerError::Compiler(source) => CliError::from_compiler_error(source),
        LinkerError::Runtime(source) => CliError::Runtime {
            message: source.to_string(),
        },
        LinkerError::Io(io) => CliError::Io(io),
        LinkerError::Config(msg) => CliError::Config { message: msg },
    }
}
