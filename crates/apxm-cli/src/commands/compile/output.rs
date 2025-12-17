use super::args::CompileArgs;
use apxm_compiler::Module;
use apxm_core::error::cli::{CliError, CliResult};
use apxm_core::log_info;
use apxm_core::paths::ApxmPaths;
use apxm_core::types::EmitFormat;
use std::io::{self, Write};
use std::path::PathBuf;

pub fn output_result(
    args: &CompileArgs,
    module: &Module,
    agent_name: &str,
    emit: EmitFormat,
) -> CliResult<()> {
    match emit {
        EmitFormat::Artifact => {
            let path = match args.output.clone() {
                Some(path) => path,
                None => default_artifact_path(agent_name)?,
            };
            write_artifact_output(&path, module)?;
        }
        EmitFormat::Rust => match &args.output {
            Some(path) => write_rust_output(path, module)?,
            None => {
                let code = module.generate_rust_code().map_err(|e| CliError::Config {
                    message: format!("Failed to generate Rust source: {}", e),
                })?;
                write_stdout(&code)?;
            }
        },
    }

    Ok(())
}

fn default_artifact_path(agent_name: &str) -> CliResult<PathBuf> {
    let paths = ApxmPaths::discover().map_err(|e| CliError::Config {
        message: format!("Failed to initialize .apxm directory: {e}"),
    })?;
    let artifacts_dir = paths.artifacts_dir().map_err(|e| CliError::Config {
        message: format!("Failed to create artifacts directory: {e}"),
    })?;

    let fallback = "agent";
    let trimmed = agent_name.trim();
    let raw = if trimmed.is_empty() {
        fallback
    } else {
        trimmed
    };
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let base = if sanitized.is_empty() {
        fallback
    } else {
        &sanitized
    };
    let filename = if base.ends_with(".apxmobj") {
        base.to_string()
    } else {
        format!("{base}.apxmobj")
    };

    Ok(artifacts_dir.join(filename))
}

fn write_artifact_output(path: &PathBuf, module: &Module) -> CliResult<()> {
    module
        .generate_artifact_to_path(path)
        .map_err(|e| CliError::OutputWrite {
            path: path.clone(),
            message: format!("Failed to emit artifact: {}", e),
        })?;
    log_info!("compile", "Wrote artifact to {}", path.display());
    Ok(())
}

fn write_rust_output(path: &PathBuf, module: &Module) -> CliResult<()> {
    let code = module
        .generate_rust_code()
        .map_err(|e| CliError::OutputWrite {
            path: path.clone(),
            message: format!("Failed to generate Rust source: {}", e),
        })?;
    write_bytes(path, code.as_bytes())?;
    log_info!("compile", "Wrote Rust source to {}", path.display());
    Ok(())
}

fn write_bytes(path: &PathBuf, data: &[u8]) -> CliResult<()> {
    std::fs::write(path, data).map_err(|e| CliError::OutputWrite {
        path: path.clone(),
        message: e.to_string(),
    })
}

fn write_stdout(text: &str) -> CliResult<()> {
    let mut stdout = io::stdout();
    stdout
        .write_all(text.as_bytes())
        .map_err(|e| CliError::Config {
            message: format!("Failed to write to stdout: {}", e),
        })?;
    stdout.flush().map_err(|e| CliError::Config {
        message: format!("Failed to flush stdout: {}", e),
    })
}
