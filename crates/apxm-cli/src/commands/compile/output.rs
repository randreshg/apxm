use super::args::CompileArgs;
use apxm_compiler::Module;
use apxm_core::error::cli::{CliError, CliResult};
use apxm_core::log_info;
use apxm_core::types::EmitFormat;
use std::io::{self, Write};
use std::path::PathBuf;

pub fn output_result(
    args: &CompileArgs,
    module: &Module,
    agent_name: &str,
    emit: EmitFormat,
) -> CliResult<()> {
    match (&args.output, emit) {
        (None, EmitFormat::Mlir) | (None, EmitFormat::Optimized) | (None, EmitFormat::Async) => {
            write_stdout(&module_text(module)?)?;
        }
        (None, EmitFormat::Json) => {
            write_stdout(&json_text(module, agent_name)?)?;
        }
        (None, EmitFormat::Rust) => {
            let code = module.generate_rust_code().map_err(|e| CliError::Config {
                message: format!("Failed to generate Rust source: {}", e),
            })?;
            write_stdout(&code)?;
        }
        (Some(path), EmitFormat::Mlir) => write_module_text(path, module, "MLIR")?,
        (Some(path), EmitFormat::Optimized) => write_module_text(path, module, "optimized MLIR")?,
        (Some(path), EmitFormat::Async) => write_module_text(path, module, "async MLIR")?,
        (Some(path), EmitFormat::Json) => write_json_output(path, module, agent_name)?,
        (Some(path), EmitFormat::Rust) => write_rust_output(path, module)?,
    }

    Ok(())
}

fn module_text(module: &Module) -> CliResult<String> {
    module.to_string().map_err(|e| CliError::Config {
        message: format!("Failed to render module: {}", e),
    })
}

fn json_text(module: &Module, agent_name: &str) -> CliResult<String> {
    let mlir_text = module_text(module)?;
    serde_json::to_string_pretty(&serde_json::json!({
        "name": agent_name,
        "mlir": mlir_text,
    }))
    .map_err(|e| CliError::Config {
        message: format!("Failed to serialize JSON: {}", e),
    })
}

fn write_module_text(path: &PathBuf, module: &Module, label: &str) -> CliResult<()> {
    let mlir_text = module_text(module)?;
    write_bytes(path, mlir_text.as_bytes())?;
    log_info!("compile", "Wrote {} to {}", label, path.display());
    Ok(())
}

fn write_json_output(path: &PathBuf, module: &Module, agent_name: &str) -> CliResult<()> {
    let json_str = json_text(module, agent_name)?;
    write_bytes(path, json_str.as_bytes())?;
    log_info!("compile", "Wrote JSON to {}", path.display());
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
