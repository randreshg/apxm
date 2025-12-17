use super::args::CompileArgs;
use super::output::output_result;
use super::pipeline::{build_pass_manager, execute_list_passes, print_pipeline};
use super::plan::resolve_plan;
use apxm_compiler::{Context, Module};
use apxm_core::error::cli::{CliError, CliResult};
use apxm_core::{log_error, log_info};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

pub fn execute(args: CompileArgs) -> CliResult<()> {
    if args.list_passes {
        return execute_list_passes();
    }

    if args.print_pipeline {
        let plan = resolve_plan(&args)?;
        return print_pipeline(&args, plan.stage);
    }

    let input = args.input.as_ref().ok_or_else(|| CliError::Config {
        message: "Input file is required (use --list-passes for pass info)".to_string(),
    })?;

    if !input.exists() {
        return Err(CliError::InputNotFound {
            path: input.clone(),
        });
    }

    let start_time = Instant::now();
    let plan = resolve_plan(&args)?;
    let stage = plan.stage;
    let emit = plan.emit;

    let agent_name = args.name.clone().unwrap_or_else(|| {
        input
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "agent".to_string())
    });

    log_info!(
        "compile",
        "Compiling {} as '{}'",
        input.display(),
        agent_name
    );

    let ctx = Context::new().map_err(|e| CliError::Config {
        message: format!("Failed to initialize compiler context: {}", e),
    })?;

    let module = parse_input(&ctx, input, args.mlir)?;

    dump_mlir_if_requested(&module, args.dump_parsed_mlir.as_ref(), "parsed MLIR")?;

    if args.dump_ir {
        log_info!("compile", "=== After parsing ===");
        match module.to_string() {
            Ok(text) => log_info!("compile", "{}", text),
            Err(e) => log_error!("compile", "(failed to render module: {})", e),
        }
    }

    let pm = build_pass_manager(&ctx, &args, stage)?;
    pm.run(&module).map_err(CliError::from_compiler_error)?;

    dump_mlir_if_requested(&module, args.dump_optimized_mlir.as_ref(), "optimized MLIR")?;

    if args.dump_ir {
        log_info!("compile", "=== After optimization ===");
        match module.to_string() {
            Ok(text) => log_info!("compile", "{}", text),
            Err(e) => log_error!("compile", "(failed to render module: {})", e),
        }
    }

    output_result(&args, &module, &agent_name, emit)?;

    if args.timing {
        let elapsed = start_time.elapsed();
        log_info!(
            "compile",
            "Compilation completed in {:.3}s",
            elapsed.as_secs_f64()
        );
    }

    Ok(())
}

fn parse_input(ctx: &Context, path: &PathBuf, is_mlir: bool) -> CliResult<Module> {
    if is_mlir || path.extension().map(|e| e == "mlir").unwrap_or(false) {
        Module::parse_file(ctx, path).map_err(CliError::from_compiler_error)
    } else {
        let source =
            fs::read_to_string(path).map_err(|_| CliError::InputNotFound { path: path.clone() })?;
        let filename = path.to_string_lossy().to_string();
        Module::parse_dsl(ctx, &source, &filename).map_err(CliError::from_compiler_error)
    }
}

fn dump_mlir_if_requested(module: &Module, path: Option<&PathBuf>, label: &str) -> CliResult<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let mlir = module.to_string().map_err(|e| CliError::Config {
        message: format!("Failed to render module: {}", e),
    })?;
    std::fs::write(path, mlir.as_bytes()).map_err(|e| CliError::OutputWrite {
        path: path.clone(),
        message: e.to_string(),
    })?;
    log_info!("compile", "Dumped {} to {}", label, path.display());
    Ok(())
}
