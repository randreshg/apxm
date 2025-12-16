use super::args::{CompileArgs, CompileProfileArg, selected_stage};
use apxm_core::error::cli::{CliError, CliResult};
use apxm_core::types::{CompilationStage, EmitFormat, stage_rank};

pub struct CompilePlan {
    pub stage: CompilationStage,
    pub emit: EmitFormat,
}

pub fn resolve_plan(args: &CompileArgs) -> CliResult<CompilePlan> {
    let mut stage = selected_stage(args.stage);
    let mut emit = EmitFormat::from(args.emit);

    match args.profile {
        CompileProfileArg::Default => {}
        CompileProfileArg::Opt => {
            if args.stage.is_none() {
                stage = CompilationStage::Optimize;
            }
            emit = EmitFormat::Optimized;
        }
        CompileProfileArg::Translate => {
            if args.stage.is_none() {
                stage = CompilationStage::Lower;
            }
            emit = EmitFormat::Async;
        }
    }

    let required_stage = emit.required_stage();
    if stage_rank(stage) < stage_rank(required_stage) {
        if args.stage.is_some() {
            return Err(CliError::Config {
                message: format!(
                    "Emit format '{:?}' requires at least '{:?}' compilation stage, but '{:?}' was requested. \
                     Use '--stage {}' or a different emit format.",
                    emit, required_stage, stage,
                    format!("{:?}", required_stage).to_lowercase()
                ),
            });
        }
        stage = required_stage;
    }

    Ok(CompilePlan { stage, emit })
}
