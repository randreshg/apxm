use super::args::CompileArgs;
use apxm_compiler::{
    Context, OptimizationLevel, PassCategory, PassInfo, PassManager, find_pass, list_passes,
};
use apxm_core::error::Suggestion;
use apxm_core::error::cli::{CliError, CliResult};
use apxm_core::log_info;
use apxm_core::types::CompilationStage;

pub fn execute_list_passes() -> CliResult<()> {
    let passes = list_passes();

    log_info!("compile", "Available passes:");
    log_info!("compile", "");
    print_pass_group(
        "Analysis passes:",
        passes
            .iter()
            .filter(|p| matches!(p.category, PassCategory::Analysis))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    print_pass_group(
        "Transform passes:",
        passes
            .iter()
            .filter(|p| matches!(p.category, PassCategory::Transform))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    print_pass_group(
        "Optimization passes:",
        passes
            .iter()
            .filter(|p| matches!(p.category, PassCategory::Optimization))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    print_pass_group(
        "Lowering passes:",
        passes
            .iter()
            .filter(|p| matches!(p.category, PassCategory::Lowering))
            .collect::<Vec<_>>()
            .as_slice(),
    );

    Ok(())
}

fn print_pass_group(title: &str, passes: &[&PassInfo]) {
    if passes.is_empty() {
        return;
    }

    log_info!("compile", "{}", title);
    for pass in passes {
        log_info!("compile", "  {:<25} {}", pass.name, pass.description);
    }
    log_info!("compile", "");
}

pub fn print_pipeline(args: &CompileArgs, stage: CompilationStage) -> CliResult<()> {
    let opt_level: OptimizationLevel = args.opt_level.into();

    log_info!(
        "compile",
        "Pipeline for -O{} at stage {:?}:",
        opt_level_flag(opt_level),
        stage
    );
    log_info!("compile", "");

    if !args.passes.is_empty() {
        log_info!("compile", "Custom passes:");
        for pass in &args.passes {
            if !is_pass_skipped(pass, &args.skip_passes) {
                log_info!("compile", "  - {}", pass);
            }
        }
        return Ok(());
    }

    let pipeline = default_pipeline(stage, opt_level);
    for pass in pipeline {
        if is_pass_skipped(pass, &args.skip_passes) {
            log_info!("compile", "  - {} (skipped)", pass);
        } else {
            log_info!("compile", "  - {}", pass);
        }
    }

    Ok(())
}

pub fn build_pass_manager<'ctx>(
    ctx: &'ctx Context,
    args: &CompileArgs,
    stage: CompilationStage,
) -> CliResult<PassManager<'ctx>> {
    let pm = PassManager::new(ctx).map_err(CliError::from_compiler_error)?;

    if !args.passes.is_empty() {
        return add_custom_passes(pm, args);
    }

    let opt_level: OptimizationLevel = args.opt_level.into();
    let pipeline = default_pipeline(stage, opt_level);
    add_named_passes(pm, &pipeline, &args.skip_passes)
}

fn add_custom_passes<'ctx>(
    mut pm: PassManager<'ctx>,
    args: &CompileArgs,
) -> CliResult<PassManager<'ctx>> {
    for pass_name in &args.passes {
        if find_pass(pass_name).is_none() {
            let suggestion = suggest_similar_pass(pass_name);
            return Err(CliError::unknown_pass(pass_name.clone(), suggestion));
        }

        if is_pass_skipped(pass_name, &args.skip_passes) {
            continue;
        }

        pm.add_pass(pass_name)
            .map_err(CliError::from_compiler_error)?;
    }

    Ok(pm)
}

fn add_named_passes<'ctx>(
    mut pm: PassManager<'ctx>,
    pass_names: &[&str],
    skip_list: &[String],
) -> CliResult<PassManager<'ctx>> {
    for pass_name in pass_names {
        if is_pass_skipped(pass_name, skip_list) {
            continue;
        }

        pm.add_pass(pass_name)
            .map_err(CliError::from_compiler_error)?;
    }

    Ok(pm)
}

fn default_pipeline(stage: CompilationStage, opt_level: OptimizationLevel) -> Vec<&'static str> {
    match stage {
        CompilationStage::Parse => vec![],
        CompilationStage::Optimize => optimization_pipeline(opt_level),
        CompilationStage::Lower => {
            let mut passes = optimization_pipeline(opt_level);
            passes.extend(["inline", "lower-to-async"]);
            passes
        }
    }
}

fn optimization_pipeline(opt_level: OptimizationLevel) -> Vec<&'static str> {
    match opt_level {
        OptimizationLevel::O0 => {
            // No optimization: just structural cleanup
            vec!["canonicalizer", "cse"]
        }
        OptimizationLevel::O1 => {
            // Basic optimization: core transformations
            vec![
                "normalize",
                "scheduling",
                "fuse-reasoning",
                "canonicalizer",
                "cse",
                "symbol-dce",
            ]
        }
        OptimizationLevel::O2 => {
            // Standard optimization: more aggressive reasoning fusion
            vec![
                "normalize",
                "scheduling",
                "fuse-reasoning",
                "canonicalizer",
                "cse",
                "symbol-dce",
                "fuse-reasoning", // Second pass for deeper fusion
                "canonicalizer",
            ]
        }
        OptimizationLevel::O3 => {
            // Aggressive optimization: multiple passes for maximum optimization
            vec![
                "normalize",
                "scheduling",
                "fuse-reasoning",
                "canonicalizer",
                "cse",
                "symbol-dce",
                "fuse-reasoning", // Second fusion pass
                "scheduling",     // Re-schedule after fusion
                "canonicalizer",
                "cse",
                "symbol-dce",
                "fuse-reasoning", // Third fusion pass for deep optimization
                "canonicalizer",
            ]
        }
    }
}

fn opt_level_flag(opt_level: OptimizationLevel) -> &'static str {
    match opt_level {
        OptimizationLevel::O0 => "0",
        OptimizationLevel::O1 => "1",
        OptimizationLevel::O2 => "2",
        OptimizationLevel::O3 => "3",
    }
}

fn is_pass_skipped(pass_name: &str, skip_list: &[String]) -> bool {
    skip_list.iter().any(|skip| skip == pass_name)
}

/// Suggest similar pass names if a pass is not found.
fn suggest_similar_pass(provided_name: &str) -> Option<Suggestion> {
    let passes = list_passes();

    // Simple Levenshtein-like distance calculation
    let suggestions: Vec<_> = passes
        .iter()
        .map(|p| {
            let distance = levenshtein_distance(provided_name, &p.name);
            (p.name.clone(), distance)
        })
        .filter(|(_, dist)| *dist <= 2) // Only suggest if within edit distance of 2
        .collect::<Vec<_>>();

    if suggestions.is_empty() {
        // If no similar passes found, suggest running --list-passes
        return Some(
            Suggestion::new(
                "Run 'apxm compile --list-passes' to see all available passes".to_string(),
            )
            .with_help(
                "Available pass categories: Analysis, Transform, Optimization, Lowering"
                    .to_string(),
            ),
        );
    }

    // Get the best match (smallest distance)
    let (best_match, _) = suggestions.into_iter().min_by_key(|(_, dist)| *dist)?;

    Some(
        Suggestion::new(format!("Did you mean '{}'?", best_match))
            .with_help("Run 'apxm compile --list-passes' to see all available passes".to_string()),
    )
}

/// Calculate Levenshtein distance between two strings.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }

    // Work on the shorter string across columns to minimize memory
    let (short, long) = if a.len() <= b.len() {
        (a.as_bytes(), b.as_bytes())
    } else {
        (b.as_bytes(), a.as_bytes())
    };

    let n = short.len();
    let m = long.len();

    if n == 0 {
        return m;
    }

    // prev[j] = distance(long[..i], short[..j]) for the previous i
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for (i, &lc) in long.iter().enumerate() {
        curr[0] = i + 1;

        for (j, &sc) in short.iter().enumerate() {
            let cost = (sc != lc) as usize;

            let delete = prev[j + 1] + 1; // drop from long
            let insert = curr[j] + 1; // add to long (match short)
            let subst = prev[j] + cost; // substitute

            curr[j + 1] = delete.min(insert).min(subst);
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}
