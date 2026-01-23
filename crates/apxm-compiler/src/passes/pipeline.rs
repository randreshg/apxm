//! Pipeline builder for the passes.

use super::PassManager;
use apxm_core::error::compiler::Result;
use apxm_core::types::OptimizationLevel;

pub fn build_pipeline(pm: &mut PassManager, level: OptimizationLevel) -> Result<()> {
    match level {
        OptimizationLevel::O0 => {
            // No optimization passes at O0
        }
        OptimizationLevel::O1 | OptimizationLevel::O2 | OptimizationLevel::O3 => {
            // Normalize first to deduplicate context operands and properly link IR
            // Build prompts for operations with empty template_str but non-empty context
            // Then run analysis passes to warn about potential issues
            pm.normalize()?
                .build_prompt()?
                .unconsumed_value_warning()?
                .scheduling()?
                .fuse_ask_ops()?
                .canonicalizer()?
                .cse()?
                .symbol_dce()?;
        }
    }
    Ok(())
}
