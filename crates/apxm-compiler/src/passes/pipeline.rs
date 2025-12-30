//! Pipeline builder for the passes.

use super::PassManager;
use apxm_core::error::compiler::Result;
use apxm_core::types::OptimizationLevel;

pub fn build_pipeline(pm: &mut PassManager, level: OptimizationLevel) -> Result<()> {
    match level {
        OptimizationLevel::O0 => {
            pm.lower_to_async()?;
        }
        OptimizationLevel::O1 | OptimizationLevel::O2 | OptimizationLevel::O3 => {
            // Run analysis passes first to warn about potential issues
            pm.unconsumed_value_warning()?;

            pm.normalize()?
                .scheduling()?
                .fuse_reasoning()?
                .canonicalizer()?
                .cse()?
                .symbol_dce()?
                .lower_to_async()?;
        }
    }
    Ok(())
}
