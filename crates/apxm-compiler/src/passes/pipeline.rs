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
            // Normalize first to deduplicate context operands and properly link IR
            // Then run analysis passes to warn about potential issues
            pm.normalize()?
                .unconsumed_value_warning()?
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
