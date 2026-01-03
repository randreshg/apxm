//! AIS Passes - Single Source of Truth
//!
//! This module contains the complete specification for all AIS compiler passes.
//! Both the compiler and the C API layer use these definitions to ensure
//! consistent pass registration across the system.
//!
//! ## Generated Files
//!
//! The `tablegen` submodule generates the following files:
//! - `Passes.generated.td` - MLIR TableGen pass definitions
//! - `PassDispatch.inc` - C API dispatch switch statement
//! - `PassDescriptors.inc` - Pass registry descriptors

mod tablegen;

pub use tablegen::{
    generate_pass_descriptors, generate_pass_dispatch, generate_passes_tablegen,
};

// ============================================================================
// Pass Categories
// ============================================================================

/// Categories of compiler passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassCategory {
    /// Domain-specific transformations (normalize, fuse-ask-ops, scheduling).
    Transform,
    /// Standard MLIR optimizations (canonicalizer, cse, symbol-dce).
    Optimization,
    /// Analysis/warning passes (unconsumed-value-warning).
    Analysis,
    /// Dialect lowering passes (lower-to-async, ais-emit-rust).
    Lowering,
}

impl PassCategory {
    /// Get the C enum value for this category.
    pub fn to_c_enum(&self) -> &'static str {
        match self {
            PassCategory::Transform => "APXM_PASS_TRANSFORM",
            PassCategory::Optimization => "APXM_PASS_OPTIMIZATION",
            PassCategory::Analysis => "APXM_PASS_ANALYSIS",
            PassCategory::Lowering => "APXM_PASS_LOWERING",
        }
    }
}

// ============================================================================
// Pass Options
// ============================================================================

/// A pass option that can be configured.
#[derive(Debug, Clone)]
pub struct PassOption {
    /// Option name (e.g., "parallel-threshold").
    pub name: &'static str,
    /// C++ variable name (e.g., "parallelThreshold").
    pub cpp_name: &'static str,
    /// Option type (e.g., "unsigned", "std::string").
    pub option_type: &'static str,
    /// Default value.
    pub default: &'static str,
    /// Description.
    pub description: &'static str,
}

impl PassOption {
    pub const fn new(
        name: &'static str,
        cpp_name: &'static str,
        option_type: &'static str,
        default: &'static str,
        description: &'static str,
    ) -> Self {
        Self {
            name,
            cpp_name,
            option_type,
            default,
            description,
        }
    }
}

// ============================================================================
// Pass Specification
// ============================================================================

/// Complete specification for a compiler pass.
#[derive(Debug, Clone)]
pub struct PassSpec {
    /// CLI name (e.g., "fuse-ask-ops").
    pub name: &'static str,
    /// TableGen class name (e.g., "FuseAskOps").
    pub class_name: &'static str,
    /// Short summary for help text.
    pub summary: &'static str,
    /// Long description for documentation.
    pub description: &'static str,
    /// Pass category.
    pub category: PassCategory,
    /// C++ constructor call.
    pub constructor: &'static str,
    /// Dependent dialects (for lowering passes).
    pub dependent_dialects: &'static [&'static str],
    /// Configurable options.
    pub options: &'static [PassOption],
    /// Whether this is a built-in MLIR pass (not generated).
    pub is_builtin: bool,
}

impl PassSpec {
    /// Create a new pass specification with minimal required fields.
    pub const fn new(
        name: &'static str,
        class_name: &'static str,
        summary: &'static str,
        description: &'static str,
        category: PassCategory,
        constructor: &'static str,
    ) -> Self {
        Self {
            name,
            class_name,
            summary,
            description,
            category,
            constructor,
            dependent_dialects: &[],
            options: &[],
            is_builtin: false,
        }
    }

    /// Builder method to add dependent dialects.
    pub const fn with_dialects(mut self, dialects: &'static [&'static str]) -> Self {
        self.dependent_dialects = dialects;
        self
    }

    /// Builder method to add options.
    pub const fn with_options(mut self, options: &'static [PassOption]) -> Self {
        self.options = options;
        self
    }

    /// Mark this as a built-in MLIR pass.
    pub const fn builtin(mut self) -> Self {
        self.is_builtin = true;
        self
    }
}

// ============================================================================
// Pass Definitions - Single Source of Truth
// ============================================================================

/// Normalize pass - canonicalizes AIS graph structure.
pub const NORMALIZE: PassSpec = PassSpec::new(
    "normalize",
    "NormalizeAgentGraph",
    "Normalize AIS graph structure",
    r#"Canonicalizes the AIS graph by:
- Deduplicating reasoning contexts
- Normalizing string attributes (lowercase capability/space names)
- Establishing SSA ordering invariants for downstream passes

This pass ensures the IR is in a canonical form that other passes
can rely on, similar to MLIR's canonicalizer but domain-specific."#,
    PassCategory::Transform,
    "mlir::ais::createNormalizeAgentGraphPass()",
);

/// Scheduling pass - annotates operations with scheduling metadata.
pub const SCHEDULING: PassSpec = PassSpec::new(
    "scheduling",
    "CapabilityScheduling",
    "Annotate operations with scheduling metadata",
    r#"Classifies capabilities into execution tiers and annotates operations
with scheduling hints for the runtime:

- Tier classification (io/compute/reasoning/memory/general)
- Cost estimation based on context size
- Parallel-safety markers for speculation

These annotations guide the runtime dataflow scheduler to overlap work
and optimize execution order."#,
    PassCategory::Transform,
    "mlir::ais::createCapabilitySchedulingPass()",
)
.with_options(&[
    PassOption::new(
        "parallel-threshold",
        "parallelThreshold",
        "unsigned",
        "3",
        "Maximum context size considered safe for parallel execution",
    ),
    PassOption::new(
        "base-cost",
        "baseCost",
        "unsigned",
        "2",
        "Base estimated cost for each operation",
    ),
    PassOption::new(
        "context-weight",
        "contextWeight",
        "unsigned",
        "2",
        "Cost multiplier per context operand",
    ),
]);

/// Fuse ask ops pass - merges adjacent ask operations.
pub const FUSE_ASK_OPS: PassSpec = PassSpec::new(
    "fuse-ask-ops",
    "FuseAskOps",
    "Fuse adjacent ask operations to reduce LLM calls",
    r#"Identifies producer-consumer ais.ask chains and merges them into single
batched operations. This is the highest-ROI optimization (100-400x) as it:

- Reduces serialized LLM API calls (each ~500ms-2s)
- Concatenates ask templates with separator
- Combines contexts from both operations

Only fuses AskOp (LOW latency). ThinkOp/ReasonOp are not fused because
they have different semantics (extended thinking, structured output).
Only fuses when producer has single use."#,
    PassCategory::Transform,
    "mlir::ais::createFuseAskOpsPass()",
);

/// Unconsumed value warning pass - analysis pass for detecting unused results.
pub const UNCONSUMED_VALUE_WARNING: PassSpec = PassSpec::new(
    "unconsumed-value-warning",
    "UnconsumedValueWarning",
    "Warn about unconsumed operation results",
    r#"Scans all operations and emits warnings when operation results are not
consumed by any other operation. This helps developers identify:

- Forgotten variable bindings (e.g., `ask "query" -> unused_result`)
- Missing return statements in flows
- Logic errors where data flows are incomplete

Operations with side effects (memory writes, invocations, communication)
are exempt as they have effects beyond their return value.

This pass is typically run after optimization passes but before artifact
emission, as a diagnostic aid rather than a transformation."#,
    PassCategory::Analysis,
    "mlir::ais::createUnconsumedValueWarningPass()",
);

// ============================================================================
// Built-in MLIR Passes (not generated, just registered)
// ============================================================================

/// MLIR canonicalizer pass.
pub const CANONICALIZER: PassSpec = PassSpec::new(
    "canonicalizer",
    "Canonicalizer",
    "MLIR canonicalizer (includes DCE)",
    "Standard MLIR canonicalization pass that applies rewrite patterns.",
    PassCategory::Optimization,
    "mlir::createCanonicalizerPass()",
)
.builtin();

/// MLIR CSE pass.
pub const CSE: PassSpec = PassSpec::new(
    "cse",
    "CSE",
    "MLIR common subexpression elimination",
    "Standard MLIR CSE pass that eliminates redundant computations.",
    PassCategory::Optimization,
    "mlir::createCSEPass()",
)
.builtin();

/// MLIR symbol DCE pass.
pub const SYMBOL_DCE: PassSpec = PassSpec::new(
    "symbol-dce",
    "SymbolDCE",
    "MLIR symbol dead code elimination",
    "Standard MLIR pass that eliminates unused symbols.",
    PassCategory::Optimization,
    "mlir::createSymbolDCEPass()",
)
.builtin();

// ============================================================================
// All Passes Collection
// ============================================================================

/// All AIS-specific passes (not built-in MLIR passes).
pub const AIS_PASSES: &[&PassSpec] = &[
    &NORMALIZE,
    &SCHEDULING,
    &FUSE_ASK_OPS,
    &UNCONSUMED_VALUE_WARNING,
];

/// All passes including built-in MLIR passes.
pub const ALL_PASSES: &[&PassSpec] = &[
    // AIS-specific
    &NORMALIZE,
    &SCHEDULING,
    &FUSE_ASK_OPS,
    &UNCONSUMED_VALUE_WARNING,
    // Built-in MLIR
    &CANONICALIZER,
    &CSE,
    &SYMBOL_DCE,
];

/// Get all pass specifications.
pub fn get_all_passes() -> impl Iterator<Item = &'static PassSpec> {
    ALL_PASSES.iter().copied()
}

/// Get only AIS-specific passes (for TableGen generation).
pub fn get_ais_passes() -> impl Iterator<Item = &'static PassSpec> {
    AIS_PASSES.iter().copied()
}

/// Find a pass by its CLI name.
pub fn find_pass_by_name(name: &str) -> Option<&'static PassSpec> {
    ALL_PASSES.iter().find(|p| p.name == name).copied()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_passes_have_unique_names() {
        let mut names: Vec<&str> = ALL_PASSES.iter().map(|p| p.name).collect();
        names.sort();
        let original_len = names.len();
        names.dedup();
        assert_eq!(names.len(), original_len, "Duplicate pass names found");
    }

    #[test]
    fn test_find_pass_by_name() {
        assert!(find_pass_by_name("normalize").is_some());
        assert!(find_pass_by_name("fuse-ask-ops").is_some());
        assert!(find_pass_by_name("canonicalizer").is_some());
        assert!(find_pass_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_ais_passes_not_builtin() {
        for pass in get_ais_passes() {
            assert!(
                !pass.is_builtin,
                "AIS pass {} should not be marked as builtin",
                pass.name
            );
        }
    }

    #[test]
    fn test_builtin_passes_marked() {
        assert!(CANONICALIZER.is_builtin);
        assert!(CSE.is_builtin);
        assert!(SYMBOL_DCE.is_builtin);
    }

    #[test]
    fn test_scheduling_has_options() {
        assert!(!SCHEDULING.options.is_empty());
        assert_eq!(SCHEDULING.options.len(), 3);
    }
}
