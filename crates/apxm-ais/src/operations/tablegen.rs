//! TableGen Generator - Generates MLIR TableGen from Rust definitions.
//!
//! This module generates `.td` files from Rust operation definitions,
//! enabling Rust to be the single source of truth for AIS operations.

use super::OperationCategory;
use super::definitions::{AISOperationType, OperationSpec, get_all_operations};

// ============================================================================
// MLIR-Specific Types for TableGen Generation
// ============================================================================

/// MLIR traits that can be applied to operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirTrait {
    /// Pure operation (no side effects).
    Pure,
    /// Operation is a terminator.
    Terminator,
    /// Operation must have a specific parent.
    HasParent(&'static str),
    /// Custom trait string (for complex traits).
    Custom(&'static str),
}

impl MlirTrait {
    /// Convert to TableGen format.
    pub fn to_tablegen(&self) -> String {
        match self {
            MlirTrait::Pure => "Pure".to_string(),
            MlirTrait::Terminator => "Terminator".to_string(),
            MlirTrait::HasParent(parent) => format!("HasParent<\"{}\">", parent),
            MlirTrait::Custom(s) => s.to_string(),
        }
    }
}

/// Memory resources for side effect tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AISResource {
    /// Belief memory resource.
    Belief,
    /// Goal memory resource.
    Goal,
    /// Capability resource.
    Capability,
    /// Episodic memory resource.
    Episodic,
}

impl AISResource {
    /// Convert to TableGen resource name.
    pub fn to_tablegen(&self) -> &'static str {
        match self {
            AISResource::Belief => "AIS_BeliefResource",
            AISResource::Goal => "AIS_GoalResource",
            AISResource::Capability => "AIS_CapabilityResource",
            AISResource::Episodic => "AIS_EpisodicResource",
        }
    }
}

/// Memory effect for an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryEffect {
    /// Read from a resource.
    MemRead(AISResource),
    /// Write to a resource.
    MemWrite(AISResource),
}

impl MemoryEffect {
    /// Convert to TableGen format.
    pub fn to_tablegen(&self) -> String {
        match self {
            MemoryEffect::MemRead(r) => format!("MemRead<{}>", r.to_tablegen()),
            MemoryEffect::MemWrite(r) => format!("MemWrite<{}>", r.to_tablegen()),
        }
    }
}

/// MLIR argument type for TableGen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirArgType {
    /// String attribute.
    StrAttr,
    /// Optional string attribute.
    OptionalStrAttr,
    /// Integer attribute.
    I64Attr,
    /// Optional integer attribute.
    OptionalI64Attr,
    /// Token type.
    Token,
    /// Optional token type.
    OptionalToken,
    /// Variadic tokens.
    VariadicTokens,
    /// Handle type.
    Handle,
    /// Any token or handle.
    AnyTokenOrHandle,
    /// Any token, handle, or goal.
    AnyTokenHandleOrGoal,
    /// Variadic any token, handle, or goal.
    VariadicAnyTokenHandleOrGoal,
    /// Custom type string.
    Custom(&'static str),
}

impl MlirArgType {
    /// Convert to TableGen format.
    pub fn to_tablegen(&self) -> &'static str {
        match self {
            MlirArgType::StrAttr => "StrAttr",
            MlirArgType::OptionalStrAttr => "OptionalAttr<StrAttr>",
            MlirArgType::I64Attr => "I64Attr",
            MlirArgType::OptionalI64Attr => "OptionalAttr<I64Attr>",
            MlirArgType::Token => "AIS_TokenType",
            MlirArgType::OptionalToken => "Optional<AIS_TokenType>",
            MlirArgType::VariadicTokens => "Variadic<AIS_TokenType>",
            MlirArgType::Handle => "AIS_HandleType",
            MlirArgType::AnyTokenOrHandle => "AIS_AnyTokenOrHandle",
            MlirArgType::AnyTokenHandleOrGoal => "AIS_AnyTokenHandleOrGoal",
            MlirArgType::VariadicAnyTokenHandleOrGoal => "Variadic<AIS_AnyTokenHandleOrGoal>",
            MlirArgType::Custom(s) => s,
        }
    }
}

/// An MLIR operation argument.
#[derive(Debug, Clone)]
pub struct MlirArgument {
    /// Argument name (will become $name in TableGen).
    pub name: &'static str,
    /// Argument type.
    pub arg_type: MlirArgType,
}

impl MlirArgument {
    /// Create a new argument.
    pub const fn new(name: &'static str, arg_type: MlirArgType) -> Self {
        Self { name, arg_type }
    }

    /// Convert to TableGen format.
    pub fn to_tablegen(&self) -> String {
        format!("{}:${}", self.arg_type.to_tablegen(), self.name)
    }
}

/// MLIR result type for TableGen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirResultType {
    /// Token type.
    Token,
    /// Handle type.
    Handle,
    /// Goal type.
    Goal,
    /// Custom type.
    Custom(&'static str),
}

impl MlirResultType {
    /// Convert to TableGen format.
    pub fn to_tablegen(&self) -> &'static str {
        match self {
            MlirResultType::Token => "AIS_TokenType",
            MlirResultType::Handle => "AIS_HandleType",
            MlirResultType::Goal => "AIS_GoalType",
            MlirResultType::Custom(s) => s,
        }
    }
}

/// An MLIR operation result.
#[derive(Debug, Clone)]
pub struct MlirResult {
    /// Result name.
    pub name: &'static str,
    /// Result type.
    pub result_type: MlirResultType,
}

impl MlirResult {
    /// Create a new result.
    pub const fn new(name: &'static str, result_type: MlirResultType) -> Self {
        Self { name, result_type }
    }

    /// Convert to TableGen format.
    pub fn to_tablegen(&self) -> String {
        format!("{}:${}", self.result_type.to_tablegen(), self.name)
    }
}

/// Region definition for operations with nested regions.
#[derive(Debug, Clone)]
pub struct MlirRegion {
    /// Region name.
    pub name: &'static str,
    /// Region type (usually "AnyRegion").
    pub region_type: &'static str,
}

impl MlirRegion {
    /// Create a new region.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            region_type: "AnyRegion",
        }
    }
}

// ============================================================================
// Extended Operation Specification for TableGen
// ============================================================================

/// Extended operation specification with MLIR-specific fields.
///
/// This extends the base OperationSpec with fields needed for TableGen generation.
pub struct MlirOperationSpec {
    /// Base operation spec (from definitions.rs).
    pub base: &'static OperationSpec,

    /// MLIR traits for the operation.
    pub traits: &'static [MlirTrait],

    /// Memory effects (reads/writes).
    pub memory_effects: &'static [MemoryEffect],

    /// MLIR arguments (ins).
    pub arguments: &'static [MlirArgument],

    /// MLIR results (outs).
    pub results: &'static [MlirResult],

    /// Assembly format string.
    pub assembly_format: Option<&'static str>,

    /// Nested regions.
    pub regions: &'static [MlirRegion],

    /// Has custom verifier.
    pub has_verifier: bool,

    /// Has canonicalizer.
    pub has_canonicalizer: bool,

    /// Has folder for constant folding.
    pub has_folder: bool,
}

// ============================================================================
// TableGen Generation
// ============================================================================

/// TableGen file header.
const TABLEGEN_HEADER: &str = r#"/**
 * @file  AISOps.generated.td
 * @brief GENERATED from Rust - DO NOT EDIT MANUALLY
 *
 * This file is generated by apxm-ais/src/operations/tablegen.rs
 * To modify operations, edit crates/apxm-ais/src/operations/definitions.rs
 */

#ifndef APXM_AIS_OPS_GENERATED
#define APXM_AIS_OPS_GENERATED

include "apxm/Dialect/AIS/IR/AISDialect.td"
include "apxm/Dialect/AIS/IR/AISTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

"#;

/// TableGen file footer.
const TABLEGEN_FOOTER: &str = "\n#endif // APXM_AIS_OPS_GENERATED\n";

/// Generate a TableGen operation definition.
fn generate_op_def(spec: &MlirOperationSpec) -> String {
    let base = spec.base;
    let op_name = to_tablegen_name(base.op_type);
    let mnemonic = base.op_type.mlir_mnemonic();

    // Build traits list
    let mut traits = Vec::new();

    // Add explicit traits
    for t in spec.traits {
        traits.push(t.to_tablegen());
    }

    // Add memory effects as a trait
    if !spec.memory_effects.is_empty() {
        let effects: Vec<String> = spec
            .memory_effects
            .iter()
            .map(|e| e.to_tablegen())
            .collect();
        traits.push(format!("MemoryEffects<[{}]>", effects.join(", ")));
    }

    let traits_str = if traits.is_empty() {
        String::new()
    } else {
        traits.join(", ")
    };

    // Build arguments
    let args_str = if spec.arguments.is_empty() {
        String::new()
    } else {
        let args: Vec<String> = spec.arguments.iter().map(|a| a.to_tablegen()).collect();
        format!("  let arguments = (ins\n    {}\n  );", args.join(",\n    "))
    };

    // Build results
    let results_str = if spec.results.is_empty() {
        String::new()
    } else {
        let results: Vec<String> = spec.results.iter().map(|r| r.to_tablegen()).collect();
        format!("  let results = (outs {});", results.join(", "))
    };

    // Build regions
    let regions_str = if spec.regions.is_empty() {
        String::new()
    } else {
        let regions: Vec<String> = spec
            .regions
            .iter()
            .map(|r| format!("{}:${}", r.region_type, r.name))
            .collect();
        format!("  let regions = (region {});", regions.join(", "))
    };

    // Assembly format
    let format_str = spec
        .assembly_format
        .map(|f| format!("  let assemblyFormat = [{{{}}}];", f))
        .unwrap_or_default();

    // Flags
    let mut flags = Vec::new();
    if spec.has_verifier {
        flags.push("  let hasVerifier = 1;".to_string());
    }
    if spec.has_canonicalizer {
        flags.push("  let hasCanonicalizer = 1;".to_string());
    }
    if spec.has_folder {
        flags.push("  let hasFolder = 1;".to_string());
    }

    // Combine all parts
    let mut parts = vec![
        format!(
            "def {} : AIS_Op<\"{}\", [{}]> {{",
            op_name, mnemonic, traits_str
        ),
        format!("  let summary = \"{}\";", base.name),
        format!(
            "  let description = [{{{}}}];",
            base.description.replace('\n', "\\n")
        ),
    ];

    if !args_str.is_empty() {
        parts.push(args_str);
    }
    if !results_str.is_empty() {
        parts.push(results_str);
    }
    if !regions_str.is_empty() {
        parts.push(regions_str);
    }
    if !format_str.is_empty() {
        parts.push(format_str);
    }
    for flag in flags {
        parts.push(flag);
    }
    parts.push("}".to_string());

    parts.join("\n")
}

/// Convert operation type to TableGen definition name.
fn to_tablegen_name(op_type: AISOperationType) -> String {
    let name = match op_type {
        AISOperationType::Agent => "Agent",
        AISOperationType::QMem => "QMem",
        AISOperationType::UMem => "UMem",
        AISOperationType::Rsn => "Rsn",
        AISOperationType::Plan => "Plan",
        AISOperationType::Reflect => "Reflect",
        AISOperationType::Verify => "Verify",
        AISOperationType::Inv => "Inv",
        AISOperationType::Exc => "Exc",
        AISOperationType::Print => "Print",
        AISOperationType::Jump => "Jump",
        AISOperationType::BranchOnValue => "BranchOnValue",
        AISOperationType::LoopStart => "LoopStart",
        AISOperationType::LoopEnd => "LoopEnd",
        AISOperationType::Return => "Return",
        AISOperationType::Switch => "Switch",
        AISOperationType::FlowCall => "FlowCall",
        AISOperationType::Merge => "Merge",
        AISOperationType::Fence => "Fence",
        AISOperationType::WaitAll => "WaitAll",
        AISOperationType::TryCatch => "TryCatch",
        AISOperationType::Err => "Err",
        AISOperationType::Communicate => "Communicate",
        AISOperationType::ConstStr => "ConstStr",
        AISOperationType::Yield => "Yield",
    };
    format!("AIS_{}Op", name)
}

/// Get the category comment for a section.
fn get_category_comment(category: OperationCategory) -> &'static str {
    match category {
        OperationCategory::Metadata => "Agent Metadata",
        OperationCategory::Memory => "Memory Operations",
        OperationCategory::Reasoning => "Reasoning Operations",
        OperationCategory::Tools => "Tool Operations",
        OperationCategory::ControlFlow => "Control Flow Operations",
        OperationCategory::Synchronization => "Synchronization Operations",
        OperationCategory::ErrorHandling => "Error Handling Operations",
        OperationCategory::Communication => "Communication Operations",
        OperationCategory::Internal => "Internal Operations",
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Generate complete TableGen file content.
///
/// This is the main entry point for TableGen generation.
/// It generates a complete .td file that can be used with mlir-tblgen.
pub fn generate_tablegen() -> String {
    let mut output = String::new();
    output.push_str(TABLEGEN_HEADER);

    // Get all operations and convert to MLIR specs
    let specs = get_mlir_operation_specs();

    // Group by category
    let mut current_category: Option<OperationCategory> = None;

    for spec in &specs {
        let category = spec.base.category;

        // Add section comment if category changed
        if current_category != Some(category) {
            current_category = Some(category);
            let comment = get_category_comment(category);
            output.push_str(&format!(
                "\n//===----------------------------------------------------------------------===//\n\
                 // {}\n\
                 //===----------------------------------------------------------------------===//\n\n",
                comment
            ));
        }

        output.push_str(&generate_op_def(spec));
        output.push_str("\n\n");
    }

    output.push_str(TABLEGEN_FOOTER);
    output
}

/// Get MLIR operation specs for all operations.
///
/// This returns the extended specs with MLIR-specific metadata.
/// In the future, this should be defined directly in definitions.rs,
/// but for now we derive it from the base specs.
pub fn get_mlir_operation_specs() -> Vec<MlirOperationSpec> {
    // For now, create basic specs from the base definitions.
    // This will be expanded to include full MLIR metadata.
    get_all_operations()
        .map(|spec| derive_mlir_spec(spec))
        .collect()
}

/// Derive MLIR spec from base operation spec.
///
/// This creates a basic MLIR spec with reasonable defaults.
/// Eventually, this should be replaced with explicit definitions.
fn derive_mlir_spec(base: &'static OperationSpec) -> MlirOperationSpec {
    use AISOperationType::*;

    // Derive traits based on operation type and category
    let traits: &'static [MlirTrait] = match base.op_type {
        Agent => &[MlirTrait::HasParent("mlir::ModuleOp")],
        Return => &[MlirTrait::Terminator],
        ConstStr | WaitAll | Merge => &[MlirTrait::Pure],
        _ => &[],
    };

    // Derive memory effects based on category
    let memory_effects: &'static [MemoryEffect] = match base.op_type {
        QMem => &[MemoryEffect::MemRead(AISResource::Belief)],
        UMem => &[MemoryEffect::MemWrite(AISResource::Belief)],
        Rsn => &[
            MemoryEffect::MemRead(AISResource::Belief),
            MemoryEffect::MemWrite(AISResource::Belief),
            MemoryEffect::MemWrite(AISResource::Goal),
        ],
        Reflect => &[
            MemoryEffect::MemRead(AISResource::Episodic),
            MemoryEffect::MemWrite(AISResource::Belief),
        ],
        Verify => &[MemoryEffect::MemRead(AISResource::Belief)],
        Plan => &[MemoryEffect::MemWrite(AISResource::Goal)],
        Inv => &[MemoryEffect::MemRead(AISResource::Capability)],
        _ => &[],
    };

    // Determine which ops have verifiers/canonicalizers
    let has_verifier = !matches!(base.op_type, Agent | ConstStr);
    let has_canonicalizer = matches!(base.op_type, Rsn | WaitAll | Merge);
    let has_folder = matches!(base.op_type, ConstStr);

    MlirOperationSpec {
        base,
        traits,
        memory_effects,
        arguments: &[], // Will be derived from fields
        results: &[],   // Will be derived from produces_output
        assembly_format: None,
        regions: &[],
        has_verifier,
        has_canonicalizer,
        has_folder,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_tablegen() {
        let output = generate_tablegen();

        // Check header
        assert!(output.contains("#ifndef APXM_AIS_OPS_GENERATED"));
        assert!(output.contains("GENERATED from Rust"));

        // Check all 21 operations are present
        for op_type in AISOperationType::all_operations() {
            let name = to_tablegen_name(*op_type);
            assert!(
                output.contains(&name),
                "Missing operation: {}",
                op_type.mlir_mnemonic()
            );
        }

        // Check footer
        assert!(output.contains("#endif // APXM_AIS_OPS_GENERATED"));
    }

    #[test]
    fn test_trait_to_tablegen() {
        assert_eq!(MlirTrait::Pure.to_tablegen(), "Pure");
        assert_eq!(MlirTrait::Terminator.to_tablegen(), "Terminator");
        assert_eq!(
            MlirTrait::HasParent("mlir::ModuleOp").to_tablegen(),
            "HasParent<\"mlir::ModuleOp\">"
        );
    }

    #[test]
    fn test_memory_effect_to_tablegen() {
        assert_eq!(
            MemoryEffect::MemRead(AISResource::Belief).to_tablegen(),
            "MemRead<AIS_BeliefResource>"
        );
        assert_eq!(
            MemoryEffect::MemWrite(AISResource::Goal).to_tablegen(),
            "MemWrite<AIS_GoalResource>"
        );
    }

    #[test]
    fn test_tablegen_op_names() {
        assert_eq!(to_tablegen_name(AISOperationType::Agent), "AIS_AgentOp");
        assert_eq!(to_tablegen_name(AISOperationType::Rsn), "AIS_RsnOp");
        assert_eq!(to_tablegen_name(AISOperationType::QMem), "AIS_QMemOp");
    }
}
