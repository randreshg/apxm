//! AIS Operation Definitions - Single Source of Truth
//!
//! This module contains the complete specification for all 21 AIS operations
//! (19 public + 1 metadata + 1 internal). Both the compiler and runtime use
//! these definitions to ensure consistent semantics.

use super::category::OperationCategory;
use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Operation Type Enum
// ============================================================================

/// Represents all AIS operation types.
///
/// This enum is the canonical list of operations (21 total):
/// - 1 metadata operation (AgentOp)
/// - 19 public operations
/// - 1 internal operation (ConstStr)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AISOperationType {
    // Metadata Operation (1) - matches tablegen ais.agent
    /// Agent metadata declaration (memory, beliefs, goals, capabilities).
    Agent,

    // Memory Operations (2)
    /// Query memory (read from memory system).
    QMem,
    /// Update memory (write to memory system).
    UMem,

    // Reasoning Operations (4)
    /// Reasoning operation (LLM call for reasoning).
    Rsn,
    /// Planning operation (generate a plan using LLM).
    Plan,
    /// Reflection operation (analyze execution trace).
    Reflect,
    /// Verification operation (fact-check against evidence).
    Verify,

    // Tool Operations (2)
    /// Invoke a capability (tool/function call).
    Inv,
    /// Execute code in sandbox.
    Exc,

    // Control Flow Operations (5)
    /// Unconditional jump to label.
    Jump,
    /// Branch based on value comparison.
    BranchOnValue,
    /// Loop start marker.
    LoopStart,
    /// Loop end marker.
    LoopEnd,
    /// Return from subgraph with result.
    Return,

    // Synchronization Operations (3)
    /// Merge multiple tokens into one.
    Merge,
    /// Memory fence (synchronization barrier).
    Fence,
    /// Wait for all input tokens to be ready.
    WaitAll,

    // Error Handling Operations (2)
    /// Try-catch exception handling.
    TryCatch,
    /// Error handler invocation.
    Err,

    // Communication Operations (1)
    /// Communication between agents.
    Communicate,

    // Internal Operations (not part of public AIS)
    /// String constant (compiler internal).
    ConstStr,
}

impl fmt::Display for AISOperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Metadata
            AISOperationType::Agent => write!(f, "AGENT"),
            // Memory
            AISOperationType::QMem => write!(f, "QMEM"),
            AISOperationType::UMem => write!(f, "UMEM"),
            // Reasoning
            AISOperationType::Rsn => write!(f, "RSN"),
            AISOperationType::Plan => write!(f, "PLAN"),
            AISOperationType::Reflect => write!(f, "REFLECT"),
            AISOperationType::Verify => write!(f, "VERIFY"),
            // Tools
            AISOperationType::Inv => write!(f, "INV"),
            AISOperationType::Exc => write!(f, "EXC"),
            // Control Flow
            AISOperationType::Jump => write!(f, "JUMP"),
            AISOperationType::BranchOnValue => write!(f, "BRANCH_ON_VALUE"),
            AISOperationType::LoopStart => write!(f, "LOOP_START"),
            AISOperationType::LoopEnd => write!(f, "LOOP_END"),
            AISOperationType::Return => write!(f, "RETURN"),
            // Synchronization
            AISOperationType::Merge => write!(f, "MERGE"),
            AISOperationType::Fence => write!(f, "FENCE"),
            AISOperationType::WaitAll => write!(f, "WAIT_ALL"),
            // Error Handling
            AISOperationType::TryCatch => write!(f, "TRY_CATCH"),
            AISOperationType::Err => write!(f, "ERR"),
            // Communication
            AISOperationType::Communicate => write!(f, "COMMUNICATE"),
            // Internal
            AISOperationType::ConstStr => write!(f, "CONST_STR"),
        }
    }
}

impl AISOperationType {
    /// Returns the MLIR mnemonic for this operation (e.g., "rsn", "qmem").
    pub fn mlir_mnemonic(&self) -> &'static str {
        match self {
            AISOperationType::Agent => "agent",
            AISOperationType::QMem => "qmem",
            AISOperationType::UMem => "umem",
            AISOperationType::Rsn => "rsn",
            AISOperationType::Plan => "plan",
            AISOperationType::Reflect => "reflect",
            AISOperationType::Verify => "verify",
            AISOperationType::Inv => "inv",
            AISOperationType::Exc => "exc",
            AISOperationType::Jump => "jump",
            AISOperationType::BranchOnValue => "branch_on_value",
            AISOperationType::LoopStart => "loop_start",
            AISOperationType::LoopEnd => "loop_end",
            AISOperationType::Return => "return",
            AISOperationType::Merge => "merge",
            AISOperationType::Fence => "fence",
            AISOperationType::WaitAll => "wait_all",
            AISOperationType::TryCatch => "try_catch",
            AISOperationType::Err => "err",
            AISOperationType::Communicate => "communicate",
            AISOperationType::ConstStr => "const_str",
        }
    }

    /// Returns true if this is a public AIS operation (part of the 19).
    pub fn is_public(&self) -> bool {
        !matches!(
            self,
            AISOperationType::ConstStr | AISOperationType::Agent
        )
    }

    /// Returns true if this is a metadata operation.
    pub fn is_metadata(&self) -> bool {
        matches!(self, AISOperationType::Agent)
    }

    /// Returns true if this is an internal operation.
    pub fn is_internal(&self) -> bool {
        matches!(self, AISOperationType::ConstStr)
    }

    /// Get all public operation types (19 operations).
    pub fn public_operations() -> &'static [AISOperationType] {
        &[
            AISOperationType::QMem,
            AISOperationType::UMem,
            AISOperationType::Rsn,
            AISOperationType::Plan,
            AISOperationType::Reflect,
            AISOperationType::Verify,
            AISOperationType::Inv,
            AISOperationType::Exc,
            AISOperationType::Jump,
            AISOperationType::BranchOnValue,
            AISOperationType::LoopStart,
            AISOperationType::LoopEnd,
            AISOperationType::Return,
            AISOperationType::Merge,
            AISOperationType::Fence,
            AISOperationType::WaitAll,
            AISOperationType::TryCatch,
            AISOperationType::Err,
            AISOperationType::Communicate,
        ]
    }

    /// Get all operation types (21 total).
    pub fn all_operations() -> &'static [AISOperationType] {
        &[
            AISOperationType::Agent,
            AISOperationType::QMem,
            AISOperationType::UMem,
            AISOperationType::Rsn,
            AISOperationType::Plan,
            AISOperationType::Reflect,
            AISOperationType::Verify,
            AISOperationType::Inv,
            AISOperationType::Exc,
            AISOperationType::Jump,
            AISOperationType::BranchOnValue,
            AISOperationType::LoopStart,
            AISOperationType::LoopEnd,
            AISOperationType::Return,
            AISOperationType::Merge,
            AISOperationType::Fence,
            AISOperationType::WaitAll,
            AISOperationType::TryCatch,
            AISOperationType::Err,
            AISOperationType::Communicate,
            AISOperationType::ConstStr,
        ]
    }
}

// ============================================================================
// Field Specifications
// ============================================================================

/// A field in an operation specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationField {
    /// Field name.
    pub name: &'static str,
    /// Whether the field is required.
    pub required: bool,
    /// Description of the field.
    pub description: &'static str,
}

impl OperationField {
    /// Create a required field.
    pub const fn required(name: &'static str, description: &'static str) -> Self {
        Self {
            name,
            required: true,
            description,
        }
    }

    /// Create an optional field.
    pub const fn optional(name: &'static str, description: &'static str) -> Self {
        Self {
            name,
            required: false,
            description,
        }
    }
}

// ============================================================================
// Operation Specification
// ============================================================================

/// Complete specification of an AIS operation.
///
/// This is the single source of truth for operation semantics. Both the
/// compiler (for validation and code generation) and runtime (for dispatch
/// and execution) use these specifications.
#[derive(Debug, Clone)]
pub struct OperationSpec {
    /// Operation type identifier.
    pub op_type: AISOperationType,
    /// Human-readable name (e.g., "QueryMemory", "Reason").
    pub name: &'static str,
    /// Operation category.
    pub category: OperationCategory,
    /// Description of what the operation does.
    pub description: &'static str,
    /// Required and optional input fields.
    pub fields: &'static [OperationField],
    /// Whether this operation needs async execution (submission to executor).
    pub needs_submission: bool,
    /// Minimum number of input tokens required.
    pub min_inputs: u32,
    /// Whether operation produces output tokens.
    pub produces_output: bool,
}

impl OperationSpec {
    /// Get a field by name.
    pub fn get_field(&self, name: &str) -> Option<&OperationField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Check if a field is required.
    pub fn is_field_required(&self, name: &str) -> bool {
        self.get_field(name).is_some_and(|f| f.required)
    }

    /// Get all required fields.
    pub fn required_fields(&self) -> impl Iterator<Item = &OperationField> {
        self.fields.iter().filter(|f| f.required)
    }

    /// Get all optional fields.
    pub fn optional_fields(&self) -> impl Iterator<Item = &OperationField> {
        self.fields.iter().filter(|f| !f.required)
    }
}

// ============================================================================
// Metadata Operations (1) - Agent structural declaration
// ============================================================================

/// Metadata operations (1 operation).
pub static METADATA_OPERATIONS: &[OperationSpec] = &[OperationSpec {
    op_type: AISOperationType::Agent,
    name: "Agent",
    category: OperationCategory::Metadata,
    description: "Agent structural declaration (memory, beliefs, goals, capabilities)",
    fields: &[
        OperationField::optional("memory", "Memory configuration"),
        OperationField::optional("beliefs", "Initial beliefs"),
        OperationField::optional("goals", "Initial goals"),
        OperationField::optional("capabilities", "Available capabilities"),
    ],
    needs_submission: false,
    min_inputs: 0,
    produces_output: false,
}];

// ============================================================================
// Operation Registry: 19 Public Operations
// ============================================================================

/// All 19 public AIS operations with complete metadata.
pub static AIS_OPERATIONS: &[OperationSpec] = &[
    // ========== Memory Operations (2) ==========
    OperationSpec {
        op_type: AISOperationType::QMem,
        name: "QueryMemory",
        category: OperationCategory::Memory,
        description: "Retrieve data from memory (STM, LTM, or Episodic)",
        fields: &[
            OperationField::required("query", "Query string or key to search for"),
            OperationField::optional("memory_tier", "Target memory tier: stm, ltm, or episodic"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::UMem,
        name: "UpdateMemory",
        category: OperationCategory::Memory,
        description: "Persist or update data in memory",
        fields: &[
            OperationField::required("key", "Key to store the value under"),
            OperationField::required("value", "Value to store"),
            OperationField::optional("memory_tier", "Target memory tier: stm, ltm, or episodic"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Reasoning Operations (4) ==========
    OperationSpec {
        op_type: AISOperationType::Rsn,
        name: "Reason",
        category: OperationCategory::Reasoning,
        description: "LLM reasoning with controlled context; updates Beliefs/Goals",
        fields: &[
            OperationField::required("prompt", "Prompt template for reasoning"),
            OperationField::optional("model", "LLM model to use"),
            OperationField::optional("context", "Additional context for reasoning"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Plan,
        name: "Plan",
        category: OperationCategory::Reasoning,
        description: "Decompose goal into AIS subgraph",
        fields: &[
            OperationField::required("goal", "Goal to decompose into steps"),
            OperationField::optional("constraints", "Constraints on the plan"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Reflect,
        name: "Reflect",
        category: OperationCategory::Reasoning,
        description: "Analyze execution trace for self-improvement",
        fields: &[
            OperationField::required("trace_query", "Query to retrieve trace for reflection"),
            OperationField::optional("reflection_prompt", "Custom prompt for reflection"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Verify,
        name: "Verify",
        category: OperationCategory::Reasoning,
        description: "Fact-check outputs against evidence",
        fields: &[
            OperationField::required("claim", "Claim to verify"),
            OperationField::required("evidence", "Evidence to check against"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Tool Operations (2) ==========
    OperationSpec {
        op_type: AISOperationType::Inv,
        name: "InvokeTool",
        category: OperationCategory::Tools,
        description: "Call external tool with structured params; store result",
        fields: &[
            OperationField::required("capability", "Name of the capability/tool to invoke"),
            OperationField::optional("parameters", "Parameters to pass to the tool"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Exc,
        name: "ExecuteCode",
        category: OperationCategory::Tools,
        description: "Run code in a sandboxed environment; update Beliefs",
        fields: &[
            OperationField::required("code", "Code to execute"),
            OperationField::optional("sandbox_config", "Sandbox configuration"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Control Flow Operations (5) ==========
    OperationSpec {
        op_type: AISOperationType::Jump,
        name: "Jump",
        category: OperationCategory::ControlFlow,
        description: "Unconditional jump to a labeled instruction",
        fields: &[OperationField::required("label", "Target label to jump to")],
        needs_submission: false,
        min_inputs: 0,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::BranchOnValue,
        name: "BranchOnValue",
        category: OperationCategory::ControlFlow,
        description: "Conditional branch based on token value comparison",
        fields: &[
            OperationField::required("token", "Token to evaluate"),
            OperationField::required("value", "Value to compare against"),
            OperationField::required("label_true", "Label if comparison is true"),
            OperationField::required("label_false", "Label if comparison is false"),
        ],
        needs_submission: false,
        min_inputs: 1,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::LoopStart,
        name: "LoopStart",
        category: OperationCategory::ControlFlow,
        description: "Begin bounded loop",
        fields: &[OperationField::required(
            "count_token",
            "Token containing iteration count",
        )],
        needs_submission: false,
        min_inputs: 1,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::LoopEnd,
        name: "LoopEnd",
        category: OperationCategory::ControlFlow,
        description: "End bounded loop",
        fields: &[],
        needs_submission: false,
        min_inputs: 0,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::Return,
        name: "Return",
        category: OperationCategory::ControlFlow,
        description: "Return from subgraph with result token",
        fields: &[OperationField::required("token", "Result token to return")],
        needs_submission: false,
        min_inputs: 1,
        produces_output: true,
    },
    // ========== Synchronization Operations (3) ==========
    OperationSpec {
        op_type: AISOperationType::Merge,
        name: "Merge",
        category: OperationCategory::Synchronization,
        description: "Sync parallel paths; aggregate tokens into one",
        fields: &[OperationField::required("tokens", "List of tokens to merge")],
        needs_submission: true,
        min_inputs: 1,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Fence,
        name: "Fence",
        category: OperationCategory::Synchronization,
        description: "Memory barrier; order prior QMEM/UMEM operations",
        fields: &[OperationField::optional("ordering", "Memory ordering constraint")],
        needs_submission: true,
        min_inputs: 0,
        produces_output: false,
    },
    OperationSpec {
        op_type: AISOperationType::WaitAll,
        name: "WaitAll",
        category: OperationCategory::Synchronization,
        description: "Block until all specified tokens are available",
        fields: &[OperationField::required("tokens", "Tokens to wait for")],
        needs_submission: true,
        min_inputs: 1,
        produces_output: true,
    },
    // ========== Error Handling Operations (2) ==========
    OperationSpec {
        op_type: AISOperationType::TryCatch,
        name: "TryCatch",
        category: OperationCategory::ErrorHandling,
        description: "Structured exception handling with recovery subgraph",
        fields: &[
            OperationField::required("try_subgraph", "Subgraph to try executing"),
            OperationField::required("catch_subgraph", "Recovery subgraph on failure"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    OperationSpec {
        op_type: AISOperationType::Err,
        name: "HandleError",
        category: OperationCategory::ErrorHandling,
        description: "Invoke recovery template on failure; update Goals/Beliefs",
        fields: &[OperationField::required(
            "error_handler",
            "Error handler to invoke",
        )],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
    // ========== Communication Operations (1) ==========
    OperationSpec {
        op_type: AISOperationType::Communicate,
        name: "Communicate",
        category: OperationCategory::Communication,
        description: "Send message to recipient using protocol",
        fields: &[
            OperationField::required("target_agent", "Target agent to communicate with"),
            OperationField::required("message", "Message to send"),
            OperationField::optional("protocol", "Communication protocol to use"),
        ],
        needs_submission: true,
        min_inputs: 0,
        produces_output: true,
    },
];

// ============================================================================
// Internal Operations (not part of public AIS)
// ============================================================================

/// Internal operations used by the compiler but not exposed in the public AIS.
pub static INTERNAL_OPERATIONS: &[OperationSpec] = &[OperationSpec {
    op_type: AISOperationType::ConstStr,
    name: "ConstStr",
    category: OperationCategory::Internal,
    description: "String constant (compiler internal for string literals)",
    fields: &[OperationField::required("value", "The string constant value")],
    needs_submission: false,
    min_inputs: 0,
    produces_output: true,
}];

// ============================================================================
// Lookup Functions
// ============================================================================

/// Get the specification for an operation type.
///
/// This function never fails - all operation types have corresponding specs.
/// It checks metadata, public, and internal operations.
pub fn get_operation_spec(op_type: AISOperationType) -> &'static OperationSpec {
    // Check metadata operations
    if let Some(spec) = METADATA_OPERATIONS.iter().find(|s| s.op_type == op_type) {
        return spec;
    }

    // Check public operations
    if let Some(spec) = AIS_OPERATIONS.iter().find(|s| s.op_type == op_type) {
        return spec;
    }

    // Check internal operations
    if let Some(spec) = INTERNAL_OPERATIONS.iter().find(|s| s.op_type == op_type) {
        return spec;
    }

    // This should never happen if the operation type enum and specs are in sync
    panic!(
        "Operation {:?} has no specification. This is a bug - all operations must have specs.",
        op_type
    );
}

/// Get all operation specifications (metadata + public + internal).
pub fn get_all_operations() -> impl Iterator<Item = &'static OperationSpec> {
    METADATA_OPERATIONS
        .iter()
        .chain(AIS_OPERATIONS.iter())
        .chain(INTERNAL_OPERATIONS.iter())
}

/// Get only public operation specifications.
pub fn get_public_operations() -> impl Iterator<Item = &'static OperationSpec> {
    AIS_OPERATIONS.iter()
}

/// Find an operation specification by name.
///
/// Searches metadata, public, and internal operations.
/// Returns None if no operation with the given name exists.
pub fn find_operation_by_name(name: &str) -> Option<&'static OperationSpec> {
    get_all_operations().find(|s| s.name == name)
}

/// Find an operation specification by MLIR mnemonic.
pub fn find_operation_by_mnemonic(mnemonic: &str) -> Option<&'static OperationSpec> {
    get_all_operations().find(|s| s.op_type.mlir_mnemonic() == mnemonic)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_ops_have_specs() {
        for op_type in AISOperationType::all_operations() {
            let spec = get_operation_spec(*op_type);
            assert_eq!(spec.op_type, *op_type);
        }
    }

    #[test]
    fn test_operation_counts() {
        assert_eq!(METADATA_OPERATIONS.len(), 1, "Expected 1 metadata operation");
        assert_eq!(AIS_OPERATIONS.len(), 19, "Expected 19 public AIS operations");
        assert_eq!(INTERNAL_OPERATIONS.len(), 1, "Expected 1 internal operation");
        assert_eq!(
            AISOperationType::all_operations().len(),
            21,
            "Expected 21 total operations"
        );
    }

    #[test]
    fn test_agent_is_metadata() {
        assert!(AISOperationType::Agent.is_metadata());
        assert!(!AISOperationType::Agent.is_public());
        assert!(!AISOperationType::Agent.is_internal());
    }

    #[test]
    fn test_const_str_is_internal() {
        assert!(AISOperationType::ConstStr.is_internal());
        assert!(!AISOperationType::ConstStr.is_public());
        assert!(!AISOperationType::ConstStr.is_metadata());
    }

    #[test]
    fn test_rsn_is_public() {
        assert!(AISOperationType::Rsn.is_public());
        assert!(!AISOperationType::Rsn.is_internal());
        assert!(!AISOperationType::Rsn.is_metadata());
    }

    #[test]
    fn test_mlir_mnemonics() {
        assert_eq!(AISOperationType::Agent.mlir_mnemonic(), "agent");
        assert_eq!(AISOperationType::Rsn.mlir_mnemonic(), "rsn");
        assert_eq!(AISOperationType::QMem.mlir_mnemonic(), "qmem");
        assert_eq!(AISOperationType::WaitAll.mlir_mnemonic(), "wait_all");
    }
}
